import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import  Conv3D, Activation, MaxPooling3D, Conv3DTranspose, Add,BatchNormalization, Dropout
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler, TensorBoard,  EarlyStopping
from tensorflow.keras.models import model_from_json
import numpy as np
import os
import math
from ResUnet import build_network
#from utils import*
from glob import glob
from random import shuffle, randint
import random
from numpy import linalg as LA
import keras.backend as K
from dataloader import *
from configparser import ConfigParser
from focal_loss import BinaryFocalLoss
focal_loss_keras=BinaryFocalLoss(gamma=2)
parser = ConfigParser()
parser.read('config_train.ini')

cce = CategoricalCrossentropy()
bce = BinaryCrossentropy()



def dice_coefficient(y_true, y_pred):
    y_true = K.cast(y_true,"float")
    intersection = tf.reduce_sum(y_pred[:, :, :,:,1] * (y_true[:, :, :,:,1]))
    union_prediction = tf.reduce_sum(y_pred[:, :, :,:,1])
    union_ground_truth = tf.reduce_sum(y_true[:, :, :,:,1])
    union = union_ground_truth + union_prediction
    dice_coef = (2 * intersection + K.epsilon()) / (union+ K.epsilon()) 
    return dice_coef


def DSL(y_true, y_pred,reg_function=None,u=0):
	dice=dice_coefficient(y_true, y_pred)
	if reg_function is None:
		return ( 1 - dice)
	else:
		reg=reg_function(y_true, y_pred)
		return ( 1 - (dice + u*reg))

def CE(y_true, y_pred,reg_function=None,u=0):
	y_true = K.cast(y_true,"float")	
	ce=bce(y_true, y_pred)
	if reg_function is None:
		return ce, 0
	else:
		reg=reg_function(y_true, y_pred)	
		return (ce - u*reg), reg
	

def ME(y_true, y_pred):
	en=entropy_coefficient(y_true, y_pred,EP=False)
	return en 

def MEEP(y_true, y_pred):
	en=entropy_coefficient(y_true, y_pred,EP=True)
	return en 

def MEEP_KL(y_true, y_pred):
 	y_pred = K.clip(y_pred, K.epsilon(), 1)
 	KL=KL_uniform(y_true, y_pred,EP=True)	
 	return -KL


def entropy_coefficient(y_true, y_pred,EP):
	y_pred_lm=K.argmax(y_pred, axis=-1)
	y_true_lm=K.argmax(y_true, axis=-1)
	if EP:
		misclassified_pixels=K.cast(K.not_equal(y_pred_lm,y_true_lm),float)
		entropy=K.mean(K.binary_crossentropy(y_pred, y_pred),axis=-1)*misclassified_pixels
		entropy=K.sum(entropy)/K.sum(misclassified_pixels)		
	else:
		entropy=K.mean(K.binary_crossentropy(y_pred, y_pred),axis=-1)
		entropy=K.mean(entropy)		 
	return entropy


def KL_uniform(y_true, y_pred,EP):
	y_pred_lm=K.argmax(y_pred, axis=-1)
	y_true_lm=K.argmax(y_true, axis=-1)
	if EP:
		misclassified_pixels=K.cast(K.not_equal(y_pred_lm,y_true_lm),float)
		kl=K.mean(K.log(y_pred),axis=-1)*misclassified_pixels 
		kl=K.sum(kl)/K.sum(misclassified_pixels)
	else: 
		kl=K.mean(K.log(y_pred),axis=-1)
		kl=K.mean(kl)	
	return -kl


def focal_loss(y_true, y_pred,u=0,full=False):
	y_true = K.cast(y_true,"float")	
	fl=focal_loss_keras(y_true[:, :, :,:,1],y_pred[:, :, :,:,1])
	return fl



def train_unet(dir_path,loss_fn,reg=None,u=None):
	if reg is None:
		u=0	
		model_fold=f"model_{loss_fn.__name__}"
	else:	
		model_fold=f"model_{loss_fn.__name__}+{reg.__name__}"

	patch_size = (144,144,80)

	train_gen = LAHeart(base_dir=dir_path,batch_size=batch_size,
		               split='train',patch_size=patch_size,random_rotflip_flag=1,random_crop_flag=1)
	val_gen = LAHeart(base_dir=dir_path,batch_size=1,
		               split='val',patch_size=patch_size,random_rotflip_flag=0,random_crop_flag=0,center_crop_flag=1)


	train_acc_dice = tf.keras.metrics.Mean()
	train_loss = tf.keras.metrics.Mean()
	train_reg = tf.keras.metrics.Mean()
	val_acc_dice = tf.keras.metrics.Mean()
	val_loss = tf.keras.metrics.Mean()


	def train_on_batch(data):
		x,y,name =data

		with tf.GradientTape() as tape:
	    		probs = cnn(x)
	    		loss,reg_coef = loss_fn(y,probs,reg,u=u)
	    		gradients = tape.gradient(loss, cnn.trainable_weights)

		optimizer.apply_gradients(zip(gradients, cnn.trainable_weights))
		train_loss.update_state(loss)
		train_reg.update_state(reg_coef)
		

	
	def val_on_batch(data):
		x,y,name=data
		probs = cnn(x)

		loss , _ = loss_fn(y, probs,reg,u=u)
		dice = dice_coefficient(y, probs)
		val_acc_dice.update_state(dice)
		val_loss.update_state(loss)
		return loss,dice

	
	inputs, outputs = build_network()
	cnn = keras.Model(inputs,outputs)
	
	initial_learning_rate = lr
	optimizer = tf.keras.optimizers.Adam(lr=initial_learning_rate)

	summary_writer = tf.summary.create_file_writer(os.path.join('./logs',model_fold))
	os.makedirs(os.path.join('./models',model_fold),exist_ok=True)

	
	best_val_acc = 0 
	for epoch in range(1,epochs):
		print('--- Epoch  ',epoch,' /',epochs )
		for data in train_gen:
			train_on_batch(data)

		if epoch % 1 == 0:
			for val_data in val_gen:
				loss,dc=val_on_batch(val_data)
		
			with summary_writer.as_default():

					tf.summary.scalar('train_loss', train_loss.result(),step=epoch)
					tf.summary.scalar('train_reg', train_reg.result(),step=epoch)
					tf.summary.scalar('val_loss',  val_loss.result(),step=epoch)	
					tf.summary.scalar('val_dice: ',val_acc_dice.result(),step=epoch)	
					tf.summary.scalar('Learning_rate: ',optimizer.learning_rate.numpy(),step=epoch)
					
			print('Epoch n: ',epoch, '  Validation dice coeff: ',val_acc_dice.result())


		if val_acc_dice.result()>best_val_acc:

			print('Saving model in epoch: ',epoch)
			os.makedirs(os.path.join('./models',model_fold),exist_ok=True)
			save_model(cnn,os.path.join('./models',model_fold))
			best_val_acc = val_acc_dice.result()


		if epoch % 10 == 0:
			optimizer.learning_rate.assign(optimizer.learning_rate*lrd)

		train_loss.reset_states()
		train_reg.reset_states()
		val_acc_dice.reset_states()
		val_loss.reset_states()   


	del unet
	del optimizer
	summary_writer.close()
	K.clear_session()



if __name__ == "__main__":

	
	dir_path="../data_LVHeart/"

	# Reproducible hiperparameters
	batch_size=2
	lrd=0.85
	lr=0.0001
	epochs=200
	network='ResUNet'


	lamb=0.3

	# Loss functions: DSL, CE, focal_loss
	# Regularization term: MEEP, MEEP_KL, EP

	loss=CE
	reg_function=MEEP_KL

	train_unet(dir_path,loss,reg_function,lamb)
