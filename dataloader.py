import pdb
import pandas as pd
import h5py
import math
import numpy as np
import nibabel as nib
from image_augmentation import *
import tensorflow

def one_hot_labels(data, n_labels, labels=None):
    new_shape = [data.shape[0] , data.shape[1], data.shape[2], n_labels]
    y = np.zeros(new_shape, np.int8)
    for label_index in range(n_labels):
        if labels is not None:
            y[:, :,:,label_index][data == labels[label_index]] = 1

    return y

class LAHeart(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self,base_dir=None,batch_size=4, split='train',patch_size=None, num=None,shuffle=True, random_crop_flag=1,center_crop_flag=0,random_rotflip_flag=0):
        'Initialization'
        self._base_dir = base_dir
        self.center_crop_flag = center_crop_flag
        self.random_crop_flag = random_crop_flag
        self.random_rotflip_flag=random_rotflip_flag
        self.batch_size = batch_size
        self.split=split
        self.shuffle = shuffle
        self.patch_size=patch_size
        self.sample_list = []
 
        if split=='train':
            with open(self._base_dir+'/train.list', 'r') as f:
                self.image_list = f.readlines()

        elif split == 'test':
            with open(self._base_dir+'/test.list', 'r') as f:
                self.image_list = f.readlines()

        elif split == 'val':
            with open(self._base_dir+'/val.list', 'r') as f:
                self.image_list = f.readlines()

        self.image_list = [item.replace('\n','') for item in self.image_list]

        if num is not None:
            self.image_list = self.image_list[:num]

        print("total {} samples".format(len(self.image_list)))
        print("total {} batches".format((self.__len__())))
        self.on_epoch_end()

    def __len__(self):
        return (len(self.image_list) // self.batch_size)

    def __getitem__(self, idx):
        

        batch = self.image_list[idx * self.batch_size:(idx + 1) * self.batch_size]
        X, y , names= self.__get_data(batch)

        return X, y , names

    def __get_data(self, batch):

        X = np.empty((self.batch_size,self.patch_size[0],self.patch_size[1],self.patch_size[2]))
        y =np.empty((self.batch_size,self.patch_size[0],self.patch_size[1],self.patch_size[2],2))
        
        for i, image_name in enumerate(batch):

          
            h5f = h5py.File(self._base_dir+"/2018LA_Seg_Training_Set/"+image_name+"/mri_norm2.h5", 'r')
            image = h5f['image'][:]
            label = h5f['label'][:]
   
   
            im,label = augment(image,label, output_size = self.patch_size, 
                                center_crop_flag=self.center_crop_flag,
				random_crop_flag=self.random_crop_flag, 
				random_rotflip_flag=self.random_rotflip_flag,
                                resize_flag=0)


            if self.split =='test':
            	X=im
            	y=label
            	return X,y,batch
            else:
            	labels=one_hot_labels(label,2,[0,1])	
            	X[i,:]=im
            	y[i,:]=labels
        
        return X,y,batch



    def on_epoch_end(self):
        if self.shuffle == True:
            random.shuffle(self.image_list)
        print('Shuffling data......')

class LAHeart_test(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self,base_dir=None, patch_size=None, num=None,shuffle=True, random_crop_flag=1,center_crop_flag=0,random_rotflip_flag=0):
        'Initialization'
        self._base_dir = base_dir
        self.center_crop_flag = center_crop_flag
        self.random_crop_flag = random_crop_flag
        self.random_rotflip_flag=random_rotflip_flag
        self.batch_size = 1
        self.shuffle = shuffle
        self.patch_size=patch_size
        self.sample_list = []

        with open(self._base_dir+'/test.list', 'r') as f:
        	self.image_list = f.readlines()


        self.image_list = [item.replace('\n','') for item in self.image_list]

        if num is not None:
            self.image_list = self.image_list[:num]

        print("total {} samples".format(len(self.image_list)))
        print("total {} batches".format((self.__len__())))
       
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        h5f = h5py.File(self._base_dir+"/2018LA_Seg_Training_Set/"+self.image_list[idx]+"/mri_norm2.h5", 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]


        X,y = augment(image,label, output_size = self.patch_size, 
                                center_crop_flag=self.center_crop_flag,
				random_crop_flag=self.random_crop_flag, 
				random_rotflip_flag=self.random_rotflip_flag,
                                resize_flag=0)


        print('name . ',self.image_list[idx])


        return X,y,self.image_list[idx]


class LAHeart_val(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self,base_dir=None, patch_size=None, num=None,shuffle=True, random_crop_flag=1,center_crop_flag=0,random_rotflip_flag=0):
        'Initialization'
        self._base_dir = base_dir
        self.center_crop_flag = center_crop_flag
        self.random_crop_flag = random_crop_flag
        self.random_rotflip_flag=random_rotflip_flag
        self.batch_size = 1
        self.shuffle = shuffle
        self.patch_size=patch_size
        self.sample_list = []
        #print('......bz: ',self.batch_size)
	# base_dir ---> carpeta donde están guardadas las imágenes y las listas 

        with open(self._base_dir+'/val.list', 'r') as f:
        	self.image_list = f.readlines()


        self.image_list = [item.replace('\n','') for item in self.image_list]

        if num is not None:
            self.image_list = self.image_list[:num]

        print("total {} samples".format(len(self.image_list)))
        print("total {} batches".format((self.__len__())))
        #self.on_epoch_end()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        h5f = h5py.File(self._base_dir+"/2018LA_Seg_Training_Set/"+self.image_list[idx]+"/mri_norm2.h5", 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]


        X,y = augment(image,label, output_size = self.patch_size, 
                                center_crop_flag=self.center_crop_flag,
				random_crop_flag=self.random_crop_flag, 
				random_rotflip_flag=self.random_rotflip_flag,
                                resize_flag=0)


        print('name . ',self.image_list[idx])


        return X,y,self.image_list[idx]



		           
