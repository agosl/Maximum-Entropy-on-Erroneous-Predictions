
from tensorflow.keras.models import model_from_json
import json




def save_model(modelInput, modelNameInput='model'):
	model_json = modelInput.to_json()
	model_json = json.loads(model_json)
	model_json['class_name'] = 'Model' # this attribute sometimes is not properly set
	model_json = json.dumps(model_json)
	with open(modelNameInput+".json","w") as json_file:
		json_file.write(model_json)
	modelInput.save_weights(modelNameInput+".h5")
	print("Saved "+modelNameInput)

def load_model(modelNameInput = 'model'):
	# load json and create model
	json_file = open(modelNameInput+'.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights(modelNameInput+".h5")
	return loaded_model
	print("Loaded model from disk")

