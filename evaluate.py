import os
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


## List to save metrics
ACC = np.zeros(10)
MCC = np.zeros(10)
cum_cm = np.zeros((12, 12))
test_datagen = ImageDataGenerator(rescale=1.0/255)

## Evaluate the best model for each fold experiment
for i in range(1,11):

	## Load the best model
	best_model_filename = f"Models/best_model_fold_{i}.h5"
	best_model = load_model(best_model_filename) 

	## Test data
	test_generator = test_datagen.flow_from_directory(f"Dataset/fold-{i}/Test", 
	                                                  target_size=(224, 224),
	                                                  batch_size=1,
	                                                  class_mode='categorical',
	                                                  shuffle=False
	                                                  )


	## True labels and classes' name  
	Y = test_generator.classes
	classes_map = test_generator.class_indices
	class_names = [k for k, v in sorted(classes_map.items(), key=lambda item: item[1])] 

	## get model predictions 
	Ypp = best_model.predict(test_generator)
	Ypp = np.argmax(Ypp,axis = 1)

	## Get metrics and confusion matrix
	CM = metrics.confusion_matrix(Y,Ypp)
	mcc = metrics.matthews_corrcoef(Y,Ypp)
	acc = metrics.accuracy_score(Y,Ypp)


	cum_cm += CM
	ACC[i-1] = acc
	MCC[i-1] = mcc

## Get the mean of 10 cross-validation experiments (Confusion matrix)
sum_cm = cum_cm.sum(axis = 1, keepdims =True)
cm = cum_cm/sum_cm


## Rename classes to display confusion matrix
for i in range(12):
	if class_names[i] == 'Citrus_leafminer_':
		class_names[i] = 'CLM'

	elif class_names[i] == 'Greasy_spot_':
		class_names[i] = 'GS'

	elif class_names[i] ==  'Red_scale_':
		class_names[i] = 'RS'

	elif class_names[i] == 'Red_scale_sequelae_':
		class_names[i] = 'RSS'

	elif class_names[i] == 'Texas_mite_':
		class_names[i] = 'TM'

	class_names[i] = class_names[i][:-1]


## Print metrics for each fold
for i in range(len(ACC)):
	print(f'Fold-{i}  Accuracy: {ACC[i]} MCC: {MCC[i]}')

## Get the mean of 10 cross-validation experiments (MCC & ACC)
print(f'Average Accuracy: {np.mean(ACC)}      Average MCC: {np.mean(MCC)}')

## Display confusion matrix
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = class_names)
cm_display.plot()
plt.show()

