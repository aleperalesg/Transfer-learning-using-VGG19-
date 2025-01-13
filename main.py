import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

## function to create the model
def create_model(num_classes,lr,momentum):

    ## Load VGG19 model
    base_model = VGG19(weights = 'imagenet', include_top = False, input_shape = (224,224,3))

    ## Add ccustom layers
    x = base_model.output
    x = tf.keras.layers.Flatten(name ='flatten_custom')(x)
    x = tf.keras.layers.Dense(256,activation ='relu', name = 'dense_custom')(x)
    #x = tf.keras.layers.Dropout(0.5)(x)

    ## Set output layer and model
    output = tf.keras.layers.Dense(num_classes,activation = 'softmax', name = 'output_layer')(x)
    model = tf.keras.Model(inputs = base_model.input, outputs = output)

    ## Compile the model 

    model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate = lr, momentum = 0.9),
                 loss = 'categorical_crossentropy',
                 metrics = ['accuracy'])

    return model

## function to save the best model using the validation set
def set_checkpoint_callback(filename):

    ## Set checkpoint to save the best model using the validation set 
    checkpoint_callback = ModelCheckpoint(
        filename, 
        monitor='val_accuracy', 
        save_best_only=True,  
        mode='max',  
        verbose=1  
    )

    return checkpoint_callback


################################################################################################################################################

## Numuber of classes and list of curves
num_classes = 12


## Set hyperparameters 
lr = 0.0001
momentum = 0.9
epochs = 100

## k-fold
k = 10


## K-fold cross validation
for i in range(1,11):

    ## Load model
    model = create_model(num_classes,lr,momentum)

    ## Set checkpoint 
    checkpoint_filename = f"Models/best_model_fold_{i}.h5"
    checkpoint_callback = set_checkpoint_callback(checkpoint_filename)


    ## Prepare data
    ## Train data using online data augmentation and set evaluate data
    train_datagen = ImageDataGenerator(rescale=1.0/255, rotation_range=20, 
                                       width_shift_range=0.2, height_shift_range=0.2, 
                                       horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1.0/255)

    ## Get training and validation sets
    train_generator = train_datagen.flow_from_directory(f"Dataset/fold-{i}/Training",
                                                        target_size=(224, 224),
                                                        batch_size=16,
                                                        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(f"Dataset/fold-{i}/Validation",
                                                             target_size=(224, 224),
                                                             batch_size=16,
                                                             class_mode='categorical')


    ## Training model
    history = model.fit(train_generator,
                        epochs=epochs,
                        validation_data=validation_generator,        
                        callbacks=[checkpoint_callback])





