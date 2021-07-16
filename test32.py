import warnings
warnings.filterwarnings("always")

import numpy as np, matplotlib.pyplot as plt
import keras

'''
LOAD PRE-TRAINED MODEL
'''

from keras.applications import VGG16

#Load the VGG model
vgg16 = VGG16(include_top=False, input_shape=(224, 224, 3)) # load vgg16 with all defaults except top

'''
MARK ALL LAYERS AS NON-TRAINABLE SO TRAINING DOESN'T ALTER WEIGHTS
'''

for layer in vgg16.layers:
    layer.trainable = False

print (vgg16.summary())


'''
TUNE MODEL TO GLUE NEEDS
'''

from keras import layers

# my_model_output = vgg16.layers[10].output # Only use the first three blocks of convolution
my_model_output = vgg16.output # use all 5 conv layers of vgg
my_model_output = layers.GlobalAveragePooling2D()(my_model_output) # Then add a GlobalAveragePooling layer to get 1d representation
# my_model_output = layers.Dense(128, activation='relu')(my_model_output) # add a Dense layer just to increase trainable params
# my_model_output = layers.Dropout(0.5)(my_model_output) # Add regularization to help decrease overfitting data
my_model_output = layers.Dense(2, activation='softmax')(my_model_output) # binary classification prediction layer

my_model = keras.models.Model(inputs=vgg16.input, outputs=my_model_output) #created my neural network

my_model.summary()

'''
SET UP TRAINING AND VALIDATION DATA
'''

from keras.preprocessing.image import ImageDataGenerator, load_img

train_datagen = ImageDataGenerator(
                            rescale=1./255,
                            rotation_range=20,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            horizontal_flip=True,
                            vertical_flip=True,
                            fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_batchsize = 20
val_batchsize = 5

train_generator = train_datagen.flow_from_directory(
                                                'train_data',
                                                target_size=(224, 224),
                                                batch_size=train_batchsize,
                                                class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
                                                    'validation_data',
                                                    target_size=(224, 224),
                                                    batch_size=val_batchsize,
                                                    class_mode='categorical',
                                                    shuffle=False)

'''
COMPILE AND TRAIN MODEL
'''

# Compile
try:
    my_model = keras.models.load_model('glue_vgg16.h5') # trains from last epoch if model exists
except:
    print ("Training new model...")
    my_model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['acc'])

# Train
checkpoint = keras.callbacks.ModelCheckpoint('glue_vgg16.h5', monitor='val_loss', verbose=1, period=1)

my_model_hist = my_model.fit_generator(
                            train_generator,
                            steps_per_epoch=2*train_generator.samples/train_generator.batch_size ,
                            epochs=25,
                            validation_data=validation_generator,
                            validation_steps=validation_generator.samples/validation_generator.batch_size,
                            verbose=1,
                            callbacks=[checkpoint])

# Save the Model
my_model.save('glue_vgg16.h5')
