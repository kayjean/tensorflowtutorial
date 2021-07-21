
'''
PREDICT
'''

import numpy as np
import keras
from keras.models import load_model
from keras.preprocessing import image

model = keras.models.load_model('glue_vgg16.h5')

def load_image_as_tensor(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.
    return img

def predict(img_path):
    img = load_image_as_tensor(img_path)
    prob = model.predict(img)
    classes = np.argmax(prob, axis=1)
    label_map = (validation_generator.class_indices)
    print (img_path)
    print (prob, classes, label_map)

predict('test/0009.jpg') 
predict('test/0023.jpg')

