
## ------------------- IMPORT PACKAGES ----------------------------------------

import os
# this command makes that the Tensorlow warning is not written in console
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from PIL import Image
from random import randint
import keras
import tensorflow as tf
import numpy as np
from numpy import asarray
import matplotlib.pyplot as plt
import configparser
config = configparser.ConfigParser()
config.read('config.ini')

## ------------------- DECLARE FUNCTIONS -------------------------------

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

## ------------------- MANAGE NP.ARRAYS BASED ON FOLDERS ---------------

# Get the list of all files and directories 
path = "./pictures"
dir_list = os.listdir(path) 

nb_of_pictures = 0
for dir in dir_list:
    nb_of_pictures += len(os.listdir("./pictures/" + dir))

labels_dict = {}

data_set = np.zeros(shape=(nb_of_pictures, int(config['IMAGE_PROCESSING']['image_size']), int(config['IMAGE_PROCESSING']['image_size'])))
label_set = np.zeros(shape=(nb_of_pictures))

counter = 0
item_counter = 0
for dir in dir_list:
    counter += 1
    labels_dict[dir] = counter-1

    for filename in os.listdir("./pictures/" + dir):

        my_image = Image.open("./pictures/" + dir + "/" +filename)
        my_image = my_image.convert('1') 
        my_image = my_image.resize((int(config['IMAGE_PROCESSING']['image_size']),int(config['IMAGE_PROCESSING']['image_size'])))
        numpydata = asarray(my_image)
        #print(numpydata.shape)
        data_set[item_counter] = numpydata
        label_set[item_counter] = counter-1
        item_counter += 1


# shuffle data set and label set with the same order
data_set, label_set = unison_shuffled_copies(data_set, label_set)

limit = round(len(data_set)*int(config['TRAINING_RULES']['percentage_of_pictures_in_the_training_set'])/100)

train_images = data_set[:limit]
train_labels = label_set[:limit]

test_images = data_set[limit:]
test_labels = label_set[limit:]

## ------------------- DEFINE, COMPILE AND FIT MODEL ---------------

# Define Sequential model with 3 layers
goban_size_recognition_model = tf.keras.Sequential([
    # input layer
    tf.keras.layers.Flatten(input_shape=(int(config['IMAGE_PROCESSING']['image_size']), int(config['IMAGE_PROCESSING']['image_size']))),
    # hidden layer
    tf.keras.layers.Dense(128, activation='relu'),
    # output layer
    tf.keras.layers.Dense(3)
])

goban_size_recognition_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

goban_size_recognition_model.fit(train_images, train_labels, epochs=int(config['NEURAL_NETWORK']['number_of_epochs']))

## ------------------- MEASURE THE ACCURACY OF THE MODEL ---------------

test_loss, test_acc = goban_size_recognition_model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

## make predictions
probability_model = tf.keras.Sequential([goban_size_recognition_model, 
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
print(predictions[0])
#print(predictions.shape)
