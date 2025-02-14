
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
    labels_dict[dir] = counter

    for filename in os.listdir("./pictures/" + dir):

        my_image = Image.open("./pictures/" + dir + "/" +filename)
        my_image = my_image.convert('1') 
        my_image = my_image.resize((int(config['IMAGE_PROCESSING']['image_size']),int(config['IMAGE_PROCESSING']['image_size'])))
        numpydata = asarray(my_image)
        #print(numpydata.shape)
        data_set[item_counter] = numpydata
        label_set[item_counter] = counter
        item_counter += 1


# shuffle data set and label set with the same order
data_set, label_set = unison_shuffled_copies(data_set, label_set)

limit = round(len(data_set)*int(config['TRAINING_RULES']['percentage_of_pictures_in_the_training_set'])/100)

train_images = data_set[:limit]
train_labels = label_set[:limit]

test_images = data_set[limit:]
test_labels = label_set[limit:]

print("data_set.shape : ")
print(data_set.shape)

print("train_images.shape : ")
print(train_images.shape)

print("test_images.shape : ")
print(test_images.shape)

print("labels_set.shape : ")
print(label_set.shape)

print("train_labels.shape : ")
print(train_labels.shape)

print("test_labels.shape : ")
print(test_labels.shape)



#plt.imshow(train_images[0])


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(int(config['IMAGE_PROCESSING']['image_size']), int(config['IMAGE_PROCESSING']['image_size']))),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


model.fit(train_images, train_labels, epochs=int(config['NEURAL_NETWORK']['number_of_epochs']))



test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)





## make predictions
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
print(predictions[0])
print(predictions)

