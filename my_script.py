
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


labels_dict = {"19x19": 1, "13x13": 2, "9x9" : 3}

## declare lists to contain images
pics_19x19 = []
pics_13x13 = []
pics_9x9 = []

## fill the lists with images from the folders
for filename in os.listdir('./pictures/19x19'):
    my_image = Image.open('./pictures/19x19/'+filename)
    my_image = my_image.convert('1') 
    my_image = my_image.resize((int(config['IMAGE_PROCESSING']['image_size']),int(config['IMAGE_PROCESSING']['image_size'])))
    numpydata = asarray(my_image)
    #print(numpydata.shape)
    pics_19x19.append(numpydata)

for filename in os.listdir('./pictures/13x13'):
    my_image = Image.open('./pictures/13x13/'+filename)
    my_image = my_image.convert('1') 
    my_image = my_image.resize((int(config['IMAGE_PROCESSING']['image_size']),int(config['IMAGE_PROCESSING']['image_size'])))
    numpydata = asarray(my_image)
    #print(numpydata.shape)
    pics_13x13.append(numpydata)

for filename in os.listdir('./pictures/9x9'):
    my_image = Image.open('./pictures/9x9/'+filename)
    my_image = my_image.convert('1') 
    my_image = my_image.resize((int(config['IMAGE_PROCESSING']['image_size']),int(config['IMAGE_PROCESSING']['image_size'])))
    numpydata = asarray(my_image)
    #print(type(numpydata))
    #print(numpydata.shape)
    pics_9x9.append(numpydata)

nd_array_19x19 = np.zeros(shape=(len(pics_19x19), int(config['IMAGE_PROCESSING']['image_size']), int(config['IMAGE_PROCESSING']['image_size'])))
nd_array_13x13 = np.zeros(shape=(len(pics_13x13), int(config['IMAGE_PROCESSING']['image_size']), int(config['IMAGE_PROCESSING']['image_size'])))
nd_array_9x9 = np.zeros(shape=(len(pics_9x9), int(config['IMAGE_PROCESSING']['image_size']), int(config['IMAGE_PROCESSING']['image_size'])))


for counter in range(len(pics_19x19)):
    nd_array_19x19[counter] = pics_19x19[counter]

for counter in range(len(pics_13x13)):
    nd_array_13x13[counter] = pics_13x13[counter]

for counter in range(len(pics_9x9)):
    nd_array_9x9[counter] = pics_9x9[counter]

'''
print(nd_array_19x19.shape)
print(nd_array_13x13.shape)
print(nd_array_9x9.shape)
'''


## create empty train and test sets
x_train, y_train, x_test, y_test = [],[],[],[]


## for every image in each of the 3 lists, it will be sorted randomly to either
## the train set or the test set
for pic_19x19 in nd_array_19x19:
    res=randint(0,100)
    if res<80:
        x_train.append(pic_19x19)
        y_train.append("19x19")
    else:
        x_test.append(pic_19x19)
        y_test.append("19x19")

for pic_13x13 in nd_array_13x13:
    res=randint(0,100)
    if res<80:
        x_train.append(pic_13x13)
        y_train.append("13x13")
    else:
        x_test.append(pic_13x13)
        y_test.append("13x13")

for pic_9x9 in nd_array_9x9:
    res=randint(0,100)
    if res<80:
        x_train.append(pic_9x9)
        y_train.append("9x9")
    else:
        x_test.append(pic_9x9)
        y_test.append("9x9")



'''
print(y_train)
print(y_test)
'''

'''
## let's display some images
x_train[0].show()
x_test[0].show()
'''

#train_images, test_images = [], []

train_images = np.zeros(shape=(len(x_train), int(config['IMAGE_PROCESSING']['image_size']), int(config['IMAGE_PROCESSING']['image_size'])))
test_images = np.zeros(shape=(len(x_test), int(config['IMAGE_PROCESSING']['image_size']), int(config['IMAGE_PROCESSING']['image_size'])))


for counter in range(len(x_train)) :
    #train_images.append(keras.utils.img_to_array(x_train[0]))
    train_images [counter] = (x_train[counter])
for counter in range(len(x_test)) :
    #test_images.append(keras.utils.img_to_array(x_test[0]))
    train_images [counter] = (x_test[counter])



train_labels, test_labels = y_train, y_test


#train_images, test_images = np.array(train_images), np.array(test_images)


'''
x = keras.utils.img_to_array(x_test[0])
print(x)
'''

'''
train_images = train_images / 255.0

test_images = test_images / 255.0
'''

train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)


for counter in range(len(train_labels)):
    train_labels[counter] = labels_dict[train_labels[counter]]
#train_labels = list(map(int, train_labels))
train_labels = train_labels.astype(int)

for counter in range(len(test_labels)):
    test_labels[counter] = labels_dict[test_labels[counter]]
#train_labels = list(map(int, train_labels))
test_labels = test_labels.astype(int)

print(train_images.shape)
print(len(train_labels))

print(type(train_images))
print(type(train_labels))

print(train_images)
print(train_labels)



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

'''

## make predictions
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
#predictions[0]
'''