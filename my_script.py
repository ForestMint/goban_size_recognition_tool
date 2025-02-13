
print("Hello World!")

## import dependancies

import os
# this command makes that the Tensorlow warning is not written in console
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from PIL import Image
from random import randint
import keras
import tensorflow as tf
import numpy as np

print('Hello World!')


## declare lists to contain images
pics_19x19 = []
pics_13x13 = []
pics_9x9 = []

## fill the lists with images from the folders
for filename in os.listdir('./pictures/19x19'):
    my_image = Image.open('./pictures/19x19/'+filename)
    my_image = my_image.resize((25,25))
    pics_19x19.append(my_image)

for filename in os.listdir('./pictures/13x13'):
    my_image = Image.open('./pictures/13x13/'+filename)
    my_image = my_image.resize((25,25))
    pics_13x13.append(my_image)

for filename in os.listdir('./pictures/9x9'):
    my_image = Image.open('./pictures/9x9/'+filename)
    my_image = my_image.resize((25,25))
    pics_9x9.append(my_image)

## create empty train and test sets
x_train, y_train, x_test, y_test = [],[],[],[]


## for every image in each of the 3 lists, it will be sorted randomly to either
## the train set or the test set
for pic_19x19 in pics_19x19:
    res=randint(0,100)
    if res<80:
        x_train.append(pic_19x19)
        y_train.append("19x19")
    else:
        x_test.append(pic_19x19)
        y_test.append("19x19")

for pic_13x13 in pics_13x13:
    res=randint(0,100)
    if res<80:
        x_train.append(pic_13x13)
        y_train.append("13x13")
    else:
        x_test.append(pic_13x13)
        y_test.append("13x13")

for pic_9x9 in pics_9x9:
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

train_images, test_images = [], []
for image in x_train :
    train_images.append(keras.utils.img_to_array(x_train[0]))
for image in x_test :
    test_images.append(keras.utils.img_to_array(x_test[0]))


train_labels, test_labels = y_train, y_test
train_images, test_images = np.array(train_images), np.array(test_images)
'''
x = keras.utils.img_to_array(x_test[0])
print(x)
'''

print(train_images.shape)


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(25, 25)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

'''
model.fit(train_images, train_labels, epochs=4)



test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

## make predictions
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
#predictions[0]
'''