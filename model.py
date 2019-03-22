import cv2
import csv
#import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import sklearn
# import tensorflow as tf
from sklearn.model_selection import train_test_split


lines = []
lines1 = []
lines2 = []
meaurements = []
# read counter clockwise data run
with open('./data5/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
# clockwise data run        
with open('./data6/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines1.append(line)
# bridge data to improve vehicle ability to cross the bridge        
with open('./data8/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines2.append(line)
        
images = []
measurements = []
count = 0
for line in lines:
    for i in range(3):
        source_path = line[i]   # read the middle, left, right images
        source_path=source_path.replace("\\","/")
        filename = source_path.split('/')[-1]
        count +=1
        print(count, "   CCW file data5 Appd",end=" \r",)
        current_path='./data5/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        
        correction = 0.25 
        measurement = float(line[3])
        # Center CAM
        if i == 0 :
            measurements.append(measurement)
        # Left CAM
        elif i == 1 :
            measurements.append(measurement+correction)
        #  Right CAM    
        else:
            measurements.append(measurement-correction)
    
print("")      
for line in lines1:
    for i in range(3):
        source_path = line[i]   # read the middle, left, right images
        source_path=source_path.replace("\\","/")
        filename = source_path.split('/')[-1]
        count +=1
        print(count, "   CW file data6 Appd",end=" \r",)
        current_path='./data6/IMG/' + filename  
        image = cv2.imread(current_path)
        images.append(image)
         
        correction = 0.25 
        measurement = float(line[3])
        # Center CAM
        if i == 0 :
            measurements.append(measurement)
        # Left CAM  
        elif i == 1 :    
            measurements.append(measurement+correction)
        #  Right CAM   
        else:    
            measurements.append(measurement-correction)
    
print("")      
for line in lines2:
    for i in range(3):
        source_path = line[i]   # read the middle, left, right images
        source_path=source_path.replace("\\","/")
        filename = source_path.split('/')[-1]
        count +=1
        print(count, "   CCW file data8 bridge Appd",end=" \r",)
        current_path='./data8/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
         
        correction = 0.25 
        measurement = float(line[3])
        # Center CAM
        if i == 0 :
            measurements.append(measurement)
        # Left CAM
        elif i == 1 :    
            measurements.append(measurement+correction)
        #  right CAM    
        else:
            measurements.append(measurement-correction)    

augment_images, augmented_measurements = [],[]
print("")
# augment images
for image,measurement in zip(images, measurements):
    count +=1
    print(count, "   Aug file Appd",end=" \r",)
    augment_images.append(image)
    augmented_measurements.append(measurement)
    augment_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)

print("") 
 # training
X_train = np.array(augment_images)
y_train = np.array(augmented_measurements)   

import keras

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

model = Sequential()

# test 
#model.add(Flatten(input_shape=(160,320,3)))
#model.add(Dense(1))

# real model
# Normalize image
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(3,160,320)))
model.add(Convolution2D(24,5,5, subsample=(2,2)))
# Dropout
model.add(Dropout(0.5))
# Activation Layer
model.add(Activation('relu'))
# Convolution Layer 2
model.add(Convolution2D(36,5,5, subsample=(2,2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
# Convolution Layer 3         
model.add(Convolution2D(48,5,5, subsample=(2,2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
# Convolution Layer 4         
model.add(Convolution2D(64,3,3, subsample=(1,1)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
# Convolution Layer 5          
model.add(Convolution2D(24,3,3, subsample=(1,1)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
# Flatten to single 1-D array
model.add(Flatten())
# Fully connected Layer 1          
model.add(Dense(100))
# Fully connected Layer 2          
model.add(Dense(50))
# Fully connected Layer 3          
model.add(Dense(10))
 # fully connected Layer 4         
model.add(Dense(1))

adam = Adam(lr=1e-5)
model.compile(loss = 'mse',optimizer = 'adam',metrics =['accuracy'])
hist_obj = model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, epochs = 30)

model.save('model.h5')
    
model.summary()          

    
    