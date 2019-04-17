
import cv2
import csv
#import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import sklearn
# import tensorflow as tf
from sklearn.model_selection import train_test_split


correction = 0.2 # 0.1 0.25 parameter to tune
del_rate =  0.1 # 0.2 0.4 0.8

cut_value = .1   # 0.02  0.5

def RandomBrightness(image):
    # convert to HSV 
    RandomImage = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    # randomly generate brightness value
    # define range dark to bright
    random_bright = np.random.uniform(0.25,1.0)
    # Apply the brightness to V channel
    RandomImage[:,:,2] = RandomImage[:,:,2]*random_bright
    # back to RGB
    RandomImage = cv2.cvtColor(RandomImage,cv2.COLOR_HSV2RGB)
    return RandomImage

lines = []
lines1 = []
 

meaurements = []
#udacity data
with open('./datatrainData/data0/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)        

        
                
        
images = []
measurements = []
count = 0
for line in lines:
    for i in range(3):
       
        source_path = line[i]   # read the middle, left, right images limit 2^15
        source_path=source_path.replace("\\","/")
        filename = source_path.split('/')[-1]
        count +=1
        print(count,end="\r",)
        current_path='./datatrainData/data0/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        
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

    
                 
augment_images, augmented_measurements = [],[]
print("")
for image,measurement in zip(images, measurements):
    if abs(measurement) > cut_value or np.random.random() > del_rate:
        if image is not None:
            image = RandomBrightness(image) # random brightness
        count +=1
        print(count,end="\r",)
        augment_images.append(image)
        augmented_measurements.append(measurement)
        augment_images.append(cv2.flip(image,1))
        augmented_measurements.append(measurement*-1.0)

print("") 
 # training   limit  2^16
X_train = np.array(augment_images)   
y_train = np.array(augmented_measurements)   

print("Loading Done")
  
        



import keras

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation,ELU
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.models import load_model
#from keras.utils.visualize_util import plot
from keras.utils import plot_model
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

def resize(img):
    import tensorflow
    return tensorflow.image.resize_images(img,(60,120))

# network
model = Sequential()

# Crop 70 pixels from the top of the image and 25 from the bottom
model.add(Cropping2D(cropping=((75, 25), (0, 0)),
                     input_shape=(160, 320, 3),
                     data_format="channels_last"))

# Resize the data
model.add(Lambda(resize))

# Normalize the data
model.add(Lambda(lambda x: (x/127.5) - 0.5))

model.add(Conv2D(3, (1, 1), padding='same'))
model.add(ELU())

model.add(BatchNormalization())
model.add(Conv2D(16, (5, 5), strides=(2, 2), padding="same"))
model.add(ELU())

model.add(BatchNormalization())
model.add(Conv2D(32, (5, 5), strides=(2, 2), padding="same"))
model.add(ELU())

model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same"))
model.add(ELU())

model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), strides=(2, 2), padding="same"))
model.add(ELU())

model.add(Flatten())
model.add(Dropout(.5))
model.add(ELU())

model.add(Dense(512))

model.add(ELU())

model.add(Dense(100))
model.add(Dropout(.2))
model.add(ELU())

model.add(Dense(10))
model.add(Dropout(.5))
model.add(ELU())

model.add(Dense(1))

adam = Adam(lr=1e-5)
model.compile(loss = 'mse',optimizer = 'adam',metrics =['accuracy']) 

#earlystopper = EarlyStopping(patience =5, verbose =1)
#checkpointer = ModelCheckpoint('model.h5', monitor ='val_loss',verbose=1,save_best_only=True)

hist_obj = model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, epochs =11)
model.save('model.h5')
    
model.summary()          
print(model.summary())





