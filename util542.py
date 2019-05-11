import os
import cv2
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plot
from random import randint
from keras.models import Model
from keras.models import load_model
import pandas as pd


def get_images(directory):
    Images = []
    Labels = []  # 0 for Building , 1 for forest, 2 for glacier, 3 for mountain, 4 for Sea , 5 for Street
    label = 0
    
    for labels in os.listdir(directory): #Main Directory where each class label is present as folder name.
        if labels == 'buildings':
            label = [1,0,0,0,0,0]
        elif labels == 'forest':
            label = [0,1,0,0,0,0]
        elif labels == 'glacier':
            label = [0,0,1,0,0,0]
        elif labels == 'mountain':
            label = [0,0,0,1,0,0]
        elif labels == 'sea':
            label = [0,0,0,0,1,0]
        elif labels == 'street':
            label = [0,0,0,0,0,1]
        
        for image_file in os.listdir(directory+labels): #Extracting the file name of the image from Class Label folder
            image = cv2.imread(directory+labels+r'/'+image_file) #Reading the image (OpenCV)
            image = cv2.resize(image,(224,224)) #Resize the image, Some images are different sizes. (Resizing is very Important)
            Images.append(image)
            Labels.append(label)
    
    return shuffle(Images,Labels,random_state=817328462) #Shuffle the dataset you just prepared.

def get_classlabel(label_array):
    labels = {0:'buildings', 1:'forest', 2:'glacier', 3:'mountain', 4:'sea', 5:'street'}
    index = np.where(np.array(label_array)==1)
    
    return labels[index[0][0]]

def get_class_fromcode(class_code):
    labels = {2:'glacier', 4:'sea', 0:'buildings', 1:'forest', 5:'street', 3:'mountain'}
    
    return labels[class_code]

def loadTrainData():
    Images, Labels = get_images('intel-image-classification/seg_train/')
    Images = np.array(Images)
    Labels = np.array(Labels)

    print("Shape of Images:",Images.shape)
    print("Shape of Labels:",Labels.shape)
    f,ax = plot.subplots(5,5) 
    f.subplots_adjust(0,0,3,3)
    for i in range(0,5,1):
        for j in range(0,5,1):
            rnd_number = randint(0,len(Images))
            ax[i,j].imshow(Images[rnd_number])
            ax[i,j].set_title(get_classlabel(Labels[rnd_number]))
            ax[i,j].axis('off')
    
    return Images, Labels

def getTrainHistory(trained, modelName):
    plot.plot(trained.history['acc'])
    plot.plot(trained.history['val_acc'])
    plot.title('Model accuracy')
    plot.ylabel('Accuracy')
    plot.xlabel('Epoch')
    plot.legend(['Train', 'Test'], loc='upper left')
    plot.savefig(modelName)
    plot.show()

    plot.plot(trained.history['loss'])
    plot.plot(trained.history['val_loss'])
    plot.title('Model loss')
    plot.ylabel('Loss')
    plot.xlabel('Epoch')
    plot.legend(['Train', 'Test'], loc='upper left')
    plot.savefig(modelName)
    plot.show()

def testModel(Modelfilename):
    model = load_model(Modelfilename)
    test_images,test_labels = get_images('intel-image-classification/seg_test/')
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    model.evaluate(test_images,test_labels, verbose=1)

    # getCM
    predicted = model.predict(test_images)
    
    tlabels = []
    for label in test_labels:
        tlabel = get_classlabel(label)
        tlabels.append(tlabel)
    true_labels = np.array(tlabels)
    true_labels.shape

    plabels = []
    for i in range(len(predicted)):
        Parr = predicted[i]
        plabelcode = np.where(Parr==np.max(Parr))
        plabels.append(get_class_fromcode(plabelcode[0][0]))
    pred_labels = np.array(plabels)
    pred_labels.shape

    pd.crosstab(true_labels, pred_labels, rownames=['Real'], colnames=['Results'])

def res(....)

def Xc(..)

def XX()

def train( mod = res)
    mod()...

trained = train(mod = Xc)