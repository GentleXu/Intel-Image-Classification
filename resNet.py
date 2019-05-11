
#%%
import tensorflow.keras.layers as Layers
import tensorflow.keras.activations as Actications
import tensorflow.keras.models as Models
import tensorflow.keras.optimizers as Optimizer
import tensorflow.keras.metrics as Metrics
import tensorflow.keras.utils as Utils
from keras.utils.vis_utils import model_to_dot
import os
import matplotlib.pyplot as plot
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix as CM
from random import randint
from IPython.display import SVG
import matplotlib.gridspec as gridspec

from keras.models import Model
from keras.models import load_model
from keras.applications import ResNet50
from keras.applications import imagenet_utils
from keras.layers import Dense,GlobalAveragePooling2D
from keras.regularizers import l2
from keras.initializers import he_normal


#%%
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


#%%
Images, Labels = get_images('intel-image-classification/seg_train/') #Extract the training images from the folders.

Images = np.array(Images) #converting the list of images to numpy array.
Labels = np.array(Labels)


#%%
print("Shape of Images:",Images.shape)
print("Shape of Labels:",Labels.shape)


#%%
f,ax = plot.subplots(5,5) 
f.subplots_adjust(0,0,3,3)
for i in range(0,5,1):
    for j in range(0,5,1):
        rnd_number = randint(0,len(Images))
        ax[i,j].imshow(Images[rnd_number])
        ax[i,j].set_title(get_classlabel(Labels[rnd_number]))
        ax[i,j].axis('off')


#%%
base_model = ResNet50(include_top=False, weights='imagenet',input_shape = (224, 224, 3), pooling='avg')


#%%
features = base_model.output


#%%
regularizer = 1e-5


#%%
predFC = Dense(6,activation='sigmoid', name='multi-labels', 
                kernel_regularizer=l2(regularizer),
                bias_regularizer=l2(regularizer),
                kernel_initializer=he_normal())(features)


#%%
model = Model(inputs = base_model.input, outputs = predFC)


#%%
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics= ['accuracy'], loss_weights = [2.0])


#%%
trained = model.fit(Images,Labels,batch_size = 16, epochs=35,validation_split=0.30)
model.save('resNetmodel.h5')


#%%
plot.plot(trained.history['acc'])
plot.plot(trained.history['val_acc'])
plot.title('Model accuracy')
plot.ylabel('Accuracy')
plot.xlabel('Epoch')
plot.legend(['Train', 'Test'], loc='upper left')
plot.savefig('resNetTrainAcc')
plot.show()

plot.plot(trained.history['loss'])
plot.plot(trained.history['val_loss'])
plot.title('Model loss')
plot.ylabel('Loss')
plot.xlabel('Epoch')
plot.legend(['Train', 'Test'], loc='upper left')
plot.savefig('resNetTrainLoss')
plot.show()

#%%
model = load_model('Xception_withoutDataAug_model.h5')
test_images,test_labels = get_images('intel-image-classification/seg_test/')
test_images = np.array(test_images)
test_labels = np.array(test_labels)
model.evaluate(test_images,test_labels, verbose=1)


#%%
predicted = model.predict(test_images, verbose = 1)

#%%
tlabels = []
for label in test_labels:
    tlabel = get_classlabel(label)
    tlabels.append(tlabel)
true_labels = np.array(tlabels)
# true_labels.shape

#%%
plabels = []
for i in range(len(predicted)):
    Parr = predicted[i]
    plabelcode = np.where(Parr==np.max(Parr))
    plabels.append(get_class_fromcode(plabelcode[0][0]))
pred_labels = np.array(plabels)
# np.where(pred_labels=='street')
#%%


#%%
import pandas as pd

tab = pd.crosstab(true_labels, pred_labels, rownames=['Real'], colnames=['Results'])
print(tab)


base = np.sum(tab.values, axis = 1)

#%%
labelAcc = []
labels = []
for i in range(0,6):
    acc = tab.values[i][i]/base[i]
    # labelAcc[get_class_fromcode(i)] = acc
    labelAcc.append(acc)
    labels.append(get_class_fromcode(i))

labeltab = pd.DataFrame({'Label':labels,'Accuracy':labelAcc})
print(labeltab)