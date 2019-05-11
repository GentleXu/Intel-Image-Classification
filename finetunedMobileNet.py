# -*- coding:UTF-8 -*-

import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
import os
import sys

import keras
from keras.preprocessing import image
import numpy as np
from keras.applications import imagenet_utils
from PIL import Image

from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import ResNet50, MobileNet
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from keras.regularizers import l2
from keras.initializers import he_normal

from preprocess import pickProtest, train_aug_multitask, get_multitask_val_data, train_aug_slice, train_aug_multitask_slice


# Get the number of cores assigned to this job.
def get_n_cores():
   # On a login node run Python with:
   # export NSLOTS=4
   # python mycode.py
   #
   nslots = os.getenv('NSLOTS')
   if nslots is not None:
     return int(nslots)
   raise ValueError('Environment variable NSLOTS is not defined.')

# Get the Tensorflow backend session.
def get_session():
   try:
      nthreads = get_n_cores() - 1
      if nthreads >= 1:
         session_conf = tf.ConfigProto(
            intra_op_parallelism_threads=nthreads,
            inter_op_parallelism_threads=1,
            allow_soft_placement=True)
         return tf.Session(config=session_conf)
   except: 
      sys.stderr.write('NSLOTS is not set, using default Tensorflow session.\n')
      sys.stderr.flush()
   return ktf.get_session()

if os.getenv('NSLOTS') is not None:
    # Assign the configured Tensorflow session to keras
    ktf.set_session(get_session())
# Rest of your Keras script starts here....



EPOCHS = 100
batch_size = 32
regularizer = 1e-5

#base_model = ResNet50(weights='imagenet',include_top=False)
base_model = MobileNet(input_shape=(224,224,3), weights='imagenet', include_top=False, pooling='avg')

# MobileNet has 1024 features
features=base_model.output
#x1=GlobalAveragePooling2D(name='avg_pool_2')(features)
preds1=Dense(10,activation='sigmoid', name='multi-labels',
              kernel_regularizer=l2(regularizer),
              bias_regularizer=l2(regularizer),
              kernel_initializer=he_normal()
             )(features)  #final layer with sigmoid activation

preds2=Dense(1,activation='sigmoid', name='vio-regression',
             kernel_regularizer=l2(regularizer),
             bias_regularizer=l2(regularizer),
             kernel_initializer=he_normal()
             )(features)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=[preds1,preds2])

'''
# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False
# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss={'multi-label':'binary_crossentropy','vio-regression':'mean_squared_error'})
# simple test
from test import prepare_image
x = prepare_image('train-30511.jpg')
y = np.array([[1,1,0,0,0,0,0,0,0]])
model.fit(x,y,32,20)
# train the model on the new data for a few epochs
model.fit_generator(...)
# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.
'''

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name, layer.trainable)

'''
# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:165]:
   layer.trainable = False
for layer in model.layers[165:]:
   layer.trainable = True
'''
'''
for layer in base_model.layers:
    layer.trainable = False
'''
# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer='adam', metrics={'multi-labels':'acc','vio-regression':'mae'},\
              loss={'multi-labels':'binary_crossentropy','vio-regression':'mean_squared_error'},\
              loss_weights=[2.,1.])

# data preprocess
pickProtest()
# save weights each epoch
checkpoint = ModelCheckpoint(filepath='weights.ep{epoch:02d}-loss{val_loss:.3f}.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
csv_logger = CSVLogger('training_history_mul.log', append=False)

# train_aug()[0] training set augmentation

# traning test
train_aug_gen, len_train = train_aug_multitask(batch_size=batch_size)
x_test, y_test = get_multitask_val_data()

print(y_test.shape)

#train_augm, len_train = train_aug_multitask(batch_size=batch_size)
H = model.fit_generator(train_aug_gen,
                        steps_per_epoch=len_train//batch_size,
                        validation_data=(x_test, [y_test[:,:10],y_test[:,10]]),
                        epochs=EPOCHS, callbacks = [checkpoint,reduce_lr,early_stopping,csv_logger])
# save the model
model.save('cnn-model-multi-task.h5')

# val
#x_test,y_test = get_multitask_val_data()
scores = model.evaluate(x_test, [y_test[:,:10],y_test[:,10]])

disp ='TEST RESULTS: '
for i,metric in enumerate(model.metrics_names):
   disp +='\n'+metric+': '+ str(scores[i])
print(disp)


# PLOT PART
import matplotlib
import numpy as np

figureName = ["Total Loss", "Multi-Label Loss", "Violation Regression Loss", "Multi-Label Accuracy"]
labelNames = [[("train loss", "loss"), ("validate loss", "val_loss")],
              [("train multi-label loss", "multi-labels_loss"), ("validate multi-label loss", "val_multi-labels_loss")],
              [("train violation regression loss", "vio-regression_loss"), ("validate violation regression loss", "val_vio-regression_loss")],
              [("train multi-label accuracy", "multi-labels_acc"), ("validate multi-label accuracy", "val_multi-labels_acc")]]

matplotlib.use("Agg")

import matplotlib.pyplot as plt
(fig, ax) = plt.subplots(4, 1, figsize=(13, 26))

for (i, l) in enumerate(figureName):
   ax[i].set_title(l)
   ax[i].set_xlabel("Epoch #")
   ax[i].set_ylabel(l)

   for attr in labelNames[i]:
      ax[i].plot(np.arange(0, len(H.epoch)), H.history[attr[1]], label=attr[0])

   ax[i].legend()

plt.tight_layout()
plt.savefig("figure_finetune.png")
plt.close()