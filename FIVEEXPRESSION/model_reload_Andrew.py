from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import os
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.optimizers import RMSprop, SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import regularizers
from keras.regularizers import l1
from keras.models import load_model

img_rows,img_cols=48,48
batch_size=32

train_data_dir='fer2013/train'
validation_data_dir='fer2013/validation'

train_datagen = ImageDataGenerator(
					rescale=1./255,
					rotation_range=30,
					shear_range=0.3,
					zoom_range=0.3,
					width_shift_range=0.4,
					height_shift_range=0.4,
					horizontal_flip=True,
					fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
					train_data_dir,
					color_mode='grayscale',
					target_size=(img_rows,img_cols),
					batch_size=batch_size,
					class_mode='categorical',
					shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
							validation_data_dir,
							color_mode='grayscale',
							target_size=(img_rows,img_cols),
							batch_size=batch_size,
							class_mode='categorical',
							shuffle=True)


# Create the model
model = Sequential()
model = load_model('EmotionDetectionModel.h5')

# model.add(Dense(1024, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(7, activation='softmax'))
print(model.summary())
# model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])


# If you want to train the same model or try other models, go for this
# filepath = os.path.join("EmotionDetectionModel{epoch:02d}-{val_accuracy:.2f}.h5")
filepath = "EmotionDetectionModel_Andrew{epoch:02d}-{val_accuracy:.2f}.h5"
# checkpoint = keras.callbacks.ModelCheckpoint(filepath,
#                                              monitor='val_accuracy',
#                                              verbose=1,
#                                              save_best_only=True,
#                                              mode='max')
checkpoint = keras.callbacks.ModelCheckpoint(filepath,
                                             monitor='val_accuracy',
                                             verbose=1,
                                             save_best_only=True,
                                             mode='max')
callbacks = [checkpoint]
# if mode == "train":

nb_train_samples = 24256
nb_validation_samples = 3006
epochs = 1505
# model_info = model.fit_generator(
#             train_generator,
#             steps_per_epoch=nb_train_samples // batch_size,
#             epochs=epochs,
#             callbacks = callbacks,
#             validation_data=validation_generator,
#             validation_steps=nb_validation_samples // batch_size)
model_info = model.fit(train_generator, steps_per_epoch=nb_train_samples // batch_size,
            epochs=epochs,
            callbacks = callbacks,
            validation_data=validation_generator,
            validation_steps=nb_validation_samples // batch_size)
# plot_model_history(model_info)
# model.save_weights('model.h5')

print(model_info.history.keys())

import matplotlib.pyplot as plt
plt.plot(model_info.history['loss'])
plt.plot(model_info.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.save("EmotionDetectionModel-Andrew_6-20.h5")
plt.plot(model_info.history['accuracy'])
plt.plot(model_info.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
print("Saved model to file")