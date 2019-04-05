import pandas as pd
import tensorflow as tf
from glob import glob
from random import randint
from keras import backend as K
DIR_WHERE_IMAGES_ARE="data/trimmed/*/*"
#In this case the classes are the 3rd level
K.clear_session()

def train_split_dataframes(dpath):

    records_all=[[g] for g in glob(DIR_WHERE_IMAGES_ARE)]
    records = [ r for r in records_all if  r[0].find("Gull")!=-1]
    df=pd.DataFrame.from_records(records,columns=['file'])

    df['Class']=df.file.apply(lambda x: x.split('/')[2])
    df['Test']=False
    df['Train']=False

    df_cnt=df.groupby('Class').size().to_frame()
    df_cnt.columns=['Cnt']

    df=df.merge(df_cnt,on='Class')

    def set_train(max_occurences_in_class):
        #We want 66% to be training data
        r=randint(1,int(max_occurences_in_class))
        if r/max_occurences_in_class <= .61:
            return True
        else:
            return False
    df.Train = df.apply(lambda row: set_train(row['Cnt']),axis=1)
    df.Test = ~df.Train
    return df

df=train_split_dataframes(DIR_WHERE_IMAGES_ARE)
CLASSES=len(set(list(df.Class)))
print("Check Percentage {}".format(len(df[df.Test==True])/len(df[df.Test==False])))

df_test=df[df.Test==True]
df_train=df[df.Train==True]

# there should be non in both data sets
df_test.merge(df_train,on='file')

from keras.models import Sequential
#Import from keras_preprocessing not from keras.preprocessing
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers

SIZE_X = SIZE_Y = 70

#datagen=ImageDataGenerator(rescale=1./255.)
datagen = ImageDataGenerator(
    rescale=1./255.,
    rotation_range=15,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               shear_range=0.01,
                               zoom_range=[0.9, 1.25],
                               horizontal_flip=True,
                               vertical_flip=False,
                               fill_mode='reflect',
                               data_format='channels_last',
                               brightness_range=[0.5, 1.5])


# Train and Valid both come from TRAIN

train_generator=datagen.flow_from_dataframe(
dataframe=df_train,
x_col="file",
y_col="Class",
batch_size=32,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(SIZE_X,SIZE_Y))

valid_generator=datagen.flow_from_dataframe(
dataframe=df_train,
x_col="file",
y_col="Class",
batch_size=32,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(SIZE_X,SIZE_Y))

# Test comes from Test
#test_datagen=ImageDataGenerator(rescale=1./255.)
test_datagen = ImageDataGenerator(
    rescale=1./255.,
    rotation_range=15,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               shear_range=0.01,
                               zoom_range=[0.9, 1.25],
                               horizontal_flip=True,
                               vertical_flip=False,
                               fill_mode='reflect',
                               data_format='channels_last',
                               brightness_range=[0.5, 1.5])

test_generator=test_datagen.flow_from_dataframe(
dataframe=df_test,
x_col="file",
y_col="Class",
batch_size=32,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(SIZE_X,SIZE_Y))

model = Sequential()
model.add(Conv2D(48, (3, 3), padding='same',
                 input_shape=(SIZE_X,SIZE_Y,3)))
model.add(Activation('relu'))
model.add(Conv2D(SIZE_X, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(96, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(96, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(CLASSES, activation='softmax'))
model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])


callbacks_wanted = [
  # Interrupt training if `val_loss` stops improving for over 2 epochs
  tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
  # Write TensorBoard logs to `./logs` directory
  tf.keras.callbacks.TensorBoard(log_dir='./logs',
                                 histogram_freq=0,
                                 write_graph=True,
                                 write_images=True)
]

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    callbacks=callbacks_wanted,
                    epochs=20
)

model.save("birds.h5")
