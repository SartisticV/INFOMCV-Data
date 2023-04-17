from collections import Counter
from sklearn.model_selection import train_test_split
import cv2 as cv
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import gc
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn

### SKELETON CODE ###
keep_stanford40 = ["applauding", "climbing", "drinking", "jumping", "pouring_liquid", "riding_a_bike", "riding_a_horse",
        "running", "shooting_an_arrow", "smoking", "throwing_frisby", "waving_hands"]
with open('Stanford40_ImageSplits/ImageSplits/train.txt', 'r') as f:
    # We won't use these splits but split them ourselves
    train_files = [file_name for file_name in list(map(str.strip, f.readlines())) if '_'.join(file_name.split('_')[:-1]) in keep_stanford40]
    train_labels = ['_'.join(name.split('_')[:-1]) for name in train_files]

with open('Stanford40_ImageSplits/ImageSplits/test.txt', 'r') as f:
    # We won't use these splits but split them ourselves
    test_files = [file_name for file_name in list(map(str.strip, f.readlines())) if '_'.join(file_name.split('_')[:-1]) in keep_stanford40]
    test_labels = ['_'.join(name.split('_')[:-1]) for name in test_files]

# Combine the splits and split for keeping more images in the training set than the test set.
all_files = train_files + test_files
all_labels = train_labels + test_labels
train_files, test_files = train_test_split(all_files, test_size=0.1, random_state=0, stratify=all_labels)
train_labels = ['_'.join(name.split('_')[:-1]) for name in train_files]
test_labels = ['_'.join(name.split('_')[:-1]) for name in test_files]
train_files, train_labels = shuffle(train_files, train_labels, random_state=0)
print(f'Train files ({len(train_files)}):\n\t{train_files}')
print(f'Train labels ({len(train_labels)}):\n\t{train_labels}\n'\
      f'Train Distribution:{list(Counter(sorted(train_labels)).items())}\n')
print(f'Test files ({len(test_files)}):\n\t{test_files}')
print(f'Test labels ({len(test_labels)}):\n\t{test_labels}\n'\
      f'Test Distribution:{list(Counter(sorted(test_labels)).items())}\n')
action_categories = sorted(list(set(train_labels)))
print(f'Action categories ({len(action_categories)}):\n{action_categories}')

# Shuffle train data so that all classes are represented in training and validation data
train_files, train_labels = shuffle(train_files, train_labels, random_state=0)

image_no = 234  # change this to a number between [0, 1200] and you can see a different training image
img = cv.imread(f'Stanford40_JPEGImages/JPEGImages/{train_files[image_no]}')
print(f'An image with the label - {train_labels[image_no]}')
print(img.shape)

#Load training and test data
train = []
for i in train_files:
    img = cv.imread(f'Stanford40_JPEGImages/JPEGImages/{i}')
    img = cv.resize(img, (224, 224), interpolation=cv.INTER_AREA)
    train.append(img)
train = np.array(train)

test = []
for i in test_files:
    img = cv.imread(f'Stanford40_JPEGImages/JPEGImages/{i}')
    img = cv.resize(img, (224, 224), interpolation=cv.INTER_AREA)
    test.append(img)
test = np.array(test)

#Prepocess data for the network
train = np.divide(train, 255)
test = np.divide(test, 255)
le = LabelEncoder()
train_labels = le.fit_transform(train_labels)
test_labels = le.fit_transform(test_labels)

model = keras.Sequential([
    keras.layers.Input(shape=(224, 224, 3)),
    keras.layers.Conv2D(8, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2), (2, 2)),
    keras.layers.Conv2D(16, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2), (2, 2)),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2), (2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2), (2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2), (2, 2)),
    keras.layers.Dropout(0.2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(12, activation='softmax')
])

model.summary()
model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics='accuracy')
history = model.fit(train, train_labels, batch_size=32,
                        epochs=15, verbose=1, validation_split=0.2)

#Plot accuracy and loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

loss, accuracy = model.evaluate(test, test_labels, verbose=0)
print(loss, accuracy)

#Uncomment to save model
#model.save('Stanford.h5')


### SKELETON CODE ###
keep_hmdb51 = ["clap", "climb", "drink", "jump", "pour", "ride_bike", "ride_horse",
        "run", "shoot_bow", "smoke", "throw", "wave"]


TRAIN_TAG, TEST_TAG = 1, 2
train_files, test_files = [], []
train_labels, test_labels = [], []
annotation_paths = glob.glob(f'test_train_splits/*test_split1.txt')
for filepath in annotation_paths:
    class_name = '_'.join(filepath.split('\\')[-1].split('_')[:-2])
    if class_name not in keep_hmdb51:
        continue  # skipping the classes that we won't use.
    with open(filepath) as fid:
        lines = fid.readlines()
    for line in lines:
        video_filename, tag_string = line.split()
        tag = int(tag_string)
        if tag == TRAIN_TAG:
            train_files.append(video_filename)
            train_labels.append(class_name)
        elif tag == TEST_TAG:
            test_files.append(video_filename)
            test_labels.append(class_name)


train_files, train_labels = shuffle(train_files, train_labels, random_state=0)
print(f'Train files ({len(train_files)}):\n\t{train_files}')
print(f'Train labels ({len(train_labels)}):\n\t{train_labels}\n'\
      f'Train Distribution:{list(Counter(sorted(train_labels)).items())}\n')
print(f'Test files ({len(test_files)}):\n\t{test_files}')
print(f'Test labels ({len(test_labels)}):\n\t{test_labels}\n'\
      f'Test Distribution:{list(Counter(sorted(test_labels)).items())}\n')
action_categories = sorted(list(set(train_labels)))
print(f'Action categories ({len(action_categories)}):\n{action_categories}')


#Function to extract middle frames of HMDB51 videos, saved in a folder 'midframe'
def middle_frame(train_files, test_files):
    for file in train_files:
        path = glob.glob(f'video_data/*/{file}')
        vidcap = cv.VideoCapture(path[0])
        total_frames = int(vidcap.get(cv.CAP_PROP_FRAME_COUNT))
        middle_frame = total_frames // 2
        vidcap.set(cv.CAP_PROP_POS_FRAMES, middle_frame)
        _, frame = vidcap.read()
        cv.imwrite(f'midframe/{file}.jpg', frame)

    for file in test_files:
        path = glob.glob(f'video_data/*/{file}')
        vidcap = cv.VideoCapture(path[0])
        total_frames = int(vidcap.get(cv.CAP_PROP_FRAME_COUNT))
        middle_frame = total_frames // 2
        vidcap.set(cv.CAP_PROP_POS_FRAMES, middle_frame)
        _, frame = vidcap.read()
        cv.imwrite(f'midframe/{file}.jpg', frame)


#Uncomment to store midframe images
#middle_frame(train_files, test_files)

gc.collect()
tf.keras.backend.clear_session()
del model

#Load model for transfer training
model = keras.models.load_model('Stanford.h5')

#Freeze all but the output layer
for layer in model.layers[:-2]:
    layer.trainable = False

model.summary()

#Load training and test data
train = []
for i in train_files:
    img = cv.imread(f'midframe/{i}.jpg')
    img = cv.resize(img, (224, 224), interpolation=cv.INTER_AREA)
    train.append(img)
train = np.array(train)

test = []
for i in test_files:
    img = cv.imread(f'midframe/{i}.jpg')
    img = cv.resize(img, (224, 224), interpolation=cv.INTER_AREA)
    test.append(img)
test = np.array(test)

#Preprocess data for network
train = np.divide(train, 255)
test = np.divide(test, 255)
le = LabelEncoder()
train_labels = le.fit_transform(train_labels)
test_labels = le.fit_transform(test_labels)

model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9), metrics='accuracy')
history = model.fit(train, train_labels, batch_size=256,
                        epochs=15, verbose=1, validation_split=0.2)

loss, accuracy = model.evaluate(test, test_labels, verbose=0)
print(loss, accuracy)

#Plot data
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#Unfreeze layers
for layer in model.layers[:-2]:
    layer.trainable = True

#Fine tune model
model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(), metrics='accuracy')
history = model.fit(train, train_labels, batch_size=256,
                        epochs=8, verbose=1, validation_split=0.2)

loss, accuracy = model.evaluate(test, test_labels, verbose=0)
print(loss, accuracy)

#Plot data
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#Uncomment to save Transfer Learning model
#model.save('TransferLearning.h5')


#Function to calculate batches of 16 optical flow maps of the middle frames
def opticalflow(files, batchsize):
    for file in files:
        path = glob.glob(f'video_data/*/{file}')
        vidcap = cv.VideoCapture(path[0])
        total_frames = int(vidcap.get(cv.CAP_PROP_FRAME_COUNT))
        start_frame = total_frames // 2 - (batchsize // 2)

        vidcap.set(cv.CAP_PROP_POS_FRAMES, start_frame)

        _, prev_frame = vidcap.read()
        prev_frame = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
        prev_frame = cv.resize(prev_frame, (224, 224), interpolation=cv.INTER_AREA)

        totalflow = []
        for i in range(batchsize):
            _, frame = vidcap.read()
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            frame = cv.resize(frame, (224, 224), interpolation=cv.INTER_AREA)

            flow = cv.calcOpticalFlowFarneback(prev_frame, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            totalflow.append(flow)
            prev_frame = frame

        totalflow = np.array(totalflow)
        np.save(f'flow/{file}', totalflow)

#Uncomment to create optical flow batches of training and test data
#opticalflow(train_files, 16)
#opticalflow(test_files, 16)

#Custom Data Generator class for optical flow batches
#Resource used: https://medium.com/analytics-vidhya/write-your-own-custom-data-generator-for-tensorflow-keras-1252b64e41c3
class OptFlowDataGen(tf.keras.utils.Sequence):

    def __init__(self, files, labels, batch_size):

        self.files = files
        self.labels = labels
        self.batch_size = batch_size

    def __getitem__(self, index):
        batches = self.files[index * self.batch_size:(index + 1) * self.batch_size]
        flow = np.array([np.load(f'flow/{file}.npy') for file in batches])

        labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]

        return flow, labels

    def __len__(self):
        return int(np.ceil(len(self.files) / float(self.batch_size)))


gc.collect()
tf.keras.backend.clear_session()
del model

#Create own validation data so it can be turned into the correct Data Generator Structure
train_files, val_files, train_labels, val_labels = train_test_split(train_files, train_labels, test_size=0.1, random_state=0, stratify=train_labels)

traindata = OptFlowDataGen(train_files, train_labels, 16)
valdata = OptFlowDataGen(val_files, val_labels, 16)
testdata = OptFlowDataGen(test_files, test_labels, 16)


model = keras.Sequential([
    keras.layers.Input(shape=(16, 224, 224, 2), name='input_2'),
    keras.layers.Conv3D(8, (3, 3, 3), activation='relu'),
    keras.layers.MaxPooling3D((2, 2, 2)),
    keras.layers.Conv3D(16, (3, 3, 3), activation='relu'),
    keras.layers.MaxPooling3D((1, 2, 2)),
    keras.layers.Conv3D(32, (3, 3, 3), activation='relu'),
    keras.layers.MaxPooling3D((1, 2, 2)),
    keras.layers.Conv3D(64, (2, 3, 3), activation='relu'),
    keras.layers.MaxPooling3D((1, 2, 2)),
    keras.layers.Conv3D(64, (2, 3, 3), activation='relu'),
    keras.layers.MaxPooling3D((1, 2, 2)),
    keras.layers.Dropout(0.2, name='dropout_2'),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(12, activation='softmax')
])

model.summary()

model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics='accuracy')


history = model.fit(traindata, batch_size=16,
                        epochs=10, verbose=1, validation_data=valdata)

#Plot data
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

loss, accuracy = model.evaluate(testdata, verbose=0)
print(loss, accuracy)

#Uncomment to save Optical Flow model
#model.save('OptFlow.h5')

gc.collect()
tf.keras.backend.clear_session()
del model

#Custom Data Generator class for combined optical flow batches and midframe images
class TwoStreamDataGen(tf.keras.utils.Sequence):

    def __init__(self, data, files, labels, batch_size):

        self.data = data
        self.files = files
        self.labels = labels
        self.batch_size = batch_size

    def __getitem__(self, index):

        frame = self.files[index * self.batch_size:(index + 1) * self.batch_size]
        img = [cv.imread(f'midframe/{f}.jpg') for f in frame]
        img = np.array([np.divide(cv.resize(i, (224, 224), interpolation=cv.INTER_AREA), 255) for i in img])

        batches = self.data[index][0]

        labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]

        return {'input_1':img, 'input_2':batches}, labels

    def __len__(self):
        return int(np.ceil(len(self.files) / float(self.batch_size)))

traindata2 = TwoStreamDataGen(traindata, train_files, train_labels, 16)
valdata2 = TwoStreamDataGen(valdata, val_files, val_labels, 16)
testdata2 = TwoStreamDataGen(testdata, test_files, test_labels, 16)

#Load models
model1 = keras.models.load_model('Final/TransferLearning.h5')
model2 = keras.models.load_model('Final/OptFlow.h5')

#Add new layers (see report for explanation)
new_output1 = keras.layers.Conv2D(64,(1,1), activation='relu', name='identity')(model1.layers[-5].output)
new_output2 = keras.layers.Reshape((5,5,64))(model2.layers[-5].output)

#Fuse layers using the Average layer
fusion = keras.layers.Average()([new_output1, new_output2])
x = keras.layers.Flatten()(fusion)
x = keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
x = keras.layers.Dropout(0.5, name='dropout_3')(x)
output = keras.layers.Dense(12, activation='softmax')(x)

model = keras.Model(inputs=[model1.input, model2.input], outputs=output)

model.summary()

model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics='accuracy')

history = model.fit(traindata2, batch_size=16,
                        epochs=10, verbose=1, validation_data=valdata2)

#Plot data
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

loss, accuracy = model.evaluate(testdata2, verbose=0)
print(loss, accuracy)

#Uncomment to save TwoStream model
#model.save('TwoStream.h5')


