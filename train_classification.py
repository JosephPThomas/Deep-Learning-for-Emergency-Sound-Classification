import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
from IPython.display import Audio
import warnings
warnings.filterwarnings('ignore')

paths = []
labels = []
for dirname, _, filenames in os.walk('dataset/classification_training'):
    for filename in filenames:
        paths.append(os.path.join(dirname, filename))
        label = filename.split('_')[-3]
        label = label.split('.')[0]
        labels.append(label.lower())
    if len(paths) == 3000:
        break

## Create a dataframe
df = pd.DataFrame()
df['alarm'] = paths
df['label'] = labels
df.head(10)

df['label'].value_counts()
sns.countplot(df['label'])

alarm = 'danger'
path = np.array(df['alarm'][df['label'] == alarm])[3]
data, sampling_rate = librosa.load(path)
Audio(path)

alarm = 'fire'
path = np.array(df['alarm'][df['label']==alarm])[0]
data, sampling_rate = librosa.load(path)
Audio(path)

alarm = 'gas'
path = np.array(df['alarm'][df['label']==alarm])[0]
data, sampling_rate = librosa.load(path)
Audio(path)

alarm = 'non'
path = np.array(df['alarm'][df['label']==alarm])[0]
data, sampling_rate = librosa.load(path)
Audio(path)

alarm = 'tsunami'
path = np.array(df['alarm'][df['label']==alarm])[0]
data, sampling_rate = librosa.load(path)
Audio(path)


def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=1, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    return mfcc

extract_mfcc(df['alarm'][0])

X_mfcc = df['alarm'].apply(lambda x: extract_mfcc(x))

X = [x for x in X_mfcc]
X = np.array(X)
X.shape

## input split
X = np.expand_dims(X, -1)
X.shape

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
y = enc.fit_transform(df[['label']])
y = y.toarray()
y.shape


from sklearn.model_selection import train_test_split
import tensorflow as tf

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2)


from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

model = Sequential([
    LSTM(256, return_sequences=False, input_shape=(13,1)),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(5, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=50)

epochs = list(range(50))
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

model.evaluate(X, y)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

val_loss, val_acc = model.evaluate(X_validation, y_validation, verbose=0)

# train Accuracy and Loss
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)

# Val Accuracy and Loss
val_loss, val_acc = model.evaluate(X_validation, y_validation, verbose=0)

model.save('model_alarm_classification_RNN.h5')






