import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import mysql.connector as mysql


def available_gpu():  # Available nvidia gpu function
    gpus = tf.config.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(gpus))
    print('----------------------------------------------------------------')
    if gpus:
        # Optionally, print details about each GPU
        for gpu in gpus:
            print('----------------------------------------------------------------')
            print(gpu)

    print('----------------------------------------------------------------')
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print('----------------------------------------------------------------')


class Stage1ANN:  # Classification stage
    def __init__(self):
        self.x = None  # Input variables for ANN
        self.y = None  # Output variables for ANN

        self.myresult = None  # DATASET
        self.mycursor = None
        self.mydb = None

    def load_data(self):
        # Load from MySQL
        self.mydb = mysql.connect(
            host="172.17.0.2",
            user="user",
            database="dataset",
            password="userpass", port=3306
        )

        self.mycursor = self.mydb.cursor()
        self.mycursor.execute("SELECT * FROM base_dataset")
        self.myresult = self.mycursor.fetchall()
        self.mydb.close()
        loaded_dataset = pd.DataFrame(self.myresult, columns=['ID', 'X', 'Y', 'RESULT', 'T0'])
        # print(loaded_dataset.head())
        return loaded_dataset

    def prepare_data(self, loaded_dataset):
        self.x = loaded_dataset.drop(columns=['ID', 'RESULT'])
        self.y = loaded_dataset['RESULT']
        # print('x head')
        # print(self.x.head())
        # print('y head')
        # print(self.y.head())
        # return self.x, self.y

    def model(self):
        self.num_classes = len(np.unique(self.y))

        self.model = Sequential()
        self.model.add(Dense(units=32, activation='relu', input_dim=self.x.shape[1]))
        self.model.add(Dense(units=64, activation='relu'))
        self.model.add(Dense(units=128, activation='relu'))
        self.model.add(Dense(units=256, activation='relu'))
        self.model.add(Dense(units=1, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self):
        history = self.model.fit(self.x, self.y, epochs=3, batch_size=32, verbose=1, validation_split=0.2)

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs_range = range(1000)
        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()


if __name__ == "__main__":
    available_gpu()
    stage1 = Stage1ANN()
    dataset = stage1.load_data()
    stage1.prepare_data(dataset)
    stage1.model()
    stage1.train()
