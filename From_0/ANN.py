import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.optimizers.schedules import ExponentialDecay
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import mysql.connector as mysql
from tqdm import tqdm



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
        self.model = None
        self.tflite_model = None
        self.prediction = [0,]
        self.num_classes = None  # Classes of the network
        self.x = None  # Input variables for ANN
        self.y = None  # Output variables for ANN

        self.myresult = None  # DATASET
        self.mycursor = None
        self.mydb = None
        self.loaded_dataset = None

    def load_data(self):
        # Load from MySQL
        self.mydb = mysql.connect(
            host="172.17.0.2",  # Use docker inspect <docker id > for ip
            user="user",
            database="dataset",
            password="userpass", port=3306
        )

        self.mycursor = self.mydb.cursor()
        self.mycursor.execute("SELECT * FROM dynamic")
        self.myresult = self.mycursor.fetchall()
        self.mydb.close()
        self.loaded_dataset = pd.DataFrame(self.myresult, columns=['ID', 'X', 'Y', 'RESULT'])
        # print(loaded_dataset.head())

    def prepare_data(self):
        self.x = self.loaded_dataset[['X', 'Y']].values
        self.y = self.loaded_dataset[['RESULT']].values

        px = pd.DataFrame(self.x)
        py = pd.DataFrame(self.y)
        print(px.info)
        print(py.info)

    def set_model(self):
        lr_schedule = ExponentialDecay(
            initial_learning_rate=1e-3,
            decay_steps=10000,  # Number of steps before decay
            decay_rate=0.9  # Factor by which the learning rate is reduced
        )
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

        self.num_classes = len(np.unique(self.y))
        print(self.x.shape[1], '------------------------------')

        self.model = Sequential()
        self.model.add(Dense(units=128, activation='relu', input_dim=self.x.shape[1]))
        self.model.add(Dense(units=128, activation='relu'))
        self.model.add(Dense(units=128, activation='relu'))
        self.model.add(Dense(units=128, activation='relu'))
        self.model.add(Dense(units=1, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.000001),
                           metrics=['accuracy'])
        self.model.summary()

    def train(self):
        # Create a callback that saves the model's weights
        cp_callback = [keras.callbacks.ModelCheckpoint(filepath='model.{epoch:03d}-{val_accuracy:.2f}.keras',
                                                       monitor='val_accuracy', verbose=1, save_freq='epoch')
                       # Path of the callback file, monitor is for monitoring a variable and use in conjunction with
                       # mode to choose the best epoch qhe use save_best_only
                       # keras.callbacks.EarlyStopping(patience=20, monitor='val_accuracy', verbose=1,
                       #                               min_delta=0.00001, mode='max')
                       # Epoch to wait without improvement is in patience argument, min_delta is minimum improvement
                       # and mode is to stop when the quantity monitored has stopped increasing (max)
                       ]

        history = self.model.fit(self.x, self.y, epochs=30, initial_epoch=0, batch_size=1, verbose=1,
                                 validation_split=0.2, callbacks=cp_callback)
        self.model.save('model.keras')
        accuracy = self.model.evaluate(self.x, self.y)
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs_range = range(30)
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

    def convert(self):
        converter = tf.lite.TFLiteConverter.from_saved_model('model.keras')  # path to the SavedModel directory
        self.tflite_model = converter.convert()

        # Save the model.
        with open('model.tflite', 'wb') as f:
            f.write(self.tflite_model)

    def predict(self):
        winning_result = np.array([[0, 0]])
        reconstructed_model = keras.models.load_model("model.keras")
        for i in tqdm(np.arange(0, 25, 0.1), desc='Predicting X'):
            if self.prediction[0] > 0.9:
                break
            for j in tqdm(np.arange(0, 25, 0.1), desc='Predicting Y'):
                # print(str(round(i, 2)) + ',' + str(round(j, 2)))
                data = pd.DataFrame([[i, j]])
                self.prediction = reconstructed_model.predict(data, verbose=0)
                # print(self.prediction[0])
                if self.prediction[0] > 0.9:
                    print(self.prediction)
                    print(str(i) + ' ' + str(j) + ' are x - y possible coords')
                    break


if __name__ == "__main__":
    # tf.config.set_visible_devices([], 'GPU')
    # available_gpu()
    # stage1 = Stage1ANN()
    # stage1.load_data()
    # stage1.prepare_data()
    # stage1.set_model()
    # stage1.train()
    # stage1.predict()

    # Testing models
    reconstructed_model = keras.models.load_model("model.keras")
    data = pd.DataFrame([[11.17, 24.3]])
    # print(type(data))
    prediction = reconstructed_model.predict(data, verbose=0)
    print(prediction)