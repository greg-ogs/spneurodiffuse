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
        self.tflite_model = None
        self.prediction = None
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
            host="172.17.0.2",
            user="user",
            database="dataset",
            password="userpass", port=3306
        )

        self.mycursor = self.mydb.cursor()
        self.mycursor.execute("SELECT * FROM base_dataset")
        self.myresult = self.mycursor.fetchall()
        self.mydb.close()
        self.loaded_dataset = pd.DataFrame(self.myresult, columns=['ID', 'X', 'Y', 'RESULT0', 'T0', 'RESULT1', 'T1',
                                                                   'RESULT2', 'T2', 'RESULT3', 'T3'])
        # print(loaded_dataset.head())

    def prepare_data(self):
        # self.x = self.loaded_dataset.drop(columns=['ID', 'RESULT0', 'RESULT1', 'RESULT2', 'RESULT3'])
        # self.y = self.loaded_dataset[['RESULT0', 'RESULT1', 'RESULT2', 'RESULT3']]
        self.x = np.vstack((np.hstack((self.loaded_dataset['X'].to_numpy(), self.loaded_dataset['X'].to_numpy(),
                                       self.loaded_dataset['X'].to_numpy(), self.loaded_dataset['X'].to_numpy())),
                            np.hstack((self.loaded_dataset['Y'].to_numpy(), self.loaded_dataset['Y'].to_numpy(),
                                       self.loaded_dataset['Y'].to_numpy(), self.loaded_dataset['Y'].to_numpy())),
                            np.hstack((self.loaded_dataset['T0'].to_numpy(), self.loaded_dataset['T1'].to_numpy(),
                                       self.loaded_dataset['T2'].to_numpy(), self.loaded_dataset['T3'].to_numpy()))
                            ))
        # print(self.x.shape)
        self.y = np.hstack((self.loaded_dataset['RESULT0'].to_numpy(), self.loaded_dataset['RESULT1'].to_numpy(),
                            self.loaded_dataset['RESULT2'].to_numpy(), self.loaded_dataset['RESULT3'].to_numpy()))

        self.x = self.x.T

    def model(self):
        self.num_classes = len(np.unique(self.y))

        self.model = Sequential()
        self.model.add(Dense(units=32, activation='relu', input_dim=self.x.shape[1]))
        self.model.add(Dense(units=64, activation='relu'))
        self.model.add(Dense(units=128, activation='relu'))
        self.model.add(Dense(units=256, activation='relu'))
        self.model.add(Dense(units=1, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                           metrics=['accuracy'])
        self.model.summary()

    def train(self):
        # Create a callback that saves the model's weights
        # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='cp.ckpt',
        #                                                  save_weights_only=True,
        #                                                  verbose=1)
        history = self.model.fit(self.x, self.y, epochs=3, batch_size=32, verbose=1, validation_split=0.2)
        self.model.save('model.keras')
        accuracy = self.model.evaluate(self.x, self.y)
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs_range = range(3)
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

    def predict(self, elapsed_time):
        reconstructed_model = keras.models.load_model("model.keras")
        for i in np.arange(0, 25, 0.03):
            for j in np.arange(0, 25, 0.03):
                # print(str(round(i, 2)) + ',' + str(round(j, 2)))
                data = pd.DataFrame([[i, j, elapsed_time]])
                self.prediction = reconstructed_model.predict(data)
                if self.prediction[0] > 0.5:
                    print(self.prediction)
                    break


if __name__ == "__main__":
    # tf.config.set_visible_devices([], 'GPU')
    available_gpu()
    stage1 = Stage1ANN()
    stage1.load_data()
    stage1.prepare_data()
    stage1.model()
    stage1.train()
    elapsed_time = 864000
    stage1.predict(elapsed_time)
