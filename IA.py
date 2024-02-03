import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import time


class BackPropagation:
    def __init__(self):
        self.val_ds = None
        self.train_ds = None
        self.class_names = ['c', 'dwl', 'dwr', 'upl', 'upr']
        self.batch_size = 32
        self.img_height = 180
        self.img_width = 180
        self.data_dir = "E:\spneurodiffuse\dataset"

        # image_count = len(list(data_dir.glob('*/*.jpg')))
        # print(image_count)

    def train_model(self):
        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)

        self.val_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)

        self.class_names = self.train_ds.class_names
        AUTOTUNE = tf.data.AUTOTUNE

        train_ds = self.train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = self.val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        normalization_layer = layers.Rescaling(1. / 255)

        normalized_ds = self.train_ds.map(lambda x, y: (normalization_layer(x), y))
        image_batch, labels_batch = next(iter(normalized_ds))
        first_image = image_batch[0]
        # Notice the pixel values are now in `[0,1]`.
        print(np.min(first_image), np.max(first_image))

        num_classes = len(self.class_names)

        model = Sequential([
            layers.Rescaling(1. / 255, input_shape=(self.img_height, self.img_width, 3)),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes)
        ])

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        model.summary()

        epochs = 5
        history = model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs
        )

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(epochs)

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

        model.save('my_model.keras')

        input("Enter to continue")

    def predict(self, data):
        start_tieme = time.time()
        img = data
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        model = tf.keras.models.load_model('my_model.keras')
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        # Change class_names for a static array
        # Identify class index in this array

        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(self.class_names[np.argmax(score)], 100 * np.max(score))
        )
        end_time = time.time()
        elapsed_time = end_time - start_tieme
        time.sleep(4 - elapsed_time)
        return self.class_names[np.argmax(score)]


if __name__ == "__main__":
    mod = BackPropagation()
    image_data = tf.keras.utils.load_img(
        "E:\spneurodiffuse\Test\IMG_0.jpg", target_size=(mod.img_height, mod.img_width)
    )
    for i in range(3):
        mod.predict(image_data)
        print(str(i) + "-")
