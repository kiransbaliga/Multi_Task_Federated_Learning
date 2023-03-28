from kivy.app import App
from kivy.uix.button import Button
from keras.datasets import cifar10
import tensorflow as tf
import numpy as np
from keras import models
from keras import layers
from keras.models import Sequential,load_model
from kivy.uix.gridlayout import GridLayout
import pickle

class main_layout(GridLayout):
    def __init__(self, **kwargs):
        super(main_layout, self).__init__(**kwargs)
        self.cols = 2
        self.rows = 2
        self.add_widget(Button(text='Click to Train',
                        size_hint=(.25, .5),
                        pos=(100,500),
                        on_press=self.train_model_cifar10,
                        ))
        self.add_widget(Button(text='Download Global Model',
                        size_hint=(.25,.5),
                        pos=(300,500),
                        on_press=self.download_global_model,
        ))
        self.add_widget(Button(text='Upload Local Model',
                        size_hint=(.25,.5),
                        pos=(500,500),
                        on_press=self.on_press_button3,
        ))

        # self.add_widget(Image(source='./download.png',size_hint=(.25,.25),pos=(100,400)))
        
    def download_global_model(self,instance):
        print("You pressed the button2")
        import os
        import requests
        import shutil
        #local host url
        url= 'http://localhost:500/getglobalmodel'
        try:

            r = requests.get(url, stream=True)
            if r.status_code == 200:
                with open('global_model.h5', 'wb') as f:
                    r.raw.decode_content = True
                    shutil.copyfileobj(r.raw, f)
        except:
            pass
        print("Downloaded the global model....")
        # load models into a list of models from a directory
    def on_press_button3():
        print("You pressed the button3")
    
        # load model
        model = load_model('./models/models.h5')
        
    def multi_task_cifar_10():
        # Load the CIFAR-10 dataset
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train = x_train / 255.0
        # Preprocess the data
        x_test = x_test / 255.0
        y_train_categorical = tf.keras.utils.to_categorical(y_train, num_classes=10)
        y_test_categorical = tf.keras.utils.to_categorical(y_test, num_classes=10)
        # Define the binary ship vs not-ship label
        ship_label = 8
        y_train_ship = np.where(y_train == ship_label, 1, 0)
        y_test_ship = np.where(y_test == ship_label, 1, 0)
        # Define the model architecture
        inputs = tf.keras.Input(shape=(32, 32, 3))
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        # Output for the 10 original CIFAR-10 classes
        task1_output = tf.keras.layers.Dense(10, activation='softmax', name='task1_output')(x)
        # Output for the binary ship vs not-ship class
        task2_output = tf.keras.layers.Dense(1, activation='sigmoid', name='task2_output')(x)
        # Define the model and compile it
        model = tf.keras.Model(inputs=inputs, outputs=[task1_output, task2_output])
        model.compile(optimizer='adam',
                      loss={'task1_output': 'categorical_crossentropy', 'task2_output': 'binary_crossentropy'},
                      loss_weights={'task1_output': 1.0, 'task2_output': 1.0},
                      metrics=['accuracy'])
        # Train the model
        model.fit(x_train, {'task1_output': y_train_categorical, 'task2_output': y_train_ship},
                  validation_data=(x_test, {'task1_output': y_test_categorical, 'task2_output': y_test_ship}),
                  epochs=10, batch_size=32)
        # Save the model
        model.save('multitask_cifar10.h5')

    def train_model_cifar10(self, instance):
        def multi_task_cifar_10():
            # Load the CIFAR-10 dataset
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
            x_train = x_train / 255.0
            # Preprocess the data
            x_test = x_test / 255.0
            y_train_categorical = tf.keras.utils.to_categorical(y_train, num_classes=10)
            y_test_categorical = tf.keras.utils.to_categorical(y_test, num_classes=10)
            # Define the binary ship vs not-ship label
            ship_label = 8
            y_train_ship = np.where(y_train == ship_label, 1, 0)
            y_test_ship = np.where(y_test == ship_label, 1, 0)
            # Define the model architecture
            inputs = tf.keras.Input(shape=(32, 32, 3))
            x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
            x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
            x = tf.keras.layers.Dropout(0.25)(x)
            x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
            x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
            x = tf.keras.layers.Dropout(0.25)(x)
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(512, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            # Output for the 10 original CIFAR-10 classes
            task1_output = tf.keras.layers.Dense(10, activation='softmax', name='task1_output')(x)
            # Output for the binary ship vs not-ship class
            task2_output = tf.keras.layers.Dense(1, activation='sigmoid', name='task2_output')(x)
            # Define the model and compile it
            model = tf.keras.Model(inputs=inputs, outputs=[task1_output, task2_output])
            model.compile(optimizer='adam',
                          loss={'task1_output': 'categorical_crossentropy', 'task2_output': 'binary_crossentropy'},
                          loss_weights={'task1_output': 1.0, 'task2_output': 1.0},
                          metrics=['accuracy'])
            # Train the model
            model.fit(x_train, {'task1_output': y_train_categorical, 'task2_output': y_train_ship},
                      validation_data=(x_test, {'task1_output': y_test_categorical, 'task2_output': y_test_ship}),
                      epochs=10, batch_size=32)
            # Save the model
            model.save('multitask_cifar10.h5')
            
        multi_task_cifar_10()
        
        
        print("successfull saved the model and saved the weights....")

class MainApp(App):
    def build(self):
        # button = Button(text='Click to Train',
        #                 size_hint=(.2, .2),
        #                 pos=(100,250))
        # button2 = Button(text='Download Global Model',
        #                 size_hint=(.5,.5),
        #                 pos=(300,250)
        # )
        # button.bind(on_press=self.on_press_button)
        # button2.bind(on_press=self.on_press_button2)
        # return button
        return main_layout()

if __name__ == '__main__':
    app = MainApp()
    app.run()