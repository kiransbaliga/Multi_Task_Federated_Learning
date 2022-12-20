from kivy.app import App
from kivy.uix.button import Button
from keras.datasets import cifar10
import tensorflow as tf
import numpy as np
from keras import models
from keras import layers
from keras.models import Sequential


class MainApp(App):
    def build(self):
        button = Button(text='Hello from Kivy',
                        size_hint=(.5, .5),
                        pos_hint={'center_x': .5, 'center_y': .5})
        button.bind(on_press=self.on_press_button)

        return button

    def on_press_button(self, instance):
        
        print('You pressed the button!')
        (X_train,y_train),(X_test,y_test)=cifar10.load_data()
        y_train=y_train.reshape(-1,)
        classes=["somethingelse","somethingelse","somethingelse","somethingelse","somethingelse","somethingelse","somethingelse","somethingelse","ship","truck"]
        X_train=X_train/255
        X_test=X_test/255

        cnn =models.Sequential([
                        layers.Conv2D( filters=32 , kernel_size=(3,3) , activation='relu' , input_shape=(32,32,3) ),
                        layers.MaxPooling2D((2,2)),
                        layers.Conv2D( filters=64 , kernel_size=(3,3) , activation='relu'  ),
                        layers.MaxPooling2D((2,2)),
                        layers.Flatten(),
                        layers.Dense(64,activation='relu'),
                        layers.Dense(10,activation='softmax'),
        ])
        cnn.compile( optimizer='adam' , loss='sparse_categorical_crossentropy' , metrics=['accuracy']       
            )
        cnn.fit(X_train,y_train, epochs=1)
        cnn.evaluate(X_test,y_test)
        # save the cnn model weights as numpy array
        



        cnn.save("./models/models.h5")
        cnn.save_weights("./models/weights/weights")
        print("successfull saved the model and saved the weights....")


if __name__ == '__main__':
    app = MainApp()
    app.run()