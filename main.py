from kivy.app import App
from kivy.uix.button import Button
from keras.datasets import cifar10
import tensorflow as tf
import numpy as np
from keras import models
from keras import layers
from keras.models import Sequential
from kivy.uix.gridlayout import GridLayout

class main_layout(GridLayout):
    def __init__(self, **kwargs):
        super(main_layout, self).__init__(**kwargs)
        self.cols = 1
        self.add_widget(Button(text='Click to Train',
                        size_hint=(.5, .5),
                        pos=(100,250),
                        on_press=self.on_press_button,
                        ))
        self.add_widget(Button(text='Download Global Model',
                        size_hint=(.5,.5),
                        pos=(300,250),
                        on_press=self.on_press_button2,
        ))
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
        cnn.fit(X_train,y_train, epochs=10)
        cnn.evaluate(X_test,y_test)
        # save the cnn model weights as numpy array
        cnn.save("./models/models.h5")
        # cnn.save_weights("./models/weights/weights.h5")
        print("successfull saved the model and saved the weights....")
        
    def on_press_button2(self,instance):
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
        # cnn.save_weights("./models/weights/weights.h5")
        print("successfull saved the model and saved the weights....")
    
    def on_press_button2(self,instance):
        print("You pressed the button2")
        import os
        import requests
        import shutil
        url = 'https://google.com'
        try:
            r = requests.get(url, stream=True)
        except:
            pass
        if r.status_code == 200:
            with open('global_model.h5', 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)
        print("Downloaded the global model....")
        # load models into a list of models from a directory


if __name__ == '__main__':
    app = MainApp()
    app.run()