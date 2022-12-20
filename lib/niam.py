# from kivy.app import App
# from kivy.uix.button import Button
from keras.datasets import cifar10
import tensorflow as tf
import numpy as np
from keras import models
from keras import layers
from keras.models import Sequential

from tensorflow import keras

# class MainApp(App):
#     def build(self):
#         button = Button(text='Hello from Kivy',
#                         size_hint=(.5, .5),
#                         pos_hint={'center_x': .5, 'center_y': .5})
#         button.bind(on_press=self.on_press_button)

#         return button

#     def on_press_button(self, instance):
        
#         print('You pressed the button!')
        
#         cnn=models.load_model("./models/models.h5")
    
#         weights = cnn.get_weights
        
        
# if __name__ == '__main__':
#     app = MainApp()
#     app.run()


cnn = models.load_model("./models/models.h5")
weights = cnn.get_weights()
np_weights = np.array(weights)
np_weights_divided = np_weights/2
print(np_weights[0][0][0]-np_weights_divided[0][0][0])
cnn.load_weights(weights)
