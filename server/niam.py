import os
from kivy.app import App
from kivy.uix.button import Button
from keras.datasets import cifar10
import tensorflow as tf
import numpy as np
from keras import models
from keras import layers
from keras.models import Sequential

from tensorflow import keras

class MainApp(App):
    def build(self):
        button = Button(text='Click to create the global model',
                        size_hint=(.5, .5),
                        pos_hint={'center_x': .5, 'center_y': .5})
        button.bind(on_press=self.on_press_button)

        return button

    def on_press_button(self, instance):
        
        print('You pressed the button!')
        # load models into a list of models from a directory


        # model1= models.load_model("./models/models.h5")
        # model2= models.load_model("./models/model2.h5")
        
        # new_models = [(w1+w2)/3 for (w1,w2) in zip(model1.get_weights(),model2.get_weights())]
        # model1.set_weights(new_models)
        # model1.save("./models/model3.h5")
        k=os.listdir("../lib/models")
        print(k)
        model_grp=[]
        for i in k:
            m=models.load_model("../lib/models/"+i)
            model_grp.append(m)

        weights=[]
        for i in model_grp:
            weight = i.get_weights()
            weights.append(weight)

        l=len(weights)
        new_weights= [sum(x) for x in zip(*weights)]
        new_weights=[x/l for x in new_weights]
        model_grp[0].set_weights(new_weights)
        model_grp[0].save("../lib/models/global_model.h5")
        print("Ceated the global model....")

if __name__ == '__main__':
    app = MainApp()
    app.run()





