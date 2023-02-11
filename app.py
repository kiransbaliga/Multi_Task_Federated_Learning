from flask import Flask, render_template, request,jsonify
import numpy as np

import os

from keras.datasets import cifar10
import tensorflow as tf
import numpy as np
from keras import models
from keras import layers
from keras.models import Sequential

from tensorflow import keras
from base64 import decodebytes
app = Flask(__name__)
# model = load_model('model.h5')
# model.make_predict_function()

def make_global_model():
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
# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Flask app....."

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		try:
		
			make_global_model()
			
		#return label
			return jsonify("Done")
		except Exception as e:
			return e
	else:
		return "Error"



if __name__ =='__main__':
	app.run(debug=True)