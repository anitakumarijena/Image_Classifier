import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import json
from PIL import Image
import os
import argparse
import helper

#parser initialization
parser = argparse.ArgumentParser()
parser.add_argument("image_path", help="The path of image file")
parser.add_argument("saved_model", help="Keras model file")
parser.add_argument("--top_k", type = int, help="Top k number of classes shown", default = 5)
parser.add_argument("--category_names", help="JSON file with the labels for classes", default = 'label_map.json')
args = parser.parse_args()

#Used for getting the arguments
image_path = args.image_path
model_file = args.saved_model
top_k = args.top_k

class_names = {}
if args.category_names:
    category_names = args.category_names
    with open(category_names, 'r') as f:
        class_names = json.load(f)  
        
#loading the saved model
model = tf.keras.models.load_model(model_file, custom_objects={'KerasLayer':hub.KerasLayer})

#predicting
probs, classes = helper.predict(image_path, model, top_k)

#printing the results
print(f"probabilities: {probs}")
if class_names:
    classes = [class_names[str(n+1)] for n in classes]
print(f"classes: {classes}")