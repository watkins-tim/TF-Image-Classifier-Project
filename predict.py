import numpy as np
import tensorflow as tf 
import tensorflow_hub as hub
from PIL import Image
import heapq
import json
import argparse
import os
import logging


import warnings


# Check if a path is a valid filepath
def is_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist. Please enter a valid file path." % arg)
    else:
        return open(arg, 'r')

# Get arguments from command line
def get_arguments():
    parser = argparse.ArgumentParser(description='Flower Image Classifier')
    parser.add_argument('filename', help='Enter input image filepath', metavar='Input File Path', type=lambda x: is_file(parser, x))
    parser.add_argument('model', help='Enter the model filepath', metavar='Model File Path', type=lambda x: is_file(parser, x))
    parser.add_argument('--top_k', type=int, default=1, metavar='Number of classes to return')
    parser.add_argument('--category_names', required=False, help='Enter json file of flower classnames', metavar='Classnames File Path', type=lambda x:is_file(parser ,x))
    args = parser.parse_args()
    return args

# Resize and normalize image
def process_image(image_array):
    image = tf.convert_to_tensor(image_array)
    image = tf.image.resize(image, (224,224))
    image = image/255
    return image.numpy() 

# Predict the class of input image, return the predicted class(es)
def predict(image_path, model, top_k):
    im = Image.open(image_path)
    image = np.asarray(im)
    processed_image = process_image(image)
    p = model.predict(np.expand_dims(processed_image,axis=0))
    indicies = heapq.nlargest(top_k, range(len(p[0])), p[0].__getitem__)
    preds = p[0][indicies]
    indicies = list(np.asarray(indicies) + 1)
    return list(map(str,indicies)), preds

# Main Method
if __name__ == '__main__':

    #efforts to suppress verbose logging
    logger = tf.get_logger()
    logger.setLevel(logging.ERROR)
    warnings.filterwarnings('ignore')
    tf.get_logger().setLevel('ERROR')
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    args = get_arguments()
    print(args.category_names)

    #load class names file if provided
    class_names = ''
    if (args.category_names):
        with open(args.category_names.name, 'r') as f:
            class_names = json.load(f)

    model_path = args.model.name
    loaded_model =tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer})
    classes, probabilities = predict(args.filename.name, loaded_model, args.top_k)

    #print class and probability for top_k values
    for i in range(len(classes)):
        out = ''
        if (class_names != ''):
            out += class_names[classes[i]]
        else: 
            out += classes[i]
        out += ': '
        out += str(probabilities[i])
        print(out)
    