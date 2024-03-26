from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from fileinput import filename
import tensorflow as tf
from keras.models import load_model
import os
from PIL import Image
import numpy as np
import cv2
import os

categories_labels = {'Fighting': 0, 'Shoplifting': 1, 'Burglary': 2, 'Arrest': 3, 'Shooting': 4, 'Robbery': 5, 'Stealing': 6}
labels_categories = {v: k for k, v in categories_labels.items()}  # reverse dictionary for label lookup

# Load the trained model
model = load_model('CNN_RNN.h5')

def predict_image(image):
    # Resize the image
    image = cv2.resize(image, (50, 50))

    # Reshape the image to 4D
    image_cnn = image.reshape((1,) + image.shape + (1,))
    image_rnn = image.reshape((1,) + (-1, 1))

    prediction = model.predict([image_cnn, image_rnn])
    label = np.argmax(prediction)

    return labels_categories[label]

def extract_and_predict(gif_path):

    gif = Image.open(gif_path)

    # Extract frames from the gif
    frames = []
    try:
        while True:
            gif.seek(gif.tell() + 1)
            frames.append(np.array(gif.convert('L')))  # Convert image to grayscale
    except EOFError:
        pass

    # Predict the category of each frame and save it in a new directory
    new_dir = 'predicted_images'
    os.makedirs(new_dir, exist_ok=True)
    last_category = None  # variable to keep track of the last predicted category
    for i, frame in enumerate(frames):
        category = predict_image(frame)
        if category != last_category:  # only print the category if it's different from the last one
            print('Frame', i, 'Category:', category)
        last_category = category  # update the last predicted category
        cv2.imwrite(os.path.join(new_dir, f'{category}_{i}.png'), frame)

# Test the function
# Enter your gif path
gif_path = "Test Data/test2.gif"
extract_and_predict(gif_path)