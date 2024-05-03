import argparse
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers 
from matplotlib import pyplot as plt
from PIL import Image, ImageOps
import os
import numpy as np

#parser to get test data path
parser = argparse.ArgumentParser( description='Evaluate the model')
parser.add_argument('--test_data_path', required=True, type=str, help='Path to test data')
parser.add_argument('--model_path', required=True ,type=str, help='Path to model')
args = parser.parse_args()


if "logistic_regression" in args.model_path:
    IMAGE_SIZE = 256
else:
    IMAGE_SIZE = 128
print("---------------------------------------------")
print("Reading images from: ", args.test_data_path)
# parse images
def load_images_from_folder(folder, desired_size=IMAGE_SIZE):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        if img is not None:
            # Calculate padding
            pixels = np.array(img)
            min_pixel, max_pixel = pixels.min(), pixels.max()
            padding = int((max_pixel - min_pixel) / 10)  # Example padding calculation

            # Add padding to the image
            img = ImageOps.expand(img, border=padding, fill='black')
            
            # convert to rgb
            if desired_size == 128:
                img = img.convert('RGB')

            # Resize to the desired size
            img = img.resize((desired_size, desired_size))

            images.append(np.array(img)) 
            labels.append(filename)
    return images, labels
images, image_names = load_images_from_folder(args.test_data_path)

print("Images parsed, loading model...")

# Load the model
model = tf.keras.models.load_model(args.model_path)

print("---------------------------------------------")

print("Images loaded, predicting...")
y_pred = model.predict(np.array(images))
y_pred = np.round(y_pred)

evaluation_data = pd.DataFrame({'image_name': image_names, 'prediction': y_pred[:, 0].astype(int)})
evaluation_data.to_csv('evaluation.csv', index=False)
print("---------------------------------------------")




