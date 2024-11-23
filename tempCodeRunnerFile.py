import os
import re
import json
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from tkinter import filedialog
from PIL import Image, ImageTk
import customtkinter as ctk
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from image_downloading import download_image

# Load the trained model and class dictionary
model = tf.keras.models.load_model('new_seg_model.h5')
class_dict_df = pd.read_csv('class_dict.csv')

# Paths
file_dir = os.path.dirname(__file__)
prefs_path = os.path.join(file_dir, 'preferences.json')

# Default preferences
default_prefs = {
    'url': 'https://mt.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
    'tile_size': 256,
    'channels': 3,
    'dir': os.path.join(file_dir, 'images'),
    'headers': {
        'cache-control': 'max-age=0',
        'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="99", "Google Chrome";v="99"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'document',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-site': 'none',
        'sec-fetch-user': '?1',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.82 Safari/537.36'
    },
    'tl': '',
    'br': '',
    'zoom': ''
}

# Function to load and preprocess images
def load_image(image_path, is_mask=False):
    image = load_img(image_path, target_size=(256, 256))
    image = img_to_array(image) / 255.0
    if is_mask:
        image = np.argmax(image, axis=-1)  # Convert to single-channel integer labels
        image = np.expand_dims(image, axis=-1)  # Add channel dimension back
    return image

# Function to predict and annotate image
def predict_and_annotate(model, image_path):
    image = load_image(image_path)
    prediction = model.predict(tf.expand_dims(image, axis=0))[0]
    prediction = np.argmax(prediction, axis=-1)

    # Load class dictionary
    class_dict = {idx: tuple(row[['r', 'g', 'b']]) for idx, row in class_dict_df.iterrows()}
    name_dict = {tuple(row[['r', 'g', 'b']]): row['name'] for idx, row in class_dict_df.iterrows()}

    # Create an annotated image
    annotated_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    for idx, rgb in class_dict.items():
        mask = (prediction == idx)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            cv2.drawContours(annotated_image, [contour], -1, (0, 0, 0), 2)
            x, y, w, h = cv2.boundingRect(contour)
            name = name_dict[rgb]
            cv2.putText(annotated_image, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return annotated_image

# Function to browse and predict
def browse_and_predict():
    file_path = filedialog.askopenfilename()
    if file_path:
        annotated_image = predict_and_annotate(model, file_path)
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(annotated_image_rgb)
        img_tk = ImageTk.PhotoImage(image_pil)
        
        # Display image in label
        image_label.configure(image=img_tk)
        image_label.image = img_tk

# Function to handle map download with provided coordinates
def run(cord1, cord2, zoom):
    print("Starting the tile download process...")
    with open(prefs_path, 'r', encoding='utf-8') as f:
        prefs = json.loads(f.read())

    if not os.path.isdir(prefs['dir']):
        os.makedirs(prefs['dir'])

    if (prefs['tl'] == '') or (prefs['br'] == '') or (prefs['zoom'] == ''):
        prefs['tl'] = cord1
        prefs['br'] = cord2
        prefs['zoom'] = zoom

    lat1, lon1 = map(float, re.findall(r'[+-]?\d*\.\d+', prefs['tl']))
    lat2, lon2 = map(float, re.findall(r'[+-]?\d*\.\d+', prefs['br']))
    zoom = int(prefs['zoom'])
    channels = int(prefs['channels'])
    tile_size = int(prefs['tile_size'])

    # Download all surrounding tiles
    download_image(lat1, lon1, lat2, lon2, zoom, prefs['url'], prefs['headers'], tile_size, channels, prefs['dir'])

    # Assuming the image is saved at the path 'images/downloaded_image.jpg' after download
    downloaded_image_path = os.path.join(prefs['dir'], 'downloaded_image.jpg')
    
    if os.path.exists(downloaded_image_path):
        annotated_image = predict_and_annotate(model, downloaded_image_path)
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(annotated_image_rgb)
        img_tk = ImageTk.PhotoImage(image_pil)
        
        # Display image in label
        image_label.configure(image=img_tk)
        image_label.image = img_tk
    else:
        print(f"Image file not found at {downloaded_image_path}. Please check the download process.")


# Initialize the customtkinter window
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")
root = ctk.CTk()
root.title("Image Segmentation and Map Viewer")

# UI layout
root.geometry("1000x700")

# Coordinates Frame
coordinates_frame = ctk.CTkFrame(root)
coordinates_frame.pack(pady=10)

longitude1_label = ctk.CTkLabel(coordinates_frame, text="Coordinate 1")
longitude1_label.pack(side="left", padx=10)

longitude1_entry = ctk.CTkEntry(coordinates_frame, placeholder_text="Enter coord ex.(43.6400, -79.3910)")
longitude1_entry.pack(side="left", padx=10)

latitude1_label = ctk.CTkLabel(coordinates_frame, text="Coordinate 2:")
latitude1_label.pack(side="left", padx=10)

latitude1_entry = ctk.CTkEntry(coordinates_frame, placeholder_text="Enter coord ex. (43.6450, -79.3910 )")
latitude1_entry.pack(side="left", padx=10)

# Button to get coordinates and download map tiles
get_coordinates_button = ctk.CTkButton(root, text="Get Coordinates and Download", 
                                       command=lambda: run(longitude1_entry.get(), latitude1_entry.get(), 15))
get_coordinates_button.pack(pady=10)

# Browse Button
browse_button = ctk.CTkButton(root, text="Browse Image", command=browse_and_predict)
browse_button.pack(pady=20)

# Label to display annotated image
image_label = ctk.CTkLabel(root)
image_label.pack(pady=10)

# Check if preferences file exists, otherwise create default preferences
if not os.path.isfile(prefs_path):
    with open(prefs_path, 'w', encoding='utf-8') as f:
        json.dump(default_prefs, f, indent=2, ensure_ascii=False)

# Run the application
root.mainloop()
