import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV files
dataset_df = pd.read_csv('metadata.csv')
class_dict_df = pd.read_csv('class_dict.csv')

# Ensure that the paths are strings
dataset_df['sat_image_path'] = dataset_df['sat_image_path'].astype(str)

# Function to load and preprocess images
def load_image(image_path, is_mask=False):
    image = load_img(image_path, target_size=(256, 256))
    image = img_to_array(image) / 255.0
    if is_mask:
        image = np.argmax(image, axis=-1)  # Convert to single-channel integer labels
        image = np.expand_dims(image, axis=-1)  # Add channel dimension back
    return image

# Create dataset
def create_dataset(df, is_train=True):
    images = []
    masks = []
    for idx, row in df.iterrows():
        sat_image_path = row['sat_image_path']
        if is_train:
            mask_path = row.get('mask_path', None)  # Use get to handle missing mask column
            if pd.notnull(sat_image_path) and pd.notnull(mask_path):
                try:
                    image = load_image(sat_image_path)
                    mask = load_image(mask_path, is_mask=True)
                    images.append(image)
                    masks.append(mask)
                except FileNotFoundError:
                    print(f"File not found: {sat_image_path} or {mask_path}")
        else:
            if pd.notnull(sat_image_path):
                try:
                    image = load_image(sat_image_path)
                    images.append(image)
                except FileNotFoundError:
                    print(f"File not found: {sat_image_path}")
    if is_train and len(images) > 0:
        images = np.array(images)
        masks = np.array(masks)
        return tf.data.Dataset.from_tensor_slices((images, masks)).map(lambda x, y: (tf.reshape(x, [256, 256, 3]), tf.reshape(y, [256, 256, 1])))
    elif not is_train and len(images) > 0:
        images = np.array(images)
        return tf.data.Dataset.from_tensor_slices(images).map(lambda x: tf.reshape(x, [256, 256, 3]))
    elif is_train:
        raise ValueError("Training dataset contains no valid images/masks")
    else:
        return None

# Split dataset
train_df = dataset_df[dataset_df['split'] == 'train']
test_df = dataset_df[dataset_df['split'] == 'test']
val_df = dataset_df[dataset_df['split'] == 'validate']

print("Training set:")
print(train_df.head())
print("Validation set:")
print(val_df.head())
print("Test set:")
print(test_df.head())

train_dataset = create_dataset(train_df, is_train=True)
val_dataset = create_dataset(val_df, is_train=False)
test_dataset = create_dataset(test_df, is_train=False)

def unet_model(input_size=(256, 256, 3)):
    inputs = Input(input_size)
    c1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D(pool_size=(2, 2))(c1)
    c2 = Conv2D(128, 3, activation='relu', padding='same')(p1)
    p2 = MaxPooling2D(pool_size=(2, 2))(c2)
    c3 = Conv2D(256, 3, activation='relu', padding='same')(p2)
    p3 = MaxPooling2D(pool_size=(2, 2))(c3)
    c4 = Conv2D(512, 3, activation='relu', padding='same')(p3)
    u1 = UpSampling2D(size=(2, 2))(c4)
    m1 = concatenate([u1, c3], axis=3)
    c5 = Conv2D(256, 3, activation='relu', padding='same')(m1)
    u2 = UpSampling2D(size=(2, 2))(c5)
    m2 = concatenate([u2, c2], axis=3)
    c6 = Conv2D(128, 3, activation='relu', padding='same')(m2)
    u3 = UpSampling2D(size=(2, 2))(c6)
    m3 = concatenate([u3, c1], axis=3)
    c7 = Conv2D(64, 3, activation='relu', padding='same')(m3)
    outputs = Conv2D(len(class_dict_df), 1, activation='softmax', padding='same')(c7)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = unet_model()

# Training the model only if training and validation datasets are not None
if train_dataset:
    if val_dataset:
        model.fit(train_dataset.batch(16), validation_data=val_dataset.batch(16), epochs=20)
    else:
        model.fit(train_dataset.batch(16), epochs=20)

# Function to predict and annotate image
def predict_and_annotate(model, image_path):
    image = load_image(image_path)
    prediction = model.predict(tf.expand_dims(image, axis=0))[0]
    prediction = np.argmax(prediction, axis=-1)

    # Load class dictionary
    class_dict = {tuple(row[['r', 'g', 'b']]): row['name'] for idx, row in class_dict_df.iterrows()}

    # Create an annotated image
    annotated_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    for (r, g, b), name in class_dict.items():
        mask = (prediction == np.array([r, g, b])).all(axis=-1)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            cv2.drawContours(annotated_image, [contour], -1, (0, 0, 0), 2)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.putText(annotated_image, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    plt.show()

model.save('new_seg_model.h5')

# Example usage
# predict_and_annotate(model, 'new_sat_image_path')