import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam

# Load the pretrained MobileNetV2 model and freeze its layers
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze the base model layers

# Add custom layers on top of MobileNetV2
inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
outputs = Dense(8, activation='softmax')(x)  # 8 classes for blood groups
model = Model(inputs, outputs)

# Compile the model
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Class names (update if necessary to match your dataset)
class_names = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']

# Streamlit app
st.title("Blood Group Detection from Fingerprint")

# User input fields
with st.form("user_form"):
    st.subheader("Enter your details")
    name = st.text_input("Name")
    age = st.number_input("Age", min_value=1, max_value=120, step=1)
    gender = st.radio("Gender", ["Male", "Female", "Other"])
    weight = st.number_input("Weight (kg)", min_value=1.0, max_value=300.0, step=0.1)
    uploaded_file = st.file_uploader("Upload Fingerprint Image", type=["jpg", "jpeg", "bmp"])
    
    submitted = st.form_submit_button("Predict Blood Group")

if submitted:
    if uploaded_file is not None:
        # Load and preprocess image
        image = Image.open(uploaded_file).convert("RGB")
        image_resized = image.resize((224, 224))  # Resize to 224x224 for MobileNetV2
        img_array = np.array(image_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]

        # Display success message
        st.success(f"Hello {name}! Based on the fingerprint, your predicted blood group is: **{predicted_class}**")

        # Display smaller image
        st.image(image, caption='Uploaded Fingerprint', width=150)

        # Show details in tabular format
        user_data = {
            "Name": [name],
            "Age": [age],
            "Gender": [gender],
            "Weight (kg)": [weight],
            "Predicted Blood Group": [predicted_class]
        }
        df = pd.DataFrame(user_data)
        st.write("### User Details:")
        st.table(df)
    else:
        st.error("Please upload a fingerprint image to proceed.")

    

       
