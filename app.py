import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd

# Load the trained model
model = tf.keras.models.load_model('model.h5')

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
    uploaded_file = st.file_uploader("Upload Fingerprint Image", type=["jpg", "jpeg", "png","bmp"])
    
    submitted = st.form_submit_button("Predict Blood Group")

if submitted:
    if uploaded_file is not None:
        # Load and preprocess image
        image = Image.open(uploaded_file).convert("RGB")
        image_resized = image.resize((64, 64))
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
