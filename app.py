import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Set protobuf fix
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Define image size (must match training)
IMG_SIZE = (224, 224)

# Get class labels from dataset
train_dir = 'split_data/train'  # Update path if needed
try:
    class_labels = sorted(os.listdir(train_dir))
    st.write(f"Number of classes: {len(class_labels)}")
    st.write(f"Class labels: {class_labels}")
except Exception as e:
    st.error(f"Error accessing dataset: {e}")
    st.stop()

# Streamlit app layout
st.title("Plant Disease Prediction")
st.write("Upload an image of a plant leaf to predict its disease.")

# Model selection
model_option = st.selectbox("Select Model", ["best_model.h5", "best_model_fine.h5"])
model_path = model_option

# Load the selected model
try:
    model = tf.keras.models.load_model(model_path)
    st.success(f"Model {model_option} loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Verify model output classes
model_output_classes = model.layers[-1].units
if model_output_classes != len(class_labels):
    st.error(f"Model expects {model_output_classes} classes, but dataset has {len(class_labels)} classes.")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    try:
        img = image.load_img(uploaded_file, target_size=IMG_SIZE)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = class_labels[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx]

        # Display results
        st.write(f"**Predicted Class**: {predicted_class}")
        st.write(f"**Confidence**: {confidence:.4f}")

        # Optionally show all probabilities
        if st.checkbox("Show all class probabilities"):
            prob_dict = {class_labels[i]: predictions[0][i] for i in range(len(class_labels))}
            st.write("**Class Probabilities**:")
            for cls, prob in prob_dict.items():
                st.write(f"{cls}: {prob:.4f}")

    except Exception as e:
        st.error(f"Error processing image or predicting: {e}")
