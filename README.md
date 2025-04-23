# Plant Disease Prediction

This project is a **Streamlit web application** for predicting plant diseases from leaf images using a deep learning model based on **VGG16**. The model is trained on the **Plant Village dataset**, which includes images of healthy and diseased plant leaves across various species (e.g., Potato, Tomato). The app allows users to upload an image and receive a prediction of the disease along with confidence scores.

The app is deployed using a Streamlit Community Cloud. Models (`best_model.h5` and `best_model_fine.h5`) are hosted externally on Google Drive due to their large size.

## Features
- Upload a leaf image (JPG, JPEG, PNG) to predict plant diseases.
- Select between two models: `best_model.h5` (base model) and `best_model_fine.h5` (fine-tuned model).
- Displays predicted disease and confidence score.
- Option to view probabilities for all classes.
- Built with **Streamlit**, **TensorFlow**, and **VGG16**.

## Dataset
The model is trained on the **Plant Village dataset**, split into training, validation, and test sets using Split Folders. The dataset includes classes such as:
- Potato_early_blight
- Potato_healthy
- Tomato_healthy
- Tomato_late_blight
- ... (full list in `app.py`)

