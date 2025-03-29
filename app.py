import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from PIL import Image

# Load pre-trained logistic regression model and scaler
@st.cache_resource
def load_model():
    model = joblib.load("smile_stalker.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

def preprocess_image(image):
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((64, 64))  # Resize using PIL
    image = np.array(image).flatten().reshape(1, -1)  # Flatten for logistic regression
    image = scaler.transform(image)  # Apply standard scaler
    return image

st.title("Smile Stalker - Detect Smiles in Images")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open image using PIL
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image")
    
    # Preprocess image and make prediction
    processed_img = preprocess_image(img)
    prediction_proba = model.predict_proba(processed_img)[0][1]  # Get probability of smiling
    prediction = model.predict(processed_img)
    
    # Display result
    smile_score = int(prediction_proba * 100)
    st.slider("Smile Score", 0, 100, smile_score, disabled=True)
    
    if prediction[0] == 1:
        st.success(f"😊 The person is smiling! (Confidence: {smile_score}%)")
    else:
        st.warning(f"😐 The person is not smiling. (Confidence: {smile_score}%)")
