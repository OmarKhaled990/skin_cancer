import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Load the YOLOv8 classification model (best.pt)
model = YOLO('best_skin.pt')  # replace with your model path if different

# App title
st.title("YOLOv8 Classification Demo")

# File uploader allows only image types
uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    # Open the image with PIL
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")

    # Perform classification prediction (ensure task='classify')
    results = model.predict(source=image, task='classify')  # classification task
    result = results[0]

    # Check that probabilities are available
    if result.probs is not None:
        # The Probs object provides top1 and top1conf directly
        top_idx = result.probs.top1
        class_name = result.names[top_idx]
        confidence = result.probs.top1conf

        # Display the prediction
        st.write(f"**Prediction:** {class_name} ({confidence * 100:.2f}%)")
    else:
        st.error("No classification probabilities returned. Please ensure you're using a classification model and the correct task.")