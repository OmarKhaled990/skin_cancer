import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model (best_skin.pt)
try:
    model = YOLO('best_skin.pt')  # replace with your model path if different
except ImportError as e:
    st.error(
        "Failed to import YOLO. Make sure `ultralytics` and `opencv-python-headless` are installed."
    )
    st.stop()

# App title
st.title("YOLOv8 Model Demo")

# File uploader allows only image types
uploaded_file = st.file_uploader(
    "Upload an image", type=['jpg', 'jpeg', 'png']
)

if uploaded_file:
    # Open the image with PIL
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    st.write("Running inference...")

    # Perform inference
    results = model.predict(source=np.array(image))
    result = results[0]

    # If classification probabilities are returned, show top-1 class
    if result.probs is not None:
        top_idx = result.probs.top1
        class_name = result.names[top_idx]
        confidence = result.probs.top1conf
        st.success(f"**Classification:** {class_name} ({confidence * 100:.2f}%)")
    # Otherwise, treat as detection: draw boxes
    elif hasattr(result, 'boxes') and result.boxes:
        annotated = result.plot()
        st.image(annotated, caption='Detections', use_container_width=True)
    else:
        st.error(
            "No valid output returned. Please ensure your `best_skin.pt` is a valid YOLOv8 classification or detection model."
        )
