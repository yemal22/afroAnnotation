import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import tempfile
import os

st.set_page_config(
    page_title="AfroVision AI",
    layout="centered",
    page_icon="🌍"
)

# Title and Description with African theme
st.markdown("""
    <h1 style='text-align: center; color: #1e3d59;'>🌍 AfroVision AI</h1>
    <p style='text-align: center; color: #ff914d;'>Describe African fashion and food with the power of AI.</p>
""", unsafe_allow_html=True)

st.markdown("---")

# Sidebar for options
st.sidebar.header("Upload Options")
mode = st.sidebar.radio("Select mode", ["Upload Image", "Image URL"])

category = st.sidebar.selectbox("Choose Category", ["Fashion", "Food"])

image_bytes = None
image_path = None
image_url = ""

if mode == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            image_path = tmp.name
        image_bytes = uploaded_file.read()

elif mode == "Image URL":
    image_url = st.text_input("Enter image URL")
    if image_url:
        st.image(image_url, caption="Image from URL", use_column_width=True)

if st.button("Generate Caption"):
    if image_url or image_path:
        endpoint = "http://localhost:8000/afro/fashion" if category.lower() == "fashion" else "http://localhost:8000/afro/food"
        try:
            if mode == "Image URL":
                response = requests.post(endpoint, data={"image_url": image_url})
            else:
                with open(image_path, "rb") as img_file:
                    files = {"file": (os.path.basename(image_path), img_file, "image/jpeg")}
                    response = requests.post(endpoint, files=files)
            if response.status_code == 200:
                st.success("Caption generated:")
                st.markdown(f"**\"{response.json()['caption']}\"**")
            else:
                st.error(f"Error: {response.json()['detail']}")
        except Exception as e:
            st.error(f"Error calling API: {str(e)}")
        finally:
            if image_path and os.path.exists(image_path):
                os.remove(image_path)
    else:
        st.warning("Please provide an image.")
        
# Footer with African flair
st.markdown("""
    <hr/>
    <p style='text-align: center; color: gray;'>Made with ❤️ for Africa | By Morel Yémalin</p>
""", unsafe_allow_html=True)
