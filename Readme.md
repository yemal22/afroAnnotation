# 🌍 AfroVision: Captioning Images with African Soul

[![Made in Africa](https://img.shields.io/badge/Made%20in-Africa-darkgreen?style=flat-square)](https://github.com/yemal22/afroAnnotation)  [![Dataset](https://img.shields.io/badge/Huggingface-African--Food-orange?logo=huggingface&style=flat-square)](https://huggingface.co/datasets/yemalin/african-food)  ![LoRA Fine-tuned](https://img.shields.io/badge/Model-LoRA%20fine--tuned-lightblue?style=flat-square&logo=openai)  [![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-red?logo=streamlit&style=flat-square)](https://your-app-link.streamlit.app)  ![Tests](https://img.shields.io/badge/tests-passing-brightgreen?style=flat-square&logo=pytest)

AfroVision is a powerful and culturally-aware image captioning system designed to generate **natural language descriptions** for African-themed images, especially in the domains of:

- 🥘 **African Cuisine**  
- 👗 **African Fashion**

This project blends the power of **AI** with the richness of **African heritage**, using custom fine-tuned models and a beautifully styled interface to bring image annotation to life.

---

## ✨ Features

- 📷 Upload or provide a URL to your image
- 🧠 Auto-caption generation using a BLIP model fine-tuned with LoRA
- 🧵 Two main categories: `Fashion` and `Food`
- 🌐 REST API via **FastAPI**
- 🎨 Interactive web interface via **Streamlit**, with African-themed colors
- 📦 Easy setup with bash scripts
- 🔐 Deployable to the web with `ngrok` or `HTTPS tunnels`

---

## 📁 Project Structure

```
afroAnnotation/
├── app/
│   ├── main.py                     # FastAPI application
│   └── afro_vision_ui.py           # Streamlit interface
├── launch_servers.sh               # Script to start both servers
├── install_dependencies.sh         # Script to install Python dependencies
├── requirements.txt                # List of required Python packages
├── README.md                       # You are here 🌟
├── ...
├── models/                         # Models 
|   ├── blip-afro-fashion-v1.0.0
|   └── blip-afro-food-v1.0.0
└── data/                           # Datasets (optional, .gitignored)
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yemal22/afroAnnotation.git
cd afroAnnotation
```

### 2. Set up the Python Environment

```bash
python3 -m venv annot_venv
source annot_venv/bin/activate
```

### 3. Install Dependencies

```bash
./install_dependencies.sh
```

Or manually:

```bash
pip install -r requirements.txt
```

### 4. Launch the App 🚀

```bash
./launch_servers.sh
```

- FastAPI runs on: `http://localhost:8000`
- Streamlit UI runs on: `http://localhost:8501`

---

## 🔌 API Usage

**Endpoint:** `POST /afro/food` or `/afro/fashion`

### ➕ Example Request (Upload file)

```bash
curl -X POST http://localhost:8000/afro/food \
  -F "file=@sample.jpg"
```

### ➕ Example Request (Image URL)

```bash
curl -X POST http://localhost:8000/afro/fashion \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://link-to-image.com/image.jpg"}'
```

### ✅ Response

```json
{
  "caption": "A woman wearing a colorful traditional Kente dress with matching head wrap"
}
```

---

## 🎨 Custom UI Styling

The Streamlit interface is built with care to reflect **African visual identity** using:

- 🌅 Warm tones and natural palettes
- ✨ Interactive layout and transitions

---

## 🌍 Deployment (Optional)

To share your app securely:

```bash
pip install pyngrok
ngrok http 8501
```

Copy the `https://...` URL and share it with others!

---

## 🧠 Model Training & Fine-Tuning

We use the `BLIP` (Bootstrapping Language-Image Pretraining) model, enhanced with:

- 🔄 LoRA (Low-Rank Adaptation)
- 📦 Custom Huggingface Datasets for African food and fashion
- 🧪 Training on Azure VM (optional setup details available)

If you want to fine-tune it yourself, check the training script (coming soon).

---

## 👨🏽‍💻 Author

**Yémalin Morel KPAVODE**  
AI & Data Science Enthusiast | Innovating for Africa 🌍  
📧 yemalem03@gmail.com  
📞 +229 01 95 75 41 57  

---

## 🤝 Contributions

You’re welcome to improve this project!  
Please open an issue or create a PR if you have ideas around:

- 💬 Multilingual captions (e.g. Swahili, Yoruba, Wolof)
- 📚 New datasets (e.g. African architecture, wildlife, ceremonies)
- 💡 UI/UX improvements
- 🐍 Model optimization

---

## 📜 License

MIT License. Free to use with acknowledgment.  
Let’s build for Africa, together. ✨

```

