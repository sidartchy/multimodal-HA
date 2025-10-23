# Skin Lesion Classifier - Streamlit App

## 🏥 AI-powered diagnostic assistant for skin lesion analysis

This application provides a user-friendly interface for analyzing skin lesion images using deep learning models with support for multiple input modes.

## 🚀 Quick Start

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run streamlit_app.py
```

### Usage
The app supports three input modes:

1. **🖼️ Image Only**: Upload a skin lesion image for CNN-based analysis
2. **📝 Text Only**: Provide clinical information for text-based analysis  
3. **🖼️📝 Image + Text**: Combine both image and text for multimodal analysis

## 📊 Features

- **Multiple Input Modes**: Image-only, text-only, or multimodal analysis
- **Real-time Prediction**: Instant analysis with confidence scores
- **Interactive Visualization**: Confidence breakdown with progress bars
- **Model Information**: View model architecture and training details
- **Medical Disclaimers**: Appropriate warnings for medical use

## ⚠️ Medical Disclaimer

This AI tool is for educational and research purposes only. Always consult with qualified healthcare professionals for medical diagnosis and treatment decisions.

## 📁 Project Structure

```
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt         # Python dependencies
├── modules/                 # Custom utility modules
│   ├── data_utils.py        # Data loading and preprocessing
│   └── model_utils.py        # Model definitions and training
├── models/                  # Trained model weights
├── processed_data/          # Preprocessed data and configs
└── embeddings/              # Text embeddings and vectorizers
```

## 🔧 Development

The application is built using:
- **PyTorch**: Deep learning framework
- **Streamlit**: Web application framework
- **scikit-learn**: Machine learning utilities
- **PIL/OpenCV**: Image processing

## 📈 Model Performance

The application uses a ResNet-34 CNN model trained on the PAD-UFES-20 dataset for skin lesion classification.

## 🎯 Input Modes Explained

### Image-Only Mode
- Uses CNN to analyze visual features
- Best for clear, high-quality images
- Focuses on lesion appearance and morphology

### Text-Only Mode  
- Uses clinical information and symptoms
- Best when images are not available
- Relies on patient history and description

### Multimodal Mode
- Combines image and text information
- Most comprehensive approach
- Provides highest accuracy
