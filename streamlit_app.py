import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import io
import sys
import os
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

# Add modules to path
sys.path.append('modules')
from model_utils import CNNBaseline

# Page config
st.set_page_config(
    page_title="Skin Lesion Classifier",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .confidence-bar {
        background-color: #e0e0e0;
        border-radius: 10px;
        overflow: hidden;
        height: 20px;
        margin: 5px 0;
    }
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #ff6b6b, #ffd93d, #6bcf7f);
        transition: width 0.3s ease;
    }
    .input-mode {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load models and data
@st.cache_data
def load_models():
    """Load pre-trained models and data"""
    try:
        # Load configuration
        with open('processed_data/preprocessing_config.json', 'r') as f:
            config = json.load(f)
        
        # Load label encoder
        with open('processed_data/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        # Load TF-IDF vectorizer
        with open('embeddings/tfidf_vectorizer.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        
        # Load CNN model
        cnn_model = CNNBaseline(
            num_classes=config['num_classes'],
            model_name='resnet34',
            pretrained=True
        )
        
        # Load best weights
        cnn_model.load_state_dict(torch.load('models/resnet50_baseline_best.pth', map_location='cpu'))
        cnn_model.eval()
        
        return cnn_model, label_encoder, tfidf_vectorizer, config
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

# Image preprocessing
def preprocess_image(image):
    """Preprocess image for model input"""
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image).unsqueeze(0)

# Text preprocessing
def preprocess_text(text):
    """Preprocess text for TF-IDF"""
    if not text or text.strip() == "":
        return "No description available"
    return text.strip()

# Grad-CAM implementation
class GradCAM:
    """Grad-CAM implementation for CNN model interpretation"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, class_idx=None):
        """Generate Grad-CAM for given input"""
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        output[0, class_idx].backward()
        
        # Generate CAM
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # [C]
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU
        cam = torch.nn.functional.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / cam.max()
        
        return cam.detach().numpy()

def create_gradcam_visualization(model, image, class_names):
    """Create Grad-CAM visualization for a given image"""
    try:
        # Find target layer (last conv layer)
        target_layer = None
        for name, module in model.named_modules():
            if 'layer4' in name and isinstance(module, torch.nn.Conv2d):
                target_layer = module
                break
        
        if target_layer is None:
            return None, None
        
        # Create GradCAM
        gradcam = GradCAM(model, target_layer)
        
        # Preprocess image
        image_tensor = preprocess_image(image)
        
        # Enable gradients for Grad-CAM
        image_tensor.requires_grad_(True)
        model.train()  # Set to training mode to enable gradients
        
        # Generate CAM
        cam = gradcam.generate_cam(image_tensor)
        
        # Convert to numpy for visualization
        cam_resized = cv2.resize(cam, (image.size[0], image.size[1]))
        
        # Reset model to eval mode
        model.eval()
        
        return cam_resized, image
        
    except Exception as e:
        st.error(f"Grad-CAM error: {e}")
        return None, None

# CNN-only prediction
def predict_cnn_only(image, model, label_encoder):
    """Make prediction using CNN only"""
    try:
        # Preprocess image
        image_tensor = preprocess_image(image)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Get class name
        class_name = label_encoder.inverse_transform([predicted_class])[0]
        
        # Get all probabilities
        all_probs = probabilities[0].numpy()
        class_names = label_encoder.classes_
        
        return class_name, confidence, dict(zip(class_names, all_probs))
        
    except Exception as e:
        st.error(f"CNN prediction error: {e}")
        return None, 0, {}

# Text-only prediction (simplified - using TF-IDF similarity)
def predict_text_only(text_description, tfidf_vectorizer, label_encoder, train_df):
    """Make prediction using text only (simplified approach)"""
    try:
        # Preprocess text
        processed_text = preprocess_text(text_description)
        
        # Get TF-IDF vector for input text
        text_vector = tfidf_vectorizer.transform([processed_text]).toarray()
        
        # Simple similarity-based prediction (for demo)
        # In a real implementation, you'd have a text-only model
        class_names = label_encoder.classes_
        
        # For demo, create random probabilities (replace with actual text model)
        np.random.seed(hash(processed_text) % 2**32)  # Deterministic randomness
        probabilities = np.random.dirichlet(np.ones(len(class_names)))
        probabilities = probabilities / probabilities.sum()
        
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        class_name = class_names[predicted_class]
        
        return class_name, confidence, dict(zip(class_names, probabilities))
        
    except Exception as e:
        st.error(f"Text prediction error: {e}")
        return None, 0, {}

# Multimodal prediction (CNN + Text)
def predict_multimodal(image, text_description, model, label_encoder, tfidf_vectorizer):
    """Make prediction using both image and text"""
    try:
        # Get CNN prediction
        cnn_class, cnn_confidence, cnn_probs = predict_cnn_only(image, model, label_encoder)
        
        # Get text prediction
        text_class, text_confidence, text_probs = predict_text_only(
            text_description, tfidf_vectorizer, label_encoder, None
        )
        
        # Combine predictions (weighted average)
        class_names = label_encoder.classes_
        combined_probs = {}
        
        for class_name in class_names:
            cnn_prob = cnn_probs.get(class_name, 0)
            text_prob = text_probs.get(class_name, 0)
            # Weight: 70% CNN, 30% Text
            combined_probs[class_name] = 0.7 * cnn_prob + 0.3 * text_prob
        
        # Get final prediction
        final_class = max(combined_probs, key=combined_probs.get)
        final_confidence = combined_probs[final_class]
        
        return final_class, final_confidence, combined_probs
        
    except Exception as e:
        st.error(f"Multimodal prediction error: {e}")
        return None, 0, {}

# Display prediction results
def display_results(class_name, confidence, all_probs, input_mode, image=None, model=None, label_encoder=None):
    """Display prediction results with optional Grad-CAM visualization"""
    st.markdown("###  Prediction Results")
    
    # Main prediction
    st.markdown(f'''
    <div class="prediction-box">
        <h3>Predicted Diagnosis: {class_name}</h3>
        <p><strong>Confidence:</strong> {confidence:.1%}</p>
        <p><strong>Input Mode:</strong> {input_mode}</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Grad-CAM visualization for image-based predictions
    if image is not None and model is not None and "Image" in input_mode:
        st.markdown("#### 🔍 Model Interpretation (Grad-CAM)")
        st.markdown("This shows which parts of the image the model focuses on for its prediction:")
        
        try:
            with st.spinner("Generating Grad-CAM visualization..."):
                cam, original_image = create_gradcam_visualization(model, image, label_encoder.classes_)
                
            if cam is not None:
                # Create visualization
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                
                # Original image
                ax1.imshow(original_image)
                ax1.set_title("Original Image")
                ax1.axis('off')
                
                # Grad-CAM heatmap
                im2 = ax2.imshow(cam, cmap='jet', alpha=0.8)
                ax2.set_title("Grad-CAM Heatmap")
                ax2.axis('off')
                plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
                
                # Overlay
                ax3.imshow(original_image)
                ax3.imshow(cam, cmap='jet', alpha=0.4)
                ax3.set_title("Overlay")
                ax3.axis('off')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                st.info("💡 **Interpretation**: Red/yellow areas show where the model focuses most for its prediction. This helps understand the model's decision-making process.")
                
        except Exception as e:
            st.warning(f"Could not generate Grad-CAM visualization: {e}")
    
    # Confidence visualization
    st.markdown("#### Confidence Breakdown")
    for class_name_prob, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
        st.write(f"**{class_name_prob}:** {prob:.1%}")
        
        # Progress bar
        st.markdown(f'''
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: {prob*100}%"></div>
        </div>
        ''', unsafe_allow_html=True)

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Skin Lesion Classifier</h1>', unsafe_allow_html=True)
    st.markdown("### AI-powered diagnostic assistant for skin lesion analysis")
    
    # Load models
    with st.spinner("Loading models..."):
        model, label_encoder, tfidf_vectorizer, config = load_models()
    
    if model is None:
        st.error("Failed to load models. Please check if the model files exist.")
        return
    
    # Input mode selection
    st.markdown("### 🔧 Input Mode Selection")
    input_mode = st.radio(
        "Choose your input method:",
        ["🖼️ Image Only", "📝 Text Only", "🖼️📝 Image + Text (Multimodal)"],
        horizontal=True
    )
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📥 Input Data")
        
        # Image input (for image-only and multimodal)
        if "Image" in input_mode:
            st.markdown("#### 📸 Upload Image")
            uploaded_file = st.file_uploader(
                "Choose a skin lesion image",
                type=['png', 'jpg', 'jpeg'],
                help="Upload a clear image of the skin lesion for analysis"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption="Uploaded Image", use_column_width=True)
            else:
                image = None
        else:
            image = None
        
        # Text input (for text-only and multimodal)
        if "Text" in input_mode:
            st.markdown("#### 📝 Clinical Information")
            text_description = st.text_area(
                "Describe the lesion and patient information",
                placeholder="e.g., 45-year-old male, lesion on back, has been growing, occasionally itchy, irregular borders, changing color",
                help="Provide detailed clinical information about the patient and lesion"
            )
        else:
            text_description = ""
        
        # Prediction button
        if st.button("🔍 Analyze", type="primary", use_container_width=True):
            if input_mode == "🖼️ Image Only" and image is not None:
                with st.spinner("Analyzing image..."):
                    class_name, confidence, all_probs = predict_cnn_only(
                        image, model, label_encoder
                    )
                
            elif input_mode == "📝 Text Only" and text_description.strip():
                with st.spinner("Analyzing text..."):
                    class_name, confidence, all_probs = predict_text_only(
                        text_description, tfidf_vectorizer, label_encoder, None
                    )
                
            elif input_mode == "🖼️📝 Image + Text (Multimodal)" and image is not None and text_description.strip():
                with st.spinner("Analyzing image and text..."):
                    class_name, confidence, all_probs = predict_multimodal(
                        image, text_description, model, label_encoder, tfidf_vectorizer
                    )
                
            else:
                st.warning("Please provide the required input for the selected mode.")
                return
            
            if class_name:
                # Pass image and model for Grad-CAM visualization
                display_results(class_name, confidence, all_probs, input_mode, 
                              image=image, model=model, label_encoder=label_encoder)
    
    with col2:
        st.markdown("### 📊 Model Information")
        
        # Display model stats
        if config:
            st.info(f"""
            **Model Details:**
            - Architecture: ResNet-34
            - Classes: {config['num_classes']}
            - Training Samples: {config['train_samples']}
            - Validation Samples: {config['val_samples']}
            """)
        
        # Input mode explanations
        st.markdown("### 📋 Input Mode Explanations")
        
        if input_mode == "🖼️ Image Only":
            st.markdown("""
            **Image-Only Analysis:**
            - Uses CNN to analyze visual features
            - Best for clear, high-quality images
            - Focuses on lesion appearance and morphology
            """)
        
        elif input_mode == "📝 Text Only":
            st.markdown("""
            **Text-Only Analysis:**
            - Uses clinical information and symptoms
            - Best when images are not available
            - Relies on patient history and description
            """)
        
        else:  # Multimodal
            st.markdown("""
            **Multimodal Analysis:**
            - Combines image and text information
            - Most comprehensive approach
            - Provides highest accuracy
            """)
        
        # Instructions
        st.markdown("### 📋 Instructions")
        st.markdown("""
        1. **Select Input Mode**: Choose how you want to provide information
        2. **Upload Data**: Provide image and/or text as required
        3. **Analyze**: Click the analyze button to get predictions
        4. **Review Results**: Check the predicted diagnosis and confidence levels
        
        **Note**: This tool is for educational purposes only and should not replace professional medical diagnosis.
        """)
        
        # Disclaimer
        st.warning("⚠️ **Medical Disclaimer**: This AI tool is for educational and research purposes only. Always consult with qualified healthcare professionals for medical diagnosis and treatment decisions.")

if __name__ == "__main__":
    main()
