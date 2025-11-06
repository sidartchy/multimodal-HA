import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
import pickle
import json
import os


import pandas as pd
import altair as alt
import sys
from transformers import AutoTokenizer, AutoModel

# Custom imports
sys.path.append("modules")
from model_utils import EfficientNetClassifier, HybridImageDominantFusion, EfficientNetBaseline

st.set_page_config(page_title="Skin Lesion Classifier", layout="wide", page_icon="🏥")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_biobert():
    # for text embeddings
    model_name = "emilyalsentzer/Bio_ClinicalBERT"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model

def get_bioclinical_embedding(text, tokenizer, model, max_len=128):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
    return emb


@st.cache_resource
def load_models():
    try:
        # Configs and encoders
        with open("processed_data_v2/preprocessing_config.json", "r") as f:
            config = json.load(f)
        with open("processed_data_v2/label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)

        # Image-only model
        cnn_model = EfficientNetClassifier(num_classes=config["num_classes"])
        cnn_model.load_state_dict(
            torch.load("models/efficient_net_base_best.pth", map_location=device)
        )
        cnn_model.to(device).eval()

        # Hybrid model (image + text)
        image_model = EfficientNetBaseline()
        hybrid_model = HybridImageDominantFusion(
            image_model=image_model,
            text_embedding_dim=768,
            image_embedding_dim=128,
            reduced_text_dim=32,
            num_classes=config["num_classes"],
            image_weight=0.8,
            fusion_type="weighted",
        )
        hybrid_model.load_state_dict(
            torch.load("models/hybrid_multimodal_best_f1.pth", map_location=device)
        )
        hybrid_model.to(device).eval()

        return cnn_model, hybrid_model, label_encoder, config
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor, class_idx=None):
        input_tensor = input_tensor.clone().detach()
        input_tensor.requires_grad_()   # ensure gradients

        # Enable gradients temporarily
        with torch.enable_grad():
            output = self.model(input_tensor)
            if class_idx is None:
                class_idx = output.argmax(dim=1)[0].item()

            self.model.zero_grad()
            output[0, class_idx].backward()

            gradients = self.gradients[0]  # [C,H,W]
            activations = self.activations[0]  # [C,H,W]
            weights = gradients.mean(dim=(1,2))
            cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)

            for i, w in enumerate(weights):
                cam += w * activations[i]

            cam = torch.nn.functional.relu(cam)
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().numpy()

    
def get_last_conv_layer(model):
    # Handles both wrapped and plain EfficientNet models
    if hasattr(model, "backbone"):
        return model.backbone.features[7]  # For Hybrid model (EfficientNetBaseline)
    else:
        return model.features[7]  # For plain EfficientNet model

def visualize_gradcam(image, cam):
    img_np = np.array(image)
    cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255*cam_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
    return overlay



# predict funstions
def predict_cnn(image, model, label_encoder):
    img_tensor = preprocess_image(image)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        pred_idx = probs.argmax()
    return label_encoder.inverse_transform([pred_idx])[0], probs

def predict_hybrid(image, text_emb, model, label_encoder):
    img_tensor = preprocess_image(image)
    text_tensor = torch.tensor(text_emb).unsqueeze(0).float().to(device)
    with torch.no_grad():
        outputs = model(img_tensor, text_tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        pred_idx = probs.argmax()
    return label_encoder.inverse_transform([pred_idx])[0], probs

# Interface
def main():
    st.title("🏥 Skin Lesion Classifier (Hybrid BioClinicalBERT + EfficientNet)")

    cnn_model, hybrid_model, label_encoder, config = load_models()
    tokenizer, bert_model = load_biobert()

    input_mode = st.radio("Select mode:", ["Image Only", "Image + Text"], horizontal=True)

    uploaded_file = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=300)
    else:
        image = None

    text_input = None
    if input_mode == "Image + Text":
        text_input = st.text_area("Enter clinical note:", "")

    if st.button("🔍 Predict"):
        if image is None:
            st.warning("Please upload an image first.")
            return

        #  Image Only 
        if input_mode == "Image Only":
            pred_class, probs = predict_cnn(image, cnn_model, label_encoder)
            st.success(f"**Prediction:** {pred_class}")
            
            

            # Confidence bar chart
            class_names = label_encoder.classes_
            conf_df = pd.DataFrame({
                "Class": class_names,
                "Confidence": probs
            })

            st.markdown("### 🔍 Model Confidence")
            bar_chart = (
                alt.Chart(conf_df)
                .mark_bar(size=40)
                .encode(
                    x=alt.X("Class:N", sort="-y"),
                    y=alt.Y("Confidence:Q", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("Class:N", legend=None),
                    tooltip=["Class", "Confidence"]
                )
                .properties(width=400, height=250)
            )
            st.altair_chart(bar_chart, use_container_width=True)


            # st.json({cls: f"{p:.1%}" for cls,p in zip(label_encoder.classes_, probs)})

            # GradCAM
            target_layer = get_last_conv_layer(cnn_model.backbone)
            gradcam = GradCAM(cnn_model.backbone, target_layer)
            img_tensor = preprocess_image(image)
            cam = gradcam.generate_cam(img_tensor)
            overlay = visualize_gradcam(image, cam)
            st.image(overlay, caption="Grad-CAM (Image Only)", width=300)

        #  Image + text
        else:
            if not text_input.strip():
                st.warning("Please enter clinical text.")
                return

            text_emb = get_bioclinical_embedding(text_input, tokenizer, bert_model)
            pred_class, probs = predict_hybrid(image, text_emb, hybrid_model, label_encoder)
            st.success(f"**Prediction:** {pred_class}")

            # Confidence bar chart
            class_names = label_encoder.classes_
            conf_df = pd.DataFrame({
                "Class": class_names,
                "Confidence": probs
            })

            st.markdown("### 🔍 Model Confidence")
            bar_chart = (
                alt.Chart(conf_df)
                .mark_bar(size=40)
                .encode(
                    x=alt.X("Class:N", sort="-y"),
                    y=alt.Y("Confidence:Q", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("Class:N", legend=None),
                    tooltip=["Class", "Confidence"]
                )
                .properties(width=400, height=250)
            )
            st.altair_chart(bar_chart, use_container_width=True)

            # st.json({cls: f"{p:.1%}" for cls,p in zip(label_encoder.classes_, probs)})

            # GradCAM on hybrid image branch
            target_layer = get_last_conv_layer(hybrid_model.image_model)
            gradcam = GradCAM(hybrid_model.image_model.backbone, target_layer)
            img_tensor = preprocess_image(image)
            cam = gradcam.generate_cam(img_tensor)
            overlay = visualize_gradcam(image, cam)
            st.image(overlay, caption="Grad-CAM (Hybrid Model)", width=300)

if __name__ == "__main__":
    main()
