"""
Streamlit demo app for CellMorphNet.
Interactive web interface for blood cell classification.
"""

import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.backbones import get_model
from src.infer import GradCAM, get_target_layer


# Configure page
st.set_page_config(
    page_title="CellMorphNet",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_model(checkpoint_path):
    """Load model from checkpoint (cached)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint with weights_only=False for PyTorch 2.6+ compatibility
    # This is safe as we trust our own trained checkpoints
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    class_names = checkpoint['class_names']
    
    model = get_model(
        backbone=config['backbone'],
        num_classes=config['num_classes'],
        pretrained=False
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config, class_names, device


def preprocess_image(image, img_size=224, use_downscale_fix=False):
    """
    Preprocess image for model.
    
    Args:
        image: PIL Image
        img_size: Target size (default 224)
        use_downscale_fix: If True, downscale to 28x28 first to match training data.
                          Use this for large screenshots or high-resolution images.
    """
    # Apply downscale fix for large images (matching training data preprocessing)
    if use_downscale_fix:
        # Training data was 28x28 upscaled to 224x224
        # So we downscale large images to 28x28 first, then upscale
        image = image.resize((28, 28), Image.LANCZOS)
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


def predict(model, image_tensor, device):
    """Make prediction."""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
    return probabilities[0].cpu()


def generate_gradcam(model, image_tensor, device, backbone, target_class=None):
    """Generate Grad-CAM visualization."""
    try:
        target_layer = get_target_layer(model, backbone)
        gradcam = GradCAM(model, target_layer)
        
        heatmap = gradcam.generate(image_tensor.to(device), target_class)
        return heatmap
    except Exception as e:
        st.error(f"Grad-CAM generation failed: {e}")
        return None


def overlay_heatmap(heatmap, original_image, alpha=0.5):
    """Overlay heatmap on image."""
    import cv2
    
    # Convert PIL to numpy
    original = np.array(original_image)
    
    # Resize heatmap
    heatmap_resized = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    
    # Colorize
    heatmap_colored = (heatmap_resized * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlayed = (alpha * heatmap_colored + (1 - alpha) * original).astype(np.uint8)
    
    return overlayed


def main():
    # Title and description
    st.title("CellMorphNet: Blood Cell Classifier")
    st.markdown("""
    **Real-time blood cell classification** using lightweight CNN with morphology attention.
    
    Upload a microscopy image to classify cell types with explainable AI visualizations.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Model selection
        st.subheader("Model")
        
        # Look for available checkpoints
        checkpoint_dirs = [
            Path("models/bloodmnist_full_exp/checkpoints"),
            Path("models/bloodmnist_exp/checkpoints"),
            Path("models/experiment/checkpoints")
        ]
        
        checkpoint_path = None
        checkpoint_files = []
        
        for checkpoint_dir in checkpoint_dirs:
            if checkpoint_dir.exists():
                files = list(checkpoint_dir.glob("*.pth"))
                if files:
                    checkpoint_files.extend([(checkpoint_dir, f) for f in files])
        
        if checkpoint_files:
            checkpoint_options = [f"{dir.parent.name}/{f.name}" for dir, f in checkpoint_files]
            selected = st.selectbox("Select checkpoint", checkpoint_options, index=0)
            # Find the selected checkpoint
            for dir, f in checkpoint_files:
                if f"{dir.parent.name}/{f.name}" == selected:
                    checkpoint_path = dir / f.name
                    break
        else:
            st.warning("No checkpoints found in models/")
            st.info("Please train a model first using `python src/train.py`")
            checkpoint_path = st.text_input("Enter checkpoint path:", "models/bloodmnist_full_exp/checkpoints/best.pth")
            checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        
        st.markdown("---")
        
        # Preprocessing options
        st.subheader("Preprocessing")
        use_downscale_fix = st.checkbox(
            "Enable downscale fix",
            value=False,
            help="Enable this for large screenshots or high-res images. "
                 "Downscales to 28x28 first to match training data format. "
                 "Improves accuracy for screenshots but may reduce quality for microscopy images."
        )
        
        if use_downscale_fix:
            st.info("Downscale fix enabled: Large images will be downscaled to 28x28 first")
        
        st.markdown("---")
        
        # Visualization options
        st.subheader("Visualization")
        show_gradcam = st.checkbox("Show Grad-CAM", value=True)
        show_probabilities = st.checkbox("Show all probabilities", value=True)
        confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)
        
        st.markdown("---")
        
        # Info
        st.subheader("ℹ️ About")
        st.markdown("""
        **CellMorphNet** is a lightweight CNN for blood cell classification.
        
        Features:
        -  Real-time inference
        -  High accuracy
        -  Explainable AI (Grad-CAM)
    
        """)
    
    # Main content
    if checkpoint_path and checkpoint_path.exists():
        # Load model
        with st.spinner("Loading model..."):
            try:
                model, config, class_names, device = load_model(str(checkpoint_path))
                st.success(f"Model loaded successfully!")
                
                # Display model info
                with st.expander("Model Information"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Backbone", config['backbone'])
                    with col2:
                        st.metric("Classes", len(class_names))
                    with col3:
                        st.metric("Image Size", f"{config['img_size']}x{config['img_size']}")
                    
                    st.write("**Classes:**", ", ".join(class_names))
            
            except Exception as e:
                st.error(f"Failed to load model: {e}")
                return
        
        # File uploader
        st.markdown("---")
        st.header("Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose a blood cell microscopy image",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help="Upload a microscopy image of blood cells"
        )
        
        # Example images
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Use Example"):
                # Try to find example images
                example_dir = Path("data/raw/bloodmnist/test")
                if example_dir.exists():
                    # Get random example
                    import random
                    class_dirs = [d for d in example_dir.iterdir() if d.is_dir()]
                    if class_dirs:
                        random_class = random.choice(class_dirs)
                        images = list(random_class.glob("*.png"))
                        if images:
                            uploaded_file = str(random.choice(images))
                            st.info(f"Using example from: {random_class.name}")
        
        if uploaded_file is not None:
            try:
                # Load image
                if isinstance(uploaded_file, str):
                    image = Image.open(uploaded_file).convert('RGB')
                else:
                    image = Image.open(uploaded_file).convert('RGB')
                
                # Display original image
                st.markdown("---")
                st.header("Input Image")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Predict button
                if st.button("Classify", type="primary", use_container_width=True):
                    with st.spinner("Analyzing image..."):
                        # Preprocess
                        image_tensor = preprocess_image(image, config['img_size'], use_downscale_fix=use_downscale_fix)
                        
                        # Predict
                        probabilities = predict(model, image_tensor, device)
                        
                        # Get top prediction
                        confidence, pred_idx = torch.max(probabilities, 0)
                        predicted_class = class_names[pred_idx.item()]
                        confidence_val = confidence.item()
                        
                        # Display results
                        st.markdown("---")
                        st.header("Prediction Results")
                        
                        # Main prediction
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            if confidence_val >= confidence_threshold:
                                st.success(f"**Predicted Class:** {predicted_class}")
                            else:
                                st.warning(f"**Predicted Class:** {predicted_class} (Low confidence)")
                            
                            st.metric("Confidence", f"{confidence_val:.2%}")
                        
                        # Probability bar chart
                        if show_probabilities:
                            st.markdown("### Class Probabilities")
                            
                            # Create bar chart
                            fig, ax = plt.subplots(figsize=(10, 6))
                            probs = probabilities.numpy()
                            colors = ['green' if i == pred_idx else 'steelblue' for i in range(len(probs))]
                            ax.barh(class_names, probs, color=colors)
                            ax.set_xlabel('Probability')
                            ax.set_xlim(0, 1)
                            ax.axvline(x=confidence_threshold, color='red', linestyle='--', label=f'Threshold ({confidence_threshold})')
                            ax.legend()
                            st.pyplot(fig)
                            plt.close()
                        
                        # Grad-CAM visualization
                        if show_gradcam:
                            st.markdown("### Grad-CAM Visualization")
                            st.info("Highlighting regions that influenced the prediction")
                            
                            with st.spinner("Generating Grad-CAM..."):
                                heatmap = generate_gradcam(
                                    model, image_tensor, device,
                                    config['backbone'], pred_idx.item()
                                )
                                
                                if heatmap is not None:
                                    try:
                                        overlayed = overlay_heatmap(heatmap, image, alpha=0.5)
                                        
                                        # Display side by side
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.image(image, caption="Original", use_column_width=True)
                                        with col2:
                                            st.image(overlayed, caption="Grad-CAM Overlay", use_column_width=True)
                                    except ImportError:
                                        st.warning("OpenCV (cv2) is required for Grad-CAM visualization. Install with: pip install opencv-python")
            
            except Exception as e:
                st.error(f"Error processing image: {e}")
                st.exception(e)
    
    else:
        st.warning("Please configure a valid model checkpoint in the sidebar")
        st.info("""
        To get started:
        1. Train a model using `python src/train.py --data_dir data/raw/bloodmnist --num_classes 8 --epochs 20`
        2. The trained model will be saved to `models/experiment/checkpoints/`
        3. Reload this page to see the checkpoint
        """)


if __name__ == '__main__':
    main()
