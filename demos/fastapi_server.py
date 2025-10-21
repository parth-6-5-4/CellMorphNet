"""
FastAPI REST API server for CellMorphNet.
Provides HTTP endpoints for blood cell classification.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import numpy as np
import logging
from pathlib import Path
import os
import sys
# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.backbones import get_model
from src.infer import GradCAM, get_target_layer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="CellMorphNet API",
    description="Blood cell classification API with explainable AI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
model = None
config = None
class_names = None
device = None


# Response models
class PredictionResponse(BaseModel):
    """Prediction response model."""
    predicted_class: str
    confidence: float
    probabilities: dict
    success: bool = True


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str
    num_classes: Optional[int] = None


def load_model_from_checkpoint(checkpoint_path: str):
    """Load model from checkpoint file."""
    global model, config, class_names, device
    
    # Determine device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    logger.info(f"Using device: {device}")
    
    # Load checkpoint with weights_only=False for PyTorch 2.6+ compatibility
    # This is safe as we trust our own trained checkpoints
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    class_names = checkpoint['class_names']
    
    # Create and load model
    model = get_model(
        backbone=config['backbone'],
        num_classes=config['num_classes'],
        pretrained=False
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully from {checkpoint_path}")
    logger.info(f"Backbone: {config['backbone']}")
    logger.info(f"Classes: {class_names}")


def preprocess_image(image: Image.Image, img_size: int = 224) -> torch.Tensor:
    """Preprocess image for inference."""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    # Try to find the best checkpoint
    checkpoint_paths = [
        "models/bloodmnist_full_exp/checkpoints/best.pth",
        "models/bloodmnist_exp/checkpoints/best.pth",
        "models/experiment/checkpoints/best.pth"
    ]
    
    checkpoint_path = None
    for path in checkpoint_paths:
        if os.path.exists(path):
            checkpoint_path = path
            break
    
    if checkpoint_path is None:
        logger.error("No checkpoint found! Please train a model first.")
        return
    
    logger.info(f"Loading model from {checkpoint_path}...")
    # Check if checkpoint exists
    if Path(checkpoint_path).exists():
        try:
            load_model_from_checkpoint(checkpoint_path)
            logger.info("Model loaded successfully on startup")
        except Exception as e:
            logger.error(f"Failed to load model on startup: {e}")
    else:
        logger.warning(f"Checkpoint not found at {checkpoint_path}")
        logger.warning("API will start without a loaded model")


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "CellMorphNet API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "classes": "/classes"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "no_model",
        model_loaded=model is not None,
        device=str(device) if device is not None else "unknown",
        num_classes=len(class_names) if class_names is not None else None
    )


@app.get("/classes", response_model=dict)
async def get_classes():
    """Get available class names."""
    if class_names is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "classes": class_names,
        "num_classes": len(class_names)
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    return_all_probs: bool = True
):
    """
    Predict cell type from uploaded image.
    
    Args:
        file: Image file (JPEG, PNG, etc.)
        return_all_probs: Whether to return all class probabilities
    
    Returns:
        Prediction results with confidence scores
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        image_tensor = preprocess_image(image, config['img_size'])
        
        # Predict
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
        
        # Get prediction
        confidence, pred_idx = torch.max(probabilities, 0)
        predicted_class = class_names[pred_idx.item()]
        
        # Prepare response
        response = PredictionResponse(
            predicted_class=predicted_class,
            confidence=float(confidence.item()),
            probabilities={
                class_names[i]: float(probabilities[i].item())
                for i in range(len(class_names))
            } if return_all_probs else {predicted_class: float(confidence.item())}
        )
        
        logger.info(f"Prediction: {predicted_class} ({confidence.item():.2%})")
        
        return response
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict cell types for multiple images.
    
    Args:
        files: List of image files
    
    Returns:
        List of prediction results
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    
    for file in files:
        try:
            # Read and preprocess image
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            
            image_tensor = preprocess_image(image, config['img_size'])
            
            # Predict
            with torch.no_grad():
                image_tensor = image_tensor.to(device)
                outputs = model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)[0]
            
            # Get prediction
            confidence, pred_idx = torch.max(probabilities, 0)
            predicted_class = class_names[pred_idx.item()]
            
            results.append({
                "filename": file.filename,
                "predicted_class": predicted_class,
                "confidence": float(confidence.item()),
                "success": True
            })
        
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            results.append({
                "filename": file.filename,
                "error": str(e),
                "success": False
            })
    
    return {"results": results, "total": len(files), "successful": sum(1 for r in results if r["success"])}


@app.post("/predict/gradcam")
async def predict_with_gradcam(file: UploadFile = File(...)):
    """
    Predict and generate Grad-CAM visualization.
    
    Args:
        file: Image file
    
    Returns:
        Prediction with Grad-CAM heatmap (as image)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        import cv2
        import matplotlib.pyplot as plt
        
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        original_array = np.array(image)
        
        image_tensor = preprocess_image(image, config['img_size'])
        
        # Predict
        with torch.no_grad():
            image_tensor_device = image_tensor.to(device)
            outputs = model(image_tensor_device)
            probabilities = F.softmax(outputs, dim=1)[0]
        
        confidence, pred_idx = torch.max(probabilities, 0)
        predicted_class = class_names[pred_idx.item()]
        
        # Generate Grad-CAM
        target_layer = get_target_layer(model, config['backbone'])
        gradcam = GradCAM(model, target_layer)
        heatmap = gradcam.generate(image_tensor_device, pred_idx.item())
        
        # Overlay heatmap
        heatmap_resized = cv2.resize(heatmap, (original_array.shape[1], original_array.shape[0]))
        heatmap_colored = (heatmap_resized * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        overlayed = (0.5 * heatmap_colored + 0.5 * original_array).astype(np.uint8)
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(original_array)
        axes[0].set_title(f"Prediction: {predicted_class}\nConfidence: {confidence.item():.2%}")
        axes[0].axis('off')
        
        axes[1].imshow(overlayed)
        axes[1].set_title("Grad-CAM")
        axes[1].axis('off')
        
        plt.tight_layout()
        
        # Save to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return StreamingResponse(buf, media_type="image/png")
    
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="OpenCV (cv2) is required for Grad-CAM. Install with: pip install opencv-python"
        )
    except Exception as e:
        logger.error(f"Grad-CAM error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/load_model")
async def load_model_endpoint(checkpoint_path: str):
    """
    Load or reload model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
    """
    try:
        load_model_from_checkpoint(checkpoint_path)
        return {"message": "Model loaded successfully", "checkpoint": checkpoint_path}
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    import uvicorn
    
    # Run server
    uvicorn.run(
        "fastapi_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
