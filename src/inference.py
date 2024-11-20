import os
import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
import albumentations as A
from matplotlib.patches import Patch
import json

from data.dataset import get_transform
from models.models import get_model

def load_model(model_path, device, model_type='enhanced_unet'):
    model = get_model(model_type, in_channels=3, out_channels=24).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def get_device(device_name):
    if device_name == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is not available")
        return torch.device('cuda')
    elif device_name == 'mps':
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS device requested but MPS is not available")
        return torch.device('mps')
    elif device_name == 'cpu':
        return torch.device('cpu')
    else:
        raise ValueError(f"Unknown device: {device_name}. Choose from: cuda, mps, cpu")

def preprocess_image(image_path, transform):
    # Load and transform image
    image = Image.open(image_path)
    original_size = image.size
    image = np.array(image)
    
    # Get the target size from the transform's Resize parameters
    target_size = transform.transforms[0].height  # Since width = height in our case
    
    # Store original image resized to match model input size
    resized_original = cv2.resize(image, (target_size, target_size))
    
    # Apply transformations
    transformed = transform(image=image)
    image_tensor = transformed['image']
    
    return image_tensor, original_size, resized_original

def predict_mask(model, image_tensor, device):
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        prediction = model(image_tensor)
        
        # Get class probabilities and predictions
        probabilities = prediction.cpu()
        class_predictions = torch.argmax(probabilities, dim=1).squeeze(0)
        
        return class_predictions.numpy(), probabilities.squeeze(0).numpy()

def visualize_results(original_image, class_predictions, probabilities, json_path=None, save_path=None):
    # Define colors (including background as black)
    colors = np.array([
        [0, 0, 0],       # Background (black)
        [255, 0, 0],     # Red
        [0, 255, 0],     # Green
        [0, 0, 255],     # Blue
        [255, 255, 0],   # Yellow
        [255, 0, 255],   # Magenta
        [0, 255, 255],   # Cyan
        [128, 0, 0],     # Maroon
        [0, 128, 0],     # Dark Green
        [0, 0, 128],     # Navy
        [128, 128, 0],   # Olive
        [128, 0, 128],   # Purple
        [0, 128, 128],   # Teal
        [192, 192, 192], # Silver
        [128, 128, 128], # Gray
        [255, 165, 0],   # Orange
        [255, 192, 203], # Pink
        [165, 42, 42],   # Brown
        [240, 230, 140], # Khaki
        [219, 112, 147], # Pale Violet Red
        [176, 224, 230], # Powder Blue
        [255, 218, 185], # Peach Puff
        [152, 251, 152], # Pale Green
        [147, 112, 219]  # Purple
    ], dtype=np.uint8)
    
    # Create colored mask for predictions
    colored_mask = np.zeros((*class_predictions.shape, 3), dtype=np.uint8)
    for i in range(24):
        colored_mask[class_predictions == i] = colors[i]
    
    # Create prediction overlay
    pred_overlay = cv2.addWeighted(original_image, 0.7, colored_mask, 0.3, 0)
    
    if save_path:
        if json_path and os.path.exists(json_path):
            try:
                # Load and create ground truth visualization
                with open(json_path, 'r') as f:
                    annotation = json.load(f)
                
                # Get original and target dimensions
                orig_h, orig_w = annotation['imageHeight'], annotation['imageWidth']
                target_h, target_w = original_image.shape[:2]
                
                # Create ground truth mask
                gt_mask = np.zeros(class_predictions.shape, dtype=np.uint8)
                colored_gt_mask = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
                
                # Draw polygons for each shape
                for shape in annotation['shapes']:
                    try:
                        # Scale points to match the resized image
                        points = np.array(shape['points'], dtype=np.float32)
                        points[:, 0] = points[:, 0] * (target_w / orig_w)
                        points[:, 1] = points[:, 1] * (target_h / orig_h)
                        points = points.astype(np.int32)
                        
                        label = int(shape['label'])
                        if label >= len(colors):
                            print(f"Warning: Label {label} is out of range. Using label % {len(colors)}")
                            label = label % len(colors)
                        
                        # Convert color to tuple for OpenCV
                        color = tuple(map(int, colors[label].tolist()))
                        
                        # Fill both masks
                        cv2.fillPoly(gt_mask, [points], label)
                        cv2.fillPoly(colored_gt_mask, [points], color)
                        
                    except ValueError as e:
                        print(f"Error processing label '{shape['label']}': {e}")
                        continue
                    except Exception as e:
                        print(f"Error processing shape: {e}")
                        continue
                
                # Create ground truth overlay
                gt_overlay = cv2.addWeighted(original_image, 0.7, colored_gt_mask, 0.3, 0)
                
                # Create figure with 5 subplots
                fig, axes = plt.subplots(1, 5, figsize=(25, 5))
                
                # Plot original image
                axes[0].imshow(original_image)
                axes[0].set_title('Original Image')
                axes[0].axis('off')
                
                # Plot ground truth mask
                axes[1].imshow(colored_gt_mask)
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')
                
                # Plot ground truth overlay
                axes[2].imshow(gt_overlay)
                axes[2].set_title('Ground Truth Overlay')
                axes[2].axis('off')
                
                # Plot prediction mask
                axes[3].imshow(colored_mask)
                axes[3].set_title('Prediction')
                axes[3].axis('off')
                
                # Plot prediction overlay
                axes[4].imshow(pred_overlay)
                axes[4].set_title('Prediction Overlay')
                axes[4].axis('off')
                
            except Exception as e:
                print(f"Error processing ground truth from {json_path}: {e}")
                # Fallback to 3-subplot visualization
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                axes[0].imshow(original_image)
                axes[0].set_title('Original Image')
                axes[0].axis('off')
                
                axes[1].imshow(colored_mask)
                axes[1].set_title('Predicted Segmentation')
                axes[1].axis('off')
                
                axes[2].imshow(pred_overlay)
                axes[2].set_title('Overlay')
                axes[2].axis('off')
        else:
            # Create figure with 3 subplots if no ground truth available
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(original_image)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            axes[1].imshow(colored_mask)
            axes[1].set_title('Predicted Segmentation')
            axes[1].axis('off')
            
            axes[2].imshow(pred_overlay)
            axes[2].set_title('Overlay')
            axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    
    return colored_mask, pred_overlay

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Semantic Segmentation Inference')
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, default='enhanced_unet', help='Type of model to use')
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'mps', 'cpu'], 
                      help='Device to run inference on (cuda/mps/cpu)')
    parser.add_argument('--output_path', type=str, help='Path to save visualization')
    parser.add_argument('--json_path', type=str, help='Path to class mapping JSON file')
    
    args = parser.parse_args()
    
    try:
        device = get_device(args.device)
    except RuntimeError as e:
        print(f"Error: {str(e)}")
        print("Falling back to CPU device")
        device = torch.device('cpu')
    
    # Load model
    model = load_model(args.model_path, device, args.model_type)
    
    # Get transforms
    transform = get_transform(train=False)
    
    # Preprocess image
    image_tensor, original_size, resized_original = preprocess_image(args.image_path, transform)
    
    # Get predictions
    class_predictions, probabilities = predict_mask(model, image_tensor, device)
    
    # Visualize and save results
    visualize_results(resized_original, class_predictions, probabilities, 
                     args.json_path, args.output_path)

if __name__ == '__main__':
    main()
