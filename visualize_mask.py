import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os

def visualize_json_mask(json_path):
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
    ])

    # Load JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Print shape information for debugging
    print("Shapes in JSON:")
    for shape in data['shapes']:
        print(f"Label: {shape['label']}")
    
    # Get image dimensions from JSON
    img_height = data['imageHeight']
    img_width = data['imageWidth']
    
    # Create empty mask
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    colored_mask = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    
    # Draw polygons for each shape
    for shape in data['shapes']:
        points = np.array(shape['points'], dtype=np.int32)
        try:
            label = int(shape['label'])
            if label >= len(colors):
                print(f"Warning: Label {label} is out of range. Using label % {len(colors)}")
                label = label % len(colors)
            cv2.fillPoly(mask, [points], label)
            # Convert color array to tuple
            color = tuple(map(int, colors[label]))
            cv2.fillPoly(colored_mask, [points], color)
        except ValueError as e:
            print(f"Error processing label '{shape['label']}': {e}")
            continue
    
    # Load original image
    img_path = json_path.replace('.json', '.png')
    if os.path.exists(img_path):
        original_image = cv2.imread(img_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(original_image, 0.7, colored_mask, 0.3, 0)
    else:
        original_image = None
        overlay = None
    
    # Create visualization
    if original_image is not None:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot original image
        ax1.imshow(original_image)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Plot colored mask
        ax2.imshow(colored_mask)
        ax2.set_title('Segmentation Mask')
        ax2.axis('off')
        
        # Plot overlay
        ax3.imshow(overlay)
        ax3.set_title('Overlay')
        ax3.axis('off')
    else:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(colored_mask)
        ax.set_title('Segmentation Mask')
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'mask_visualization.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Visualization saved to {output_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, required=True,
                      help='Path to the JSON file with polygon annotations')
    args = parser.parse_args()
    
    visualize_json_mask(args.json_path)
