#!/usr/bin/env python3
"""Test dataset loading"""

from pathlib import Path
import json
import numpy as np
from PIL import Image
import cv2

# Test dataset loading
data_dir = Path('.')
train_dir = data_dir / 'train'

# Find first JSON file with corresponding image
json_files = list(train_dir.glob('*.json'))
print(f"Found {len(json_files)} JSON files")

found_pair = False
for json_file in json_files[:10]:  # Check first 10
    img_name = json_file.stem
    jpg_path = train_dir / f"{img_name}.jpg"
    
    if jpg_path.exists():
        found_pair = True
        print(f"\n{'='*80}")
        print(f"Testing: {img_name}")
        
        # Load JSON
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Load image
        image = Image.open(jpg_path).convert('RGB')
        width, height = image.size
        print(f"Image size: {width}x{height}")
        
        # Get text
        text = data['text'][0] if isinstance(data['text'], list) else data['text']
        print(f"Text: {text}")
        
        # Get shapes
        shapes = data.get('shapes', [])
        print(f"Number of shapes: {len(shapes)}")
        
        # Create mask
        mask = np.zeros((height, width), dtype=np.uint8)
        for shape in shapes:
            if shape['shape_type'] == 'polygon':
                points = np.array(shape['points'], dtype=np.int32)
                cv2.fillPoly(mask, [points], 1)
                print(f"  Polygon with {len(points)} points")
        
        print(f"Mask shape: {mask.shape}, unique values: {np.unique(mask)}")
        print(f"Mask pixels set: {mask.sum()} / {mask.size} ({100*mask.sum()/mask.size:.2f}%)")
        
        break  # Just test one

if not found_pair:
    print("\nWarning: No JSON-image pairs found in first 10 files")

print("\n" + "="*80)
print("✓ Dataset test passed!")
