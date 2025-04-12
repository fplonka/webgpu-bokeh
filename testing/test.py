#!/usr/bin/env python3
import numpy as np
from PIL import Image
import os
from typing import Tuple

def read_image_to_rgba_uint32(image_path):
    """
    Reads an image file into a numpy array with RGBA uint32 values.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        tuple: (numpy array of uint32 RGBA values, image width, image height)
    """
    # Open the image and convert to RGBA
    img = Image.open(image_path).convert('RGBA')
    width, height = img.size
    
    # Convert to numpy array (RGBA)
    img_array = np.array(img)
    
    # Pack RGBA channels into a single uint32 value
    # R << 24 | G << 16 | B << 8 | A
    rgba_uint32 = (
        (img_array[:, :, 0].astype(np.uint32) << 24) | 
        (img_array[:, :, 1].astype(np.uint32) << 16) | 
        (img_array[:, :, 2].astype(np.uint32) << 8) | 
        (img_array[:, :, 3].astype(np.uint32))
    )
    
    return rgba_uint32, width, height

def read_depth_map_to_float32(depth_map_path):
    """
    Reads a grayscale depth map into a numpy array with float32 values scaled from 0 to 1.
    
    Args:
        depth_map_path (str): Path to the depth map image file
        
    Returns:
        tuple: (numpy array of float32 values normalized to 0-1, image width, image height)
    """
    # Open the depth map image in grayscale mode
    depth_img = Image.open(depth_map_path).convert('L')
    width, height = depth_img.size
    
    # Convert to numpy array
    depth_array = np.array(depth_img, dtype=np.float32)
    
    # Normalize to 0-1 range (assuming the depth map is grayscale with values 0-255)
    depth_normalized = depth_array / 255.0
    
    return depth_normalized, width, height

def unpack_rgba_uint32(rgba_uint32: np.ndarray) -> np.ndarray:
    """Unpack uint32 RGBA values into a float32 RGBA array with values 0-1."""
    r = ((rgba_uint32 >> 24) & 0xFF).astype(np.float32) / 255.0
    g = ((rgba_uint32 >> 16) & 0xFF).astype(np.float32) / 255.0
    b = ((rgba_uint32 >> 8) & 0xFF).astype(np.float32) / 255.0
    a = (rgba_uint32 & 0xFF).astype(np.float32) / 255.0
    return np.stack([r, g, b, a], axis=-1)

def pack_rgba_float32(rgba_float32: np.ndarray) -> np.ndarray:
    """Pack float32 RGBA values (0-1) into uint32 RGBA values."""
    rgba_uint8 = (rgba_float32 * 255.0).astype(np.uint32)
    return (rgba_uint8[..., 0] << 24) | (rgba_uint8[..., 1] << 16) | \
           (rgba_uint8[..., 2] << 8) | rgba_uint8[..., 3]

def box_blur_horizontal(rgba_uint32: np.ndarray, coc_array: np.ndarray) -> np.ndarray:
    """Apply horizontal box blur based on CoC values."""
    height, width = rgba_uint32.shape
    rgba_float = unpack_rgba_uint32(rgba_uint32)
    result = np.zeros_like(rgba_float)
    
    for y in range(height):
        for x in range(width):
            radius = int(np.ceil(coc_array[y, x]))
            if radius == 0:
                result[y, x] = rgba_float[y, x]
                continue
                
            # Calculate valid x range for sampling
            x_start = max(0, x - radius)
            x_end = min(width, x + radius + 1)
            
            # Calculate average color within the box
            sample_colors = rgba_float[y, x_start:x_end]
            result[y, x] = np.mean(sample_colors, axis=0)
    
    return pack_rgba_float32(result)

def box_blur_vertical(rgba_uint32: np.ndarray, coc_array: np.ndarray) -> np.ndarray:
    """Apply vertical box blur based on CoC values."""
    height, width = rgba_uint32.shape
    rgba_float = unpack_rgba_uint32(rgba_uint32)
    result = np.zeros_like(rgba_float)
    
    for x in range(width):
        for y in range(height):
            radius = int(np.ceil(coc_array[y, x]))
            if radius == 0:
                result[y, x] = rgba_float[y, x]
                continue
                
            # Calculate valid y range for sampling
            y_start = max(0, y - radius)
            y_end = min(height, y + radius + 1)
            
            # Calculate average color within the box
            sample_colors = rgba_float[y_start:y_end, x]
            result[y, x] = np.mean(sample_colors, axis=0)
    
    return pack_rgba_float32(result)

def save_rgba_uint32_as_image(rgba_uint32: np.ndarray, output_path: str) -> None:
    """Save an RGBA uint32 array as a PNG image."""
    rgba_float = unpack_rgba_uint32(rgba_uint32)
    rgba_uint8 = (rgba_float * 255).astype(np.uint8)
    Image.fromarray(rgba_uint8, 'RGBA').save(output_path)

if __name__ == "__main__":
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define paths to the images
    image_path = os.path.join(script_dir, "image.jpg")
    depth_map_path = os.path.join(script_dir, "depth_map.png")
    
    # Define constants for CoC calculation
    focus_depth = 0.4  # Focus on objects at depth 0.4 (normalized scale)
    max_coc = 10.0     # Maximum circle of confusion radius in pixels
    
    # Read the image into RGBA uint32 array
    rgba_array, img_width, img_height = read_image_to_rgba_uint32(image_path)
    
    # Read the depth map into float32 array
    depth_array, depth_width, depth_height = read_depth_map_to_float32(depth_map_path)
    
    # Compute Circle of Confusion (CoC) radius for each pixel
    # Simple formula: |depth - focus_depth| * max_coc
    coc_array = np.abs(depth_array - focus_depth) * max_coc
    
    print("Applying horizontal blur...")
    horizontal_blur = box_blur_horizontal(rgba_array, coc_array)
    
    print("Applying vertical blur...")
    final_blur = box_blur_vertical(horizontal_blur, coc_array)
    
    # Save the result
    output_path = os.path.join(script_dir, "blurred_output.png")
    save_rgba_uint32_as_image(final_blur, output_path)
    print(f"\nBlurred image saved as 'blurred_output.png'")
