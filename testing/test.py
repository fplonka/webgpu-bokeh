#!/usr/bin/env python3
import numpy as np
from PIL import Image
import os
from typing import Tuple
from numba import njit, prange, parallel

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

@njit
def srgb_to_linear(srgb_value: float) -> float:
    """Convert from sRGB to linear RGB color space."""
    if srgb_value <= 0.04045:
        return srgb_value / 12.92
    else:
        return ((srgb_value + 0.055) / 1.055) ** 2.4

@njit
def linear_to_srgb(linear_value: float) -> float:
    """Convert from linear RGB to sRGB color space."""
    if linear_value <= 0.0031308:
        return linear_value * 12.92
    else:
        return 1.055 * (linear_value ** (1.0/2.4)) - 0.055

@njit
def unpack_rgba_uint32(rgba_uint32: np.ndarray) -> np.ndarray:
    """Unpack uint32 RGBA values into a float32 RGBA array with values 0-1 in linear RGB space."""
    height, width = rgba_uint32.shape
    result = np.zeros((height, width, 4), dtype=np.float32)
    
    for y in range(height):
        for x in range(width):
            pixel = rgba_uint32[y, x]
            
            # Convert sRGB to linear RGB for RGB channels
            r_srgb = ((pixel >> 24) & 0xFF) / 255.0
            g_srgb = ((pixel >> 16) & 0xFF) / 255.0
            b_srgb = ((pixel >> 8) & 0xFF) / 255.0
            a = (pixel & 0xFF) / 255.0  # Alpha stays as is
            
            # Convert to linear RGB
            result[y, x, 0] = srgb_to_linear(r_srgb)
            result[y, x, 1] = srgb_to_linear(g_srgb)
            result[y, x, 2] = srgb_to_linear(b_srgb)
            result[y, x, 3] = a
    return result

@njit
def pack_rgba_float32(rgba_float32: np.ndarray) -> np.ndarray:
    """Pack float32 RGBA values (0-1) in linear RGB space into uint32 RGBA values in sRGB space."""
    height, width = rgba_float32.shape[:2]
    result = np.zeros((height, width), dtype=np.uint32)
    
    for y in range(height):
        for x in range(width):
            # Convert from linear RGB to sRGB for RGB channels
            r_linear = rgba_float32[y, x, 0]
            g_linear = rgba_float32[y, x, 1]
            b_linear = rgba_float32[y, x, 2]
            a = rgba_float32[y, x, 3]  # Alpha stays as is
            
            # Convert to sRGB
            r_srgb = linear_to_srgb(r_linear)
            g_srgb = linear_to_srgb(g_linear)
            b_srgb = linear_to_srgb(b_linear)
            
            # Pack as 8-bit values
            r = int(r_srgb * 255)
            g = int(g_srgb * 255)
            b = int(b_srgb * 255)
            a = int(a * 255)
            
            # Clamp to valid range
            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))
            a = max(0, min(255, a))
            
            result[y, x] = (r << 24) | (g << 16) | (b << 8) | a
    return result

@njit
def generate_sample_offsets(num_samples: int, angle_deg: float) -> np.ndarray:
    """Generate sample offsets along a line at the specified angle.
    Returns array of (x, y) offsets normalized to unit length."""
    # Convert angle to radians
    angle_rad = angle_deg * np.pi / 180.0
    
    # Create evenly spaced offsets from -0.5 to 0.5
    offsets = np.zeros((num_samples, 2), dtype=np.float32)
    step = 1.0 / (num_samples - 1)
    
    for i in range(num_samples):
        t = -0.5 + i * step
        offsets[i, 0] = t * np.cos(angle_rad)  # x
        offsets[i, 1] = t * np.sin(angle_rad)  # y
    
    return offsets

@njit(parallel=True)
def apply_directional_blur(rgba_float: np.ndarray, coc_array: np.ndarray, 
                         offsets: np.ndarray) -> np.ndarray:
    """Apply blur along the direction specified by the offsets."""
    height, width = rgba_float.shape[:2]
    result = np.zeros_like(rgba_float)
    
    # Process each row in parallel
    for y in prange(height):
        # Pre-allocate arrays for this thread
        color_sum = np.zeros(4, dtype=np.float32)
        
        for x in range(width):
            coc = coc_array[y, x]
            if coc == 0:
                result[y, x] = rgba_float[y, x]
                continue
            
            # Reset accumulators for this pixel
            color_sum.fill(0)
            weight_sum = 0.0
            
            # Sample along the scaled offsets
            for i in range(len(offsets)):
                # Calculate sample position
                sample_x = int(x + offsets[i, 0] * coc)
                sample_y = int(y + offsets[i, 1] * coc)

                if sample_x < 0 or sample_x >= width or sample_y < 0 or sample_y >= height:
                    continue
                
                sample_coc = coc_array[sample_y, sample_x]
                
                dist = np.sqrt((sample_x - x)**2 + (sample_y - y)**2)
                if dist > sample_coc:
                    continue
                
                # Add to accumulator
                color_sum += rgba_float[sample_y, sample_x]
                weight_sum += 1.0
            
            # Calculate final color
            if weight_sum > 0:
                result[y, x] = color_sum / weight_sum
            else:
                result[y, x] = rgba_float[y, x]
    
    return result

@njit(parallel=True)
def combine_less_bright(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Create a new image by taking the less bright pixel from each source image."""
    height, width = img1.shape[:2]
    result = np.zeros_like(img1)
    
    # Process each row in parallel
    for y in prange(height):
        for x in range(width):
            # Calculate brightness (using luminance formula)
            brightness1 = 0.2126 * img1[y, x, 0] + 0.7152 * img1[y, x, 1] + 0.0722 * img1[y, x, 2]
            brightness2 = 0.2126 * img2[y, x, 0] + 0.7152 * img2[y, x, 1] + 0.0722 * img2[y, x, 2]
            
            # Choose the pixel with lower brightness
            if brightness1 <= brightness2:
                result[y, x] = img1[y, x]
            else:
                result[y, x] = img2[y, x]
    
    return result

def save_rgba_uint32_as_image(rgba_uint32: np.ndarray, output_path: str) -> None:
    """Save an RGBA uint32 array as a PNG image."""
    height, width = rgba_uint32.shape
    rgba_uint8 = np.zeros((height, width, 4), dtype=np.uint8)
    
    # Directly extract bytes from the uint32 values
    for y in range(height):
        for x in range(width):
            pixel = rgba_uint32[y, x]
            rgba_uint8[y, x, 0] = (pixel >> 24) & 0xFF  # R
            rgba_uint8[y, x, 1] = (pixel >> 16) & 0xFF  # G
            rgba_uint8[y, x, 2] = (pixel >> 8) & 0xFF   # B
            rgba_uint8[y, x, 3] = pixel & 0xFF          # A
    
    Image.fromarray(rgba_uint8, 'RGBA').save(output_path)

if __name__ == "__main__":
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define paths to the images
    image_path = os.path.join(script_dir, "image.jpg")
    depth_map_path = os.path.join(script_dir, "depth_map.png")
    
    # Define constants
    focus_depth = 0.4  # Focus on objects at depth 0.4 (normalized scale)
    max_coc = 150.0    # Maximum circle of confusion radius in pixels
    num_samples = int(max_coc/2)   # Number of samples per pass
    
    # Read the image into RGBA uint32 array
    rgba_array, img_width, img_height = read_image_to_rgba_uint32(image_path)
    
    # Read the depth map into float32 array
    depth_array, depth_width, depth_height = read_depth_map_to_float32(depth_map_path)
    
    # Compute Circle of Confusion (CoC) radius for each pixel
    # Only blur objects closer than the focus plane
    coc_array = np.where(depth_array > focus_depth, 0, np.abs(depth_array - focus_depth) * max_coc)
    
    # Generate sample offsets for all passes
    offsets_0deg = generate_sample_offsets(num_samples, 0)     # Horizontal (0°)
    offsets_45deg = generate_sample_offsets(num_samples, 45)   # 45° diagonal
    offsets_60deg = generate_sample_offsets(num_samples, 60)   # 60° diagonal
    offsets_90deg = generate_sample_offsets(num_samples, 90)   # 90° diagonal
    offsets_120deg = generate_sample_offsets(num_samples, 120)   # 90° diagonal
    offsets_135deg = generate_sample_offsets(num_samples, 135) # 135° diagonal (45° + 90°)
    
    # Convert to float32 for processing
    rgba_float = unpack_rgba_uint32(rgba_array)
    
    def do_hexagon_bokeh(rgba_float: np.ndarray, coc_array: np.ndarray, max_coc: float):
        print("doing horizontal")
        horizontal_blur = apply_directional_blur(rgba_float, coc_array, offsets_0deg, max_coc)
        print("doing 60°")
        bokeh_60_float = apply_directional_blur(horizontal_blur, coc_array, offsets_60deg, max_coc)
        print("doing 120")
        bokeh_120_float = apply_directional_blur(horizontal_blur, coc_array, offsets_120deg, max_coc)
        
        # Combine the two blurs by taking the less bright pixel
        combined_float = combine_less_bright(bokeh_60_float, bokeh_120_float)

        combined = pack_rgba_float32(combined_float)
        save_rgba_uint32_as_image(combined, os.path.join(script_dir, "bokeh_combined.png"))
        print("Images saved.")
        
    def do_octagon_bokeh(rgba_float: np.ndarray, coc_array: np.ndarray):
        print("doing horizontal")
        horizontal_blur = apply_directional_blur(rgba_float, coc_array, offsets_0deg)
        print("doing 90")
        bokeh_90_float = apply_directional_blur(horizontal_blur, coc_array, offsets_90deg)

        print("doing 45")
        bokeh_45_float = apply_directional_blur(rgba_float, coc_array, offsets_45deg)
        print("doing 135")
        bokeh_135_float = apply_directional_blur(bokeh_45_float, coc_array, offsets_135deg)

        # Combine the two blurs by taking the less bright pixel
        combined_float = combine_less_bright(bokeh_90_float, bokeh_135_float)

        combined = pack_rgba_float32(combined_float)
        save_rgba_uint32_as_image(combined, os.path.join(script_dir, "bokeh_combined.png"))
        print("Images saved.")
        
    # do_hexagon_bokeh(rgba_float, coc_array)
    do_octagon_bokeh(rgba_float, coc_array)
    