# main.py

import numpy as np
import math
from PIL import Image
# from gen_test_data import generate_scene, save_images, load_images

def apply_bokeh_3(color_image, depth_map, focus_depth, max_coc_radius, depth_of_field):
    import numpy as np
    import math
    import numba
    from numba import njit, prange

    print("GOT FOCUS DEPTH:", focus_depth)
    print("Color image shape:", color_image.shape)
    print("Depth map shape:", depth_map.shape)
    print("Color image dtype:", color_image.dtype)
    print("Depth map dtype:", depth_map.dtype)
    print("Depth map range:", np.min(depth_map), "-", np.max(depth_map))

    # --- Convert color image to linear space ---
    # 1) Scale from [0..255] to [0..1]
    # 2) Apply gamma exponent (2.2)
    color_image_float = color_image.astype(np.float32) / 255.0
    gamma_val = 2.2
    color_image_linear = color_image_float ** gamma_val

    @njit(fastmath=True)
    def calculate_coc_radius(pixel_depth, focal_depth, max_coc):
        return max_coc * max(abs(focal_depth - pixel_depth) - depth_of_field / 2, 0)

    @njit(fastmath=True, parallel=True)
    def compute_coc(d_map, width, focus_depth, max_coc):
        height = d_map.shape[0]
        coc_out = np.zeros((height, width), dtype=np.float32)
        for y in prange(height):
            for x in prange(width):
                coc_radius = calculate_coc_radius(d_map[y, x], focus_depth, max_coc)
                coc_out[y, x] = coc_radius * 0.0015 * width
        return coc_out

    height, width = depth_map.shape
    coc = compute_coc(depth_map, width, focus_depth, max_coc_radius)

    zero_count = np.count_nonzero(coc == 0)
    print("Number of zeros in coc:", zero_count)

    @njit(fastmath=True)
    def apply_bokeh_loop(img_linear, coc_map, d_map):
        h, w = coc_map.shape
        new_img = np.zeros_like(img_linear, dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)

        for y in range(h):
            if y % 100 == 0:
                print("y =", y, "out of", h)
            for x in range(w):
                if coc_map[y, x] == 0:
                    new_img[y, x, 0] += img_linear[y, x, 0]
                    new_img[y, x, 1] += img_linear[y, x, 1]
                    new_img[y, x, 2] += img_linear[y, x, 2]
                    weight_map[y, x] += 1
                    continue

                coc_radius = coc_map[y, x]
                depth_val = d_map[y, x]

                coc_radius_int = math.ceil(coc_radius) + 1
                area = coc_radius * coc_radius

                # initial_w = np.pow(area, -0.6)
                # initial_w = np.pow(area, -0.6)
                # initial_w = 1 / area
                # initial_w = 1 / area
                initial_w = np.pow(area, -0.6)

                for j in range(max(0, y - coc_radius_int), min(h, y + coc_radius_int + 1)):
                    for i in range(max(0, x - coc_radius_int), min(w, x + coc_radius_int + 1)):
                        # dist = max(abs(i - x), abs(j - y))
                        dist = np.sqrt((i - x)**2 + (j - y)**2)
                        overlap_weight = coc_radius - dist + 0.5
                        overlap_weight = max(0, overlap_weight)
                        overlap_weight = min(1, overlap_weight)
                        # overlap_weight = 1

                        leakage_weight = 1.0 
                        # if depth_val > d_map[j, i]:
                        #     leakage_weight = 1.0 - 1.0 * (depth_val - d_map[j, i])

                        if depth_val < focus_depth:
                            leakage_weight = coc_radius / max_coc_radius
                        
                        # this pixel is in front, so we do something weird:
                        # we add light from that pixel to this pixel
                        # if depth_val > d_map[j, i]:
                        #     scale = 0.8 * ((depth_val - d_map[j, i]))
                        #     new_img[y, x, 0] += img_linear[j, i, 0] * scale
                        #     new_img[y, x, 1] += img_linear[j, i, 1] * scale
                        #     new_img[y, x, 2] += img_linear[j, i, 2] * scale
                        #     weight_map[y, x] += scale
                        
                        # leakage_weight = 1
                        # if depth_val < d_map[j, i]:
                        #     leakage_weight = 1 - 1 * (depth_val - d_map[j, i]) 

                        # w_val = initial_w * leakage_weight * overlap_weight
                        # w_val = initial_w * leakage_weight * overlap_weight
                        w_val = initial_w * overlap_weight * leakage_weight 
                        if (i == x and j == y):
                            w_val = initial_w

                        new_img[j, i, 0] += img_linear[y, x, 0] * w_val
                        new_img[j, i, 1] += img_linear[y, x, 1] * w_val
                        new_img[j, i, 2] += img_linear[y, x, 2] * w_val
                        weight_map[j, i] += w_val

        return new_img, weight_map

    new_image_linear, weight = apply_bokeh_loop(color_image_linear, coc, depth_map)

    # Normalize
    for y in range(height):
        for x in range(width):
            if weight[y, x] > 0:
                new_image_linear[y, x] /= weight[y, x]
            else:
                print("NOT GOOD!!!!!!!!!!!!!!!!!!!!!")
                new_image_linear[y, x] = [1.0, 0.0, 0.0]  # still red but now in [0..1] range

    # --- Convert back from linear to gamma space and then to uint8 ---
    new_image_gamma = new_image_linear ** (1.0 / gamma_val)
    new_image_8u = (new_image_gamma * 255.0).astype(np.uint8)

    return new_image_8u

def apply_bokeh_4(color_image, depth_map, focus_depth, max_coc_radius, depth_of_field):
    import numpy as np
    import math
    from numba import njit, prange
    from PIL import Image

    print("GOT FOCUS DEPTH:", focus_depth)
    print("Color image shape:", color_image.shape)
    print("Depth map shape:", depth_map.shape)
    print("Color image dtype:", color_image.dtype)
    print("Depth map dtype:", depth_map.dtype)
    print("Depth map range:", np.min(depth_map), "-", np.max(depth_map))

    # Convert color image to linear space
    color_image_float = color_image.astype(np.float32) / 255.0
    gamma_val = 2.2
    color_image_linear = color_image_float ** gamma_val

    @njit(fastmath=True)
    def calculate_coc_radius(pixel_depth, focal_depth, max_coc):
        return max_coc * max(abs(focal_depth - pixel_depth) - depth_of_field / 2, 0)

    @njit(fastmath=True)
    def calculate_overlap_weight(sample_coc_radius, distance_to_center):
        """
        O(rp) = 0 if rp <= dp
        O(rp) = rp - dp if dp <= rp < dp + 1
        O(rp) = 1 if rp >= dp + 1
        """
        if sample_coc_radius <= distance_to_center:
            return 0.0
        elif sample_coc_radius >= distance_to_center + 1:
            return 1.0
        else:
            return sample_coc_radius - distance_to_center

    @njit(fastmath=True)
    def calculate_intensity_weight(coc_radius):
        """
        I(rp) ∝ 1/rp^2
        """
        if coc_radius < 0.1:  # Avoid division by zero
            return 1.0
        return 1.0 / (coc_radius * coc_radius)

    @njit(fastmath=True)
    def calculate_leakage_weight(sample_depth, focus_depth, center_coc_radius):
        """
        L(zp) ∝ rc if zp > zf
        L(zp) = 1 if zp <= zf
        """
        if sample_depth <= focus_depth:
            return 1.0
        else:
            return center_coc_radius / max_coc_radius

    @njit(fastmath=True)
    def precompute_coc_and_intensity(d_map, focus_depth, max_coc, height, width):
        coc_map = np.zeros((height, width), dtype=np.float32)
        intensity_map = np.zeros((height, width), dtype=np.float32)
        
        for y in range(height):
            for x in range(width):
                coc = calculate_coc_radius(d_map[y, x], focus_depth, max_coc)
                coc_map[y, x] = coc
                intensity_map[y, x] = calculate_intensity_weight(coc)
        
        return coc_map, intensity_map

    @njit(fastmath=True, parallel=True)
    def process_image(img_linear, d_map, focus_depth, max_coc):
        height, width = d_map.shape
        new_img = np.zeros_like(img_linear, dtype=np.float32)
        weight_map = np.zeros((height, width), dtype=np.float32)
        print("vals are: ", max_coc, width)
        max_coc = max_coc / 1000 * width
        print("max_coc_radius:", max_coc)

        # Precompute CoC and intensity values for all pixels
        coc_map, intensity_map = precompute_coc_and_intensity(d_map, focus_depth, max_coc, height, width)

        # Process each center pixel
        for y in prange(height):
            print("Processing row", y)
            
            for x in range(width):
                center_coc = coc_map[y, x]

                # Calculate maximum radius to look at
                radius = int(max_coc + 1)
                
                # Look at neighboring pixels
                for j in range(max(0, y - radius), min(height, y + radius + 1)):
                    for i in range(max(0, x - radius), min(width, x + radius + 1)):
                        # Get precomputed values for sample pixel
                        sample_coc = coc_map[j, i]
                        # Calculate distance from sample to center
                        dist = np.sqrt((i - x)**2 + (j - y)**2)
                        
                        if dist > sample_coc:
                            continue
                        
                        overlap = max(min(sample_coc - dist, 1.0), 0.0)

                        # leakage = calculate_leakage_weight(d_map[j, i], focus_depth, center_coc)
                        leakage = 1.0
                        if d_map[j, i] > focus_depth:
                            leakage = center_coc / max_coc_radius
                            
                        intensity = intensity_map[j, i]

                        # Final weight is product of all factors
                        weight = overlap * intensity * leakage
                        
                        if (i == x and j == y):
                            weight = intensity
                        
                        # Add weighted contribution
                        new_img[y, x] += img_linear[j, i] * weight
                        weight_map[y, x] += weight

        # Normalize
        for y in prange(height):
            for x in prange(width):
                if weight_map[y, x] > 0:
                    new_img[y, x] /= weight_map[y, x]

        return new_img

    # Process the image
    new_image_linear = process_image(color_image_linear, depth_map, focus_depth, max_coc_radius)

    # Get the CoC map for visualization
    height, width = depth_map.shape
    max_coc = max_coc_radius / 1000 * width
    coc_map, _ = precompute_coc_and_intensity(depth_map, focus_depth, max_coc, height, width)
    
    # Save CoC map as image
    coc_normalized = (coc_map - np.min(coc_map)) / (np.max(coc_map) - np.min(coc_map)) * 255
    coc_image = Image.fromarray(coc_normalized.astype(np.uint8))
    coc_image.save('coc_map_bokeh4.png')

    # Convert back from linear to gamma space and then to uint8
    new_image_gamma = new_image_linear ** (1.0 / gamma_val)
    new_image_8u = (new_image_gamma * 255.0).astype(np.uint8)

    return new_image_8u