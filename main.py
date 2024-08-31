# main.py

import numpy as np
from gen_test_data import generate_scene, save_images, load_images

# User-adjustable parameters
FOCAL_LENGTH = 0.28  # Focal length (0 to 1)
FOCUS_DEPTH = 0.937    # Focus depth (0 to 1)
# APERTURE = 0.1       # Aperture size (smaller values for stronger bokeh)

def calculate_coc_radius(depth, s1):
    s2 = s1 - FOCUS_DEPTH + depth
    res = (abs(s2 - s1) * FOCAL_LENGTH) / (s2 * (s1 - FOCAL_LENGTH))
    # if abs(depth - FOCUS_DEPTH) < 0.002:
    #     print("s1 = ", s1, "s2 = ", s2, "res = ", res, "depth = ", depth)
    return res

def is_in_bokeh_shape(dx, dy, radius):
    # Circular bokeh shape
    return dx*dx + dy*dy <= radius*radius

def apply_bokeh(color_image, depth_map):
    global FOCAL_LENGTH
    minf = FOCUS_DEPTH / (FOCUS_DEPTH + 1)
    maxf = (FOCUS_DEPTH + 1)/4
    FOCAL_LENGTH = (minf + maxf) / 2
    # FOCAL_LENGTH = 0.316

    s1 = 0.5 * (FOCUS_DEPTH + 1 - np.sqrt( (FOCUS_DEPTH + 1)*(FOCUS_DEPTH + 1 - 4 * FOCAL_LENGTH) ))
    f1 = 1 + FOCUS_DEPTH - s1
    print("s1 = ", s1, "f1 = ", f1)
    print("min f:", FOCUS_DEPTH / (FOCUS_DEPTH + 1))
    print("max f:", (FOCUS_DEPTH + 1)/4)
    
    assert(minf < FOCAL_LENGTH < maxf)
    assert(0 < FOCAL_LENGTH < (FOCUS_DEPTH + 1)/4)
    

    height, width = depth_map.shape
    print("height = ", height, "width = ", width)
    new_image = np.zeros_like(color_image, dtype=np.float32)
    weight = np.zeros((height, width), dtype=np.float32)

    import numba
    from numba import njit, prange

    @njit(fastmath=True)
    def apply_bokeh_numba(color_image, depth_map, new_image, weight, s1, width):
        height = depth_map.shape[0]
        for y in range(height):
            print("y = ", y)
            for x in range(width):
                coc_radius = calculate_coc_radius(depth_map[y, x], s1)
                coc_radius *= width * 0.025
                coc_radius = round(coc_radius) 
                # coc_radius = 100
                
                area = np.pi * float(coc_radius) * float(coc_radius)
                if area <= 0:
                    area = 1
                
                # coc_radius = 50
                # coc_radius = 20 / (abs(depth_map[y, x] - FOCUS_DEPTH) + 1) 

                for j in range(max(0, y - coc_radius), min(height, y + coc_radius + 1)): 
                    for i in range(max(0, x - coc_radius), min(width, x + coc_radius + 1)):
                        if (i - x)**2+ (j - y)**2 <= coc_radius**2:
                            if depth_map[j, i] >= depth_map[y, x] or True:
                                # r, g, b = color_image[y, x]
                                # brightness = (r + g + b) / 3 / 255
                                # if (area == 0):
                                #     print("UH OH")
                                #     continue

                                # new_image[j, i, 0] += (color_image[y, x, 0] / area)**2
                                # new_image[j, i, 1] += (color_image[y, x, 1] / area)**2
                                # new_image[j, i, 2] += (color_image[y, x, 2] / area)**2
                                # weight[j, i] += (1.0 / area)**2

                                new_image[j, i, 0] += color_image[y, x, 0] / area
                                new_image[j, i, 1] += color_image[y, x, 1] / area
                                new_image[j, i, 2] += color_image[y, x, 2] / area
                                weight[j, i] += 1.0 / area
        return new_image, weight

    @njit(fastmath=True)
    def calculate_coc_radius(depth, s1):
        s2 = s1 - FOCUS_DEPTH + depth
        res = (abs(s2 - s1) * FOCAL_LENGTH) / (s2 * (s1 - FOCAL_LENGTH))
        return res

    # Call the numba function
    new_image, weight = apply_bokeh_numba(color_image, depth_map, new_image, weight, s1, width)

    # Explicitly normalize the image using the square root of the mean
    for y in range(height):
        for x in range(width):
            if weight[y, x] > 0:
                new_image[y, x] = new_image[y, x] / weight[y, x]
                # new_image[y, x, 0] = np.sqrt(new_image[y, x, 0] / weight[y, x])
                # new_image[y, x, 1] = np.sqrt(new_image[y, x, 1] / weight[y, x])
                # new_image[y, x, 2] = np.sqrt(new_image[y, x, 2] / weight[y, x])
            else:
                new_image[y, x] = [0, 0, 0]
    
    return new_image.astype(np.uint8)

def generate_and_save_new_scene():
    color_image, depth_map = generate_scene(width=1024, height=768, num_spheres=20)
    save_images(color_image, depth_map)

def load_existing_scene():
    # return load_images(color_path="images/test1_scene.png", depth_path="images/test1_depth.png")
    return load_images(color_path="images/test2_scene.jpg", depth_path="images/test2_depth.png")

if __name__ == "__main__":
    # Load the saved scene
    loaded_color, loaded_depth = load_existing_scene()
    print("Existing scene loaded.")
    
    print("Color image shape:", loaded_color.shape)
    print("Depth map shape:", loaded_depth.shape)
    print("Color image dtype:", loaded_color.dtype)
    print("Depth map range:", np.min(loaded_depth), "-", np.max(loaded_depth))

    # Normalize depth map to 0-1 range
    normalized_depth = loaded_depth.astype(np.float32) / 255.0
    
    print("AA")
    print(np.min(normalized_depth))
    print(np.max(normalized_depth))

    # Apply bokeh effect
    bokeh_image = apply_bokeh(loaded_color, normalized_depth)

    print("Bokeh effect applied.")
    print("Processed image shape:", bokeh_image.shape)
    print("Processed image dtype:", bokeh_image.dtype)

    from PIL import Image
    Image.fromarray(bokeh_image).save("images/bokeh_image.png")
