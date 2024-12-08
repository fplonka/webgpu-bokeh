# main.py

import numpy as np
import math
from PIL import Image
# from gen_test_data import generate_scene, save_images, load_images

def apply_bokeh(color_image, depth_map, focus_depth, multiplier):
    # focus_depth = 0.2
    # # focus_depth = 0.21960784494876862
    # focus_depth = np.float64(focus_depth)
    # focus_depth = 0.21960784494876862
    print("GOT FOCUS DEPTH:", focus_depth)
    # print(type(focus_depth))

    minf = focus_depth / (focus_depth + 1)
    maxf = (focus_depth + 1)/4
    focal_length = (minf + maxf) / 2
    # FOCAL_LENGTH = 0.316

    s1 = 0.5 * (focus_depth + 1 - np.sqrt( (focus_depth + 1)*(focus_depth + 1 - 4 * focal_length) ))
    f1 = 1 + focus_depth - s1
    print("s1 = ", s1, "f1 = ", f1)
    print("min f:", focus_depth / (focus_depth + 1))
    print("max f:", (focus_depth + 1)/4)
    
    assert(minf < focal_length < maxf)
    assert(0 < focal_length < (focus_depth + 1)/4)

    height, width = depth_map.shape
    print("height = ", height, "width = ", width)
    new_image = np.zeros_like(color_image, dtype=np.float32)
    weight = np.zeros((height, width), dtype=np.float32)
    
    print("Color image shape:", color_image.shape)
    print("Depth map shape:", depth_map.shape)
    print("Color image dtype:", color_image.dtype)
    print("Depth map dtype:", depth_map.dtype)
    print("Depth map range:", np.min(depth_map), "-", np.max(depth_map))

    print("Depth map range:", np.min(depth_map), "-", np.max(depth_map))

    import numba
    from numba import njit, prange

    @njit(fastmath=True)
    def calculate_coc_radius(depth, s1):
        s2 = s1 - focus_depth + depth
        res = (abs(s2 - s1) * focal_length) / (s2 * (s1 - focal_length))
        # if (res == 0):
        #     print("variable values:")
        #     print("depth = ", depth)
        #     print("s1 = ", s1)
        #     print("s2 = ", s2)
        #     print("focal_length = ", focal_length)
        #     print("res = ", res)
        #     print()
        return res
    
    @njit(fastmath=True, parallel=True)
    def compute_coc(depth_map, s1, width, multiplier):
        height = depth_map.shape[0]
        coc = np.zeros((height, width), dtype=np.float32)
        for y in prange(height):
            for x in prange(width):
                coc_radius = calculate_coc_radius(depth_map[y, x], s1)
                coc_radius /= 2.0  # actually we get the diameter
                coc_radius *= width * 0.001 * multiplier
                coc[y, x] = coc_radius
        return coc
    
    print("MULTIPLIER:", multiplier)
    coc = compute_coc(depth_map, s1, width, multiplier)
    
    zero_count = np.count_nonzero(coc == 0)
    print("Number of zeros in coc:", zero_count)
    
    @njit(fastmath=True)
    def apply_bokeh_loop(color_image, coc, depth_map, new_image, weight):
        height, width = coc.shape
        for y in range(height):
            if y % 100 == 0:
                print("y = ", y)
            for x in range(width):
                coc_radius = coc[y, x]
                depth = depth_map[y, x]

                area = np.pi * float(coc_radius) * float(coc_radius)
                if area <= 0:
                    area = 1

                coc_radius_int = math.ceil(coc_radius) 
                
                initial_w = 1.0 / area
                
                for j in range(max(0, y - coc_radius_int), min(height, y + coc_radius_int + 1)): 
                    for i in range(max(0, x - coc_radius_int), min(width, x + coc_radius_int + 1)):

                        dist = np.sqrt((i - x)**2 + (j - y)**2)
                        overlap_weight = coc_radius - dist 
                        overlap_weight = max(0, overlap_weight)
                        overlap_weight = min(1, overlap_weight)
                        # overlap_weight = 1
                        
                        leakage_weight = 1
                        if depth > depth_map[j, i]:
                            leakage_weight = 1 + 1 * min(0, depth - depth_map[j, i])
                        leakage_weight = 1

                        w = initial_w * overlap_weight * leakage_weight

                        if (i == x and j == y):
                            w = initial_w

                        new_image[j, i, 0] += color_image[y, x, 0] * w
                        new_image[j, i, 1] += color_image[y, x, 1] * w
                        new_image[j, i, 2] += color_image[y, x, 2] * w
                        weight[j, i] += w
                        

        return new_image, weight

    # Call the numba function
    new_image, weight = apply_bokeh_loop(color_image, coc, depth_map, new_image, weight)

    # Explicitly normalize the image using the square root of the mean
    for y in range(height):
        for x in range(width):
            if weight[y, x] > 0:
                new_image[y, x] = new_image[y, x] / weight[y, x]
                # new_image[y, x, 0] = np.sqrt(new_image[y, x, 0] / weight[y, x])
                # new_image[y, x, 1] = np.sqrt(new_image[y, x, 1] / weight[y, x])
                # new_image[y, x, 2] = np.sqrt(new_image[y, x, 2] / weight[y, x])
            else:
                print("NOT GOOD!!!!!!!!!!!!!!!!!!!!!")
                new_image[y, x] = [255, 0, 0]
                # new_image[y, x] = color_image[y, x]
    
    return new_image.astype(np.uint8)

# def generate_and_save_new_scene():
#     color_image, depth_map = generate_scene(width=1024, height=768, num_spheres=20)
#     save_images(color_image, depth_map)

def load_images(color_path='images/color_scene.png', depth_path='images/depth_map.png'):
    color_image = np.array(Image.open(color_path))
    depth_map = np.array(Image.open(depth_path).convert('L'))  # Convert to grayscale
    return color_image, depth_map

def load_existing_scene():
    # return load_images(color_path="images/test8_scene.png", depth_path="images/test8_depth.png")
    # return load_images(color_path="images/test6_scene.png", depth_path="images/test6_depth.png")
    # return load_images(color_path="images/test4_scene.jpeg", depth_path="images/test4_depth.png")
    # return load_images(color_path="images/test3_scene.png", depth_path="images/test3_depth.png")
    return load_images(color_path="images/test1_scene.png", depth_path="images/test1_depth.png")
    # return load_images(color_path="images/test2_scene.jpg", depth_path="images/test2_depth.png")

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

    # Save input data for comparison
    np.save("color_image.npy", loaded_color)
    np.save("depth_map.npy", normalized_depth)
    
    print("AA")
    print(np.min(normalized_depth))
    print(np.max(normalized_depth))

    # Apply bokeh effect
    bokeh_image = apply_bokeh(loaded_color, normalized_depth, 0.7137255)

    print("Bokeh effect applied.")
    print("Processed image shape:", bokeh_image.shape)
    print("Processed image dtype:", bokeh_image.dtype)

    from PIL import Image
    Image.fromarray(bokeh_image).save("images/bokeh_image.png")
