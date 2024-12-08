import numpy as np

# Load the numpy arrays
color_image_main = np.load("color_image.npy")
depth_map_main = np.load("depth_map.npy")
color_image_app = np.load("color_image_app.npy")
depth_map_app = np.load("depth_map_app.npy")

# Compare the arrays
color_diff = np.sum(np.abs(color_image_main - color_image_app))
depth_diff = np.sum(np.abs(depth_map_main - depth_map_app))

print(f"Color image difference: {color_diff}")
print(f"Depth map difference: {depth_diff}")

# Optionally, you can also compare the shapes and ranges
print(f"Color image shapes: main {color_image_main.shape}, app {color_image_app.shape}")
print(f"Depth map shapes: main {depth_map_main.shape}, app {depth_map_app.shape}")

print(f"Color image ranges: main {np.min(color_image_main)}-{np.max(color_image_main)}, app {np.min(color_image_app)}-{np.max(color_image_app)}")
print(f"Depth map ranges: main {np.min(depth_map_main)}-{np.max(depth_map_main)}, app {np.min(depth_map_app)}-{np.max(depth_map_app)}")