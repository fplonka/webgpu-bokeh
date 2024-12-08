import os
from flask import Flask, request, jsonify, send_from_directory, render_template, make_response
from werkzeug.utils import secure_filename
from gradio_client import Client, handle_file
from gradio_client.exceptions import AppError
import numpy as np
import shutil
from PIL import Image

# Import the apply_bokeh function from main.py
from main import apply_bokeh

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

depth_client = Client("depth-anything/Depth-Anything-V2")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # if True: # debugging
            # return jsonify({'depth_map_url': f'/uploads/depth_map.png'})

        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], "original.png")
        file.save(filepath)
        
        app.logger.info(f"Original image saved to {filepath}")
        
        app.logger.info("Calling Depth-Anything-V2 API")
        result = depth_client.predict(
            image=handle_file(filepath),
            api_name="/on_submit"
        )
        app.logger.info(f"API response: {result}")
        
        if not result or len(result) < 2 or not result[1]:
            raise ValueError("Unexpected API response format")
        
        depth_map_path = result[1]
        
        if not os.path.exists(depth_map_path):
            raise FileNotFoundError(f"Depth map file not found: {depth_map_path}")
        
        # Copy the depth map to our uploads folder
        depth_map_filename = 'depth_map.png'
        local_depth_map_path = os.path.join(app.config['UPLOAD_FOLDER'], depth_map_filename)
        shutil.copy(depth_map_path, local_depth_map_path)
        
        return jsonify({'depth_map_url': f'/uploads/{depth_map_filename}'})
    except AppError as e:
        app.logger.error(f"Error from Depth-Anything-V2 API: {str(e)}")
        return jsonify({'error': 'Error processing the image. Please try a different image or contact support.'}), 500
    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({'error': 'An unexpected error occurred. Please try again or contact support.'}), 500

@app.route('/process', methods=['POST'])
def process_image():
    data = request.json
    x, y = data['x'], data['y']
    multiplier = data['multiplier']
    
    # Load the original image and depth map
    original_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'original.png')
    depth_map_path = os.path.join(app.config['UPLOAD_FOLDER'], 'depth_map.png')
    
    original_image = np.array(Image.open(original_image_path))
    depth_map = np.array(Image.open(depth_map_path).convert('L'))
    depth_map = depth_map.astype(np.float32) / 255.0
    
    # Save input data for comparison
    np.save("color_image_app.npy", original_image)
    np.save("depth_map_app.npy", depth_map)
    
    # Debug logging for input data
    app.logger.info(f"Original image shape: {original_image.shape}")
    app.logger.info(f"Depth map shape: {depth_map.shape}")
    app.logger.info(f"Depth map min: {np.min(depth_map)}, max: {np.max(depth_map)}")
    
    # Get the focus depth from the clicked point
    focus_depth = depth_map[int(y * depth_map.shape[0]), int(x * depth_map.shape[1])]
    
    # Apply bokeh effect using the function from main.py
    print("HAVE DEPTH:", focus_depth)
    processed_image = apply_bokeh(original_image, depth_map, focus_depth, 2**float(multiplier))
    
    # Save processed image
    processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed.png')
    Image.fromarray(processed_image).save(processed_image_path)
    
    # Create response with no-cache headers
    response = make_response(jsonify({'processed_image_url': f'/uploads/processed.png'}))
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    
    return response

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)