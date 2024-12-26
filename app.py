import os
from flask import Flask, request, jsonify, send_from_directory, render_template, make_response
from werkzeug.utils import secure_filename
from gradio_client import Client, handle_file
from gradio_client.exceptions import AppError
import numpy as np
import shutil
from PIL import Image
import json

# from main import apply_bokeh
# from main import apply_bokeh_2
from main import apply_bokeh_3
from main import apply_bokeh_4

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
        img = handle_file(filepath)
        print("IMG IS:", img)
        
        # Add debug logging to see the actual request
        old_send_data = depth_client.send_data
        def logging_send_data(data, hash_data, protocol):
            print("\nDEBUG: Sending data to API:")
            print("Data:", json.dumps(data, indent=2))
            print("Hash data:", json.dumps(hash_data, indent=2))
            print("Protocol:", protocol)
            return old_send_data(data, hash_data, protocol)
            
        depth_client.send_data = logging_send_data
        
        result = depth_client.predict(
            image=img,
            api_name="/on_submit"
        )
        # Restore original method
        depth_client.send_data = old_send_data
        
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
    depth_of_field = float(data.get('depthOfField', 0.1))  # Default to 0.1 if not provided
    
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
    processed_image = apply_bokeh_4(original_image, depth_map, focus_depth, 2**float(multiplier), depth_of_field)
    
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