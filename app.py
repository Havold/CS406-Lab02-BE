from flask import Flask, request, jsonify, send_from_directory
import cv2
import os
import numpy as np
import json
from werkzeug.utils import secure_filename
from scipy.spatial import distance
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = './uploads'
IMAGE_FOLDER = './images/seg'  # Thư mục chứa 12000 ảnh trên server
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER

def calculate_histogram(filepath, use_equalized):
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    # Tính histogram
    if (use_equalized == True):
        image = cv2.equalizeHist(image)
    hist = cv2.calcHist([image], [0], None, [256], [0,255]).flatten()
    return hist

@app.route('/', methods=['GET'])
def hello():
    return 'HELLO'

# Hàm gửi ảnh từ server cho client
@app.route('/image/<category>/<filename>', methods=['GET'])
def get_image(category, filename):
    folder_path = os.path.join(app.config['IMAGE_FOLDER'], category)
    return send_from_directory(folder_path, filename)

# POST endpoint to handle image upload and find similar images
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    use_equalized = request.args.get('equalized').lower() == 'true'
    top = request.args.get('top')
    if (top):
        top = int(top)

    # Calculate histogram of the uploaded image
    uploaded_hist = calculate_histogram(filepath, use_equalized)

    if (use_equalized == True):
        json_name = 'histograms_equalized.json'
    else:
        json_name = 'histograms.json'

    with open(json_name, 'r') as file:
        hist_data = json.load(file)

    print(use_equalized)
    print(json_name)

    # Compare uploaded histogram with stored histograms and calculate distances
    distances = []
    for record in hist_data:
        db_hist = np.array(record['histogram']).flatten()
        dist = distance.euclidean(db_hist, uploaded_hist)
        distances.append((record['category'], record['image_name'], dist))

    # Sort by distance and select top 10 similar images
    distances.sort(key = lambda x: x[2])
    topFilter = distances[:top]

    # Send back the result to the frontend with image URLs
    similar_images = [
        {
            'category': d[0],
            'img_name': d[1],
            'distance': d[2],
            'image_url': f'/image/{d[0]}/{d[1]}'
        }
        for d in topFilter
    ]

    return jsonify({'similarImages': similar_images})


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    app.run(host='0.0.0.0', port=5000, debug=True)
