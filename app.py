"""
This file starts a WebApp using Flask in the backend. 

"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
import classification_folder.segment_and_classify_segformer as sc

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output_segmentation'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/blog')
def blog():
    return render_template('blog.html')

@app.route('/demo')
def demo():
    return render_template('demo.html')

@app.route('/output_segmentation/<filename>')
def serve_segmented_image(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({"message": "No file uploaded"}), 400
    
    image = request.files['image']
    filename = secure_filename(image.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(save_path)
    output = f"Image '{filename}' is segmented... Working on displaying it."
    masks = sc.get_masks(save_path)
    analysis, output_filename = sc.run_analysis(masks, img_name=filename)
    return jsonify({
        "message": output,
        "analysis": analysis,
        "segmented_image": f"/output_segmentation/{output_filename}"
    })

if __name__ == '__main__':
    app.run(debug=True)
