"""
This file starts a WebApp using Flask in the backend. 

"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
import classification_folder.segment_and_classify_segformer as sc
import bibtexparser
from bibtexparser.bparser import BibTexParser
from bibtexparser.customization import convert_to_unicode

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output_segmentation'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER


def parse_bibtex(file_path : str) -> list:
    """
    Parse a bibtex file and return a list of formatted citations.
    """
    with open(file_path, 'r', encoding='utf-8') as bibtex_file:
        parser = BibTexParser()
        parser.customization = convert_to_unicode
        bib_database = bibtexparser.load(bibtex_file, parser=parser)
        citations = []
        for entry in bib_database.entries:
            # Format authors
            authors = entry.get('author', '').split(' and ')
            formatted_authors = []
            for author in authors:
                parts = author.split(', ')
                if len(parts) > 1:
                    formatted_authors.append(f"{parts[1]} {parts[0]}")
                else:
                    formatted_authors.append(parts[0])
            authors_str = ', '.join(formatted_authors[:-1]) + ' & ' + formatted_authors[-1] if len(formatted_authors) > 1 else formatted_authors[0]
            # Format the citation
            citation = f"{authors_str} ({entry.get('year', 'n.d.')}). {entry.get('title', '')}"
            # Add journal/conference information
            if 'journal' in entry:
                citation += f". <i>{entry['journal']}</i>"
            elif 'booktitle' in entry:
                citation += f". In <i>{entry['booktitle']}</i>"
            # Add volume and pages if available
            if 'volume' in entry:
                citation += f", {entry['volume']}"
            if 'pages' in entry:
                citation += f", {entry['pages']}"
            # Add URL if available
            if 'url' in entry:
                citation += f". <a href='{entry['url']}' target='_blank'>{entry['url']}</a>"
            
            citations.append(citation)
        return citations

@app.route('/')
def index():
    """
    Renders index.html file
    """
    return render_template('index.html')

@app.route('/blog')
def blog():
    """
    Renders blog.html, parses citation.bib and pass it as argument
    """
    citations = parse_bibtex('citation.bib')
    return render_template('blog.html', citations=citations)

@app.route('/demo')
def demo():
    """
    Renders demo.html that contains a yt video
    """
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
