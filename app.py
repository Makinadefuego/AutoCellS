from flask import Flask, request, render_template, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
import os
from models.predict import segment_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB límite de subida
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'tiff'}
app.secret_key = os.urandom(24)  # Genera una clave secreta aleatoria

# Crea el directorio de subidas si no existe
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    files = request.files.getlist('file')
    if not files:
        flash('No selected file')
        return redirect(request.url)
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            try:
                segmented_image_path = segment_image(file_path)
            except Exception as e:
                flash(f"Error processing image: {e}")
                return redirect(request.url)
            return redirect(url_for('show_image', filename=os.path.basename(segmented_image_path)))
    
    return redirect('/')

@app.route('/uploads/<filename>')
def show_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5800)))
