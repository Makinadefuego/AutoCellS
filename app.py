import os
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

S3_BUCKET = 'autocellsstorage'
S3_KEY = 'best_model.keras'
LOCAL_FILE_PATH = 'models/best_model.keras'
# Función para listar archivos en un directorio
def list_files_in_directory(directory):
    try:
        files = os.listdir(directory)
        print(f"Archivos en el directorio {directory}: {files}")
    except FileNotFoundError:
        print(f"El directorio {directory} no existe.")
    except Exception as e:
        print(f"Error al listar archivos en {directory}: {str(e)}")


# Descargar el archivo desde S3 si no está presente localmente
def download_file_from_s3(bucket, key, local_path):
    s3 = boto3.client('s3', aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'), aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'))
    try:
        if not os.path.exists(local_path):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3.download_file(bucket, key, local_path)
            print(f"Archivo descargado desde S3: {local_path}")
        else:
            print(f"Archivo ya existe: {local_path}")
    except NoCredentialsError:
        print("Error: No se encontraron credenciales de AWS.")
    except PartialCredentialsError:
        print("Error: Credenciales incompletas de AWS.")
    except Exception as e:
        print(f"Error al descargar el archivo: {str(e)}")

# Descargar el archivo antes de que cualquier otro código lo necesite
download_file_from_s3(S3_BUCKET, S3_KEY, LOCAL_FILE_PATH)

# Listar archivos en el directorio actual y en el directorio de modelos
list_files_in_directory('.')
list_files_in_directory('models')


# Asegurarse de que el archivo existe
if not os.path.exists(LOCAL_FILE_PATH):
    print(f"Error: el archivo {LOCAL_FILE_PATH} no existe después de la descarga")
else:
    print(f"El archivo {LOCAL_FILE_PATH} está presente")

from flask import Flask, request, render_template, redirect, url_for, send_from_directory, flash, jsonify
from werkzeug.utils import secure_filename
import os
import json
from models.predict import segment_image
from models.preprocess import preprocess_images
from models.segmentation import segment_dataset
from models.classify import clasificar_celula  

app = Flask(__name__)
app.config.from_pyfile('config.py')

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff'}

# Las clases de células son las siguientes:
# 0: Interfase
# 1: Profase
# 2: Metafase
# 3: Anafase
# 4: Telofase

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route('/handle_upload', methods=['POST'])
def handle_upload():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']
    nombre_clasificacion = request.form.get('nombre_clasificacion')
    model = request.form.get('model')

    if not nombre_clasificacion or not model:
        flash('Missing classification name or model')
        return redirect(request.url)

    classification_folder = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(nombre_clasificacion))
    os.makedirs(classification_folder, exist_ok=True)

    print(f'Classification folder created: {classification_folder}')

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(classification_folder, filename))
        print(f'File saved as {filename}')
        # Ejecuta la segmentación:
        segment_dataset(classification_folder)
        
        # Ejecuta el preprocesamiento:
        preprocess_images(classification_folder)

        # Ejecuta la clasificación:
        cells_dir = os.path.join(classification_folder, 'preprocessed')
        cell_classes = {}

        for cell_filename in os.listdir(cells_dir):
            if allowed_file(cell_filename):
                cell_path = os.path.join(cells_dir, cell_filename)
                cell_class = clasificar_celula(cell_path)

                # Obtener la imagen original a la que pertenece la célula (nombre de la imagen sin incluir el número de célula (_#))
                original_image = cell_filename.split('_')[:-1]

                # Añadir la información de la clase y la imagen original
                cell_classes[cell_filename] = {
                    'id': cell_filename.split('_')[2].split('.')[0],
                    'class': cell_class,
                    'original_image': original_image
                }

        # Guardar la clasificación en un archivo JSON
        classification_json_path = os.path.join(classification_folder, 'classification.json')
        with open(classification_json_path, 'w') as f:
            json.dump(cell_classes, f)
    else:
        flash('Invalid file extension')
        return redirect(request.url)

    flash('File successfully uploaded')
    return redirect(url_for('my_classifications'))

@app.route('/myclassifications')
def my_classifications():
    classifications = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('myclassifications.html', classifications=classifications)

@app.route('/myclassifications/<classification_name>')
def review_classification(classification_name):  
    classification_folder = os.path.join(app.config['UPLOAD_FOLDER'], classification_name)
    json_path = os.path.join(classification_folder, "classification.json")
    with open(json_path, 'r') as f:
        classification_data = json.load(f)
    return render_template('review_classification.html', classification_name=classification_name, classification_data=classification_data)

@app.route('/save_classification/<classification_name>', methods=['POST'])
def save_classification(classification_name):
    classification_folder = os.path.join(app.config['UPLOAD_FOLDER'], classification_name)
    json_path = os.path.join(classification_folder, "classification.json")

    # Obtener la información de la solicitud AJAX
    filename = request.form.get('filename')
    new_class = request.form.get('newClass')

    # Cargar el JSON actual
    with open(json_path, 'r') as f:
        classification_data = json.load(f)

    # Actualizar la clasificación de la célula
    classification_data[filename] = new_class

    # Guardar el JSON actualizado
    with open(json_path, 'w') as f:
        json.dump(classification_data, f)

    # Puedes devolver una respuesta simple, como "OK"
    return "OK"

@app.route('/segment', methods=['GET', 'POST'])
def segment():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            output_path = segment_image(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'models/best_model.keras')
            return send_from_directory(app.config['UPLOAD_FOLDER'], os.path.basename(output_path))
        else:
            return jsonify({'error': 'Invalid file extension'})
    else:
        return render_template('predict.html')

@app.route('/uploads/<classification_name>/<path:filename>')
def uploaded_file(classification_name, filename):
    try:
        return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER'], classification_name), filename)
    except FileNotFoundError:
        app.logger.error(f"Archivo no encontrado: /uploads/{classification_name}/{filename}")
        return "Archivo no encontrado", 404
    
@app.route('/clasificacion_interactiva/<classification_name>')
def clasificacion_interactiva(classification_name):
    print(classification_name)
    classification_folder = os.path.join(app.config['UPLOAD_FOLDER'], classification_name)
    json_path = os.path.join(classification_folder, "classification.json")

    # Carga los datos de clasificación desde el archivo JSON
    with open(json_path, 'r') as f:
        classification_data = json.load(f)
    
    # Se cargan las imagenes de microscopio que estan en una carpeta antes de preprocessed
    original_images = []
    
    for file in os.listdir(classification_folder):
        if allowed_file(file):
            original_images.append(file)

    print(classification_folder)
    print(classification_data)
    print(original_images)

    return render_template('clasificacion_interactiva.html', 
                           classification_name=classification_name, 
                           classification=classification_data,
                            original_images=original_images)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5800)))
