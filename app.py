from flask import Flask, request, render_template, make_response, redirect
from werkzeug.utils import secure_filename


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = None
app.config['ALLOWED_EXTENSIONS'] = [ '.png', '.jpg', '.jpeg', '.tiff']

#Para solucionar el error 413 Request Entity Too Large: The data value transmitted exceeds the capacity limit.


@app.route('/')
def home():
    print('Home')
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        files = request.files
        list_files = files.getlist('file')

        for file in list_files:
            if file.filename == '':
                return redirect('/')
            if file:
                name = secure_filename(file.filename)
                ext = name.split('.')[-1]
                if '.' + ext in app.config['ALLOWED_EXTENSIONS']:
                    file.save(f'{app.config["UPLOAD_FOLDER"]}/{name}')
                else:
                    return 'Invalid file extension'
    except Exception as e:
        print(e)
            
    
    return redirect('/')


if __name__ == "__main__":
    app.run(debug=True, port=6969)