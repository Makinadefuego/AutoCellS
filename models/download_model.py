import gdown, sys

def download_model(model_name, model_path):
    url = f'https://drive.google.com/uc?id={model_name}'
    gdown.download(url, model_path, quiet=False)
    