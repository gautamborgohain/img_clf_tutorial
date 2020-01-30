import requests
import io
from PIL import Image
import os
import logging

log_level = os.environ.get('LOG_LEVEL',  logging.DEBUG)
logging.basicConfig(level=log_level)

def download_image(url):
	img_request = requests.get(url, stream=True)
	img = Image.open(io.BytesIO(img_request.content))
	return img
