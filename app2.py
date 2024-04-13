# main.py
import numpy as np
from flask import Flask, jsonify, request, render_template

import pyttsx3

import base64

import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests
import json
import os


app = Flask(__name__)

upload_folder = os.path.join('static', 'img')

app.config['UPLOAD'] = upload_folder


@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Speech generated successfully"})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=False)
