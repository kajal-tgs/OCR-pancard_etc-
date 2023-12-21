from flask import Flask, request, jsonify
import easyocr
from PIL import Image
import io
import numpy as np

import boto3

from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
import urllib.parse
from bs4.element import Comment

import json
import os
from dotenv import load_dotenv
import logging
import uuid

import functools
import time
from operator import itemgetter

## Constants
SERVICE_PORT=4001
RUN_GUID='dev'



# SERVICE_REF="postman-dev"
SERVICE_REF="02d45aaa"

# load_dotenv()

# Loggers
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', '%m-%d-%Y %H:%M:%S')

# file_handler = logging.FileHandler('logs-{}.log'.format(RUN_GUID))
# file_handler.setLevel(logging.INFO)
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)

consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.INFO)
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler)

# Wrapper function to measure execution time for any function
# include @timer on top of funtion 
def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        logger.info("[TIMER] Finished {} in {} secs".format(repr(func.__name__), round(run_time, 3)))
        return value
    return wrapper



app = Flask(__name__)
@timer
def extract_text_from_image(image_file):
    image = Image.open(image_file)
    image_np = np.array(image)
    reader = easyocr.Reader(['en', 'hi'])
    result = reader.readtext(image_np, paragraph=True)
    text_only = [item[1] for item in result]

    return text_only
    #return result

@timer
def convert_text_to_json(text):
    data = {}
    for item in text:
        for (bbox, content, confidence) in item:
            data[content] = confidence
    return data

@timer
@app.route('/extract_pan_text', methods=['POST'])
def extract_pan_text():
    pan_card = request.files['pan_card']
    text = extract_text_from_image(pan_card)
    return jsonify({"text": text})

@timer
@app.route('/extract_aadhaar_text', methods=['POST'])
def extract_aadhaar_text():
    if 'aadhaar_card' not in request.files:
        return jsonify({"error": "No image file provided for Aadhaar card"})

    aadhaar_card = request.files['aadhaar_card']
    text = extract_text_from_image(aadhaar_card)
    return jsonify({"text": text})

@timer
@app.route('/pan_to_json', methods=['POST'])
def pan_to_json():
    if 'pan_card' not in request.files:
        return jsonify({"error": "No image file provided for PAN card"})

    pan_card = request.files['pan_card']
    text = extract_text_from_image(pan_card)
    json_data = convert_text_to_json(text)
    return jsonify({"json_data": json_data})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4001)














#import json
#import pytesseract
#import numpy as np
#import sys2
#import re
#import os
#from PIL import Image
#import ftfy
#import io
#import easyocr
#reader = easyocr.Reader(['en', 'hi'])
#result = reader.readtext(filename,paragraph=True)
#result
#adhaar='adhaar_card.jpg'
#adhaar_read = reader.readtext(adhaar,paragraph=True)
#adhaar_read

