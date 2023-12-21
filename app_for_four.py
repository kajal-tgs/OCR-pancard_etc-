from flask import Flask, request, jsonify
import easyocr
from PIL import Image
import io
import numpy as np
import openai, numpy as np
import google.generativeai as palm
from langchain.llms import OpenAI
from langchain import PromptTemplate

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


OPENAI_API_KEY="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"
BARD_API_KEY = 'AIzaSyCNlJJDudY5D9qr4v_U1K9FsZYCn4u5m5Q'
BARD_MODEL = 'models/chat-bison-001'


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

##########################  PAN-CARD ###########################


@timer
def pan_card_openai(text_only):
    # initialize the models
    #logger.info("[PARSER] {}() resume word count:: {}".format("parse_resume", (len(resume_txt.split(" ")))))
    #resume_txt = (" ").join(resume_txt.split(" ")[:500])
    #logger.info("[PARSER] {}() resume word count:: {}".format("parse_resume", (len(resume_txt.split(" ")))))
    openai = OpenAI(model_name="text-davinci-003",openai_api_key=OPENAI_API_KEY,
                    temperature=0.0, max_tokens=2000)

    template = """Given a Pan Card, your task is to parse it and extract  the following information and convert to JSON format with keys: name (as String);  pan_no. (as String);  Date_of_Birth (format YYYY/MM/DD);  Father_name (as a string). 

    Pan Card: {pan_txt}

    The JSON object:
    """

    prompt_template = PromptTemplate(input_variables=["resume"], template=template)
    pan_json = openai(prompt_template.format(pan_txt=text_only))
    #logger.info("[PARSER] {}() resume_json:: {}".format("parse_resume",resume_json.replace('\n',' ')))
    
    # Parse JSON text to object
    pan_json = json.loads(pan_json)
    return pan_json



@timer
def pan_card_bard(text_only):
    # initialize the models
    #logger.info("[pan_card] {}() pan word count:: {}".format("pan_card_bard", (len(text_only.split(" ")))))
    #pan_txt = (" ").join(pan_txt.split(" ")[:1000])
    #logger.info("[pan_card] {}() pan word count:: {}".format("pan_card_bard", (len(text_only.split(" ")))))
    
    palm.configure(api_key=BARD_API_KEY)
    prompt = """Given a Pan Card, your task is to parse it and extract  the following information and convert to JSON format with keys: name (as String);  pan_no. (as String);  Date_of_Birth (format YYYY/MM/DD);  Father_name (as a string). 

    Pan Card: {}

    The JSON object:
    """.format(text_only)    

    logger.info("[pan_card] {}() prompt:: {}".format("pan_card_bard", prompt))
    palm_messages = palm.chat(model=BARD_MODEL,messages=prompt,temperature=0.0)
    logger.info("[pan_card] {}() palm_messages:: {}".format("pan_card_bard",palm_messages))
    _res = palm_messages.messages[1]['content']
    #pan_json = json.loads(_res.split('```json')[1].replace('```','').replace('\n',''))
    #return pan_json
    return _res


#################################  AADHAAR CARD  ####################################
@timer
def aadhaar_card_openai(text_only):
    # initialize the models
    #logger.info("[PARSER] {}() resume word count:: {}".format("parse_resume", (len(resume_txt.split(" ")))))
    #resume_txt = (" ").join(resume_txt.split(" ")[:500])
    #logger.info("[PARSER] {}() resume word count:: {}".format("parse_resume", (len(resume_txt.split(" ")))))
    openai = OpenAI(model_name="text-davinci-003",openai_api_key=OPENAI_API_KEY,
                    temperature=0.0, max_tokens=2000)

    template = """Given a Aadhaar Card, your task is to parse it and extract  the following information and convert to JSON format with keys: name (as String);  Aadhaar_no. (as String);  Date_of_Birth (format YYYY/MM/DD);  address (as a string). 

    Aadhaar Card: {aadhaar_txt}

    The JSON object:
    """

    prompt_template = PromptTemplate(input_variables=["aadhaar_text"], template=template)
    aadhaar_json = openai(prompt_template.format(aadhaar_txt=text_only))
    #logger.info("[PARSER] {}() resume_json:: {}".format("parse_resume",resume_json.replace('\n',' ')))
    
    # Parse JSON text to object
    aadhaar_json = json.loads(aadhaar_json)
    return aadhaar_json

@timer
def aadhaar_card_bard(text_only):
    # initialize the models
    #logger.info("[pan_card] {}() pan word count:: {}".format("pan_card_bard", (len(text_only.split(" ")))))
    #pan_txt = (" ").join(pan_txt.split(" ")[:1000])
    #logger.info("[pan_card] {}() pan word count:: {}".format("pan_card_bard", (len(text_only.split(" ")))))
    
    palm.configure(api_key=BARD_API_KEY)
    prompt = """Given a Aadhaar Card, your task is to parse it and extract  the following information and convert to JSON format with keys: name (as String);  Aadhaar_no. (as String);  Date_of_Birth (format YYYY/MM/DD);  address (as a string). 

    Aadhaar Card: {}

    The JSON object:
    """.format(text_only)    

    logger.info("[aadhaar_card] {}() prompt:: {}".format("aadhaar_card_bard", prompt))
    palm_messages = palm.chat(model=BARD_MODEL,messages=prompt,temperature=0.0)
    logger.info("[aadhaar_card] {}() palm_messages:: {}".format("aadhaar_card_bard",palm_messages))
    _res = palm_messages.messages[1]['content']
    #pan_json = json.loads(_res.split('```json')[1].replace('```','').replace('\n',''))
    #return pan_json
    return _res


############################### BANK STATEMENT ###############################

@timer
def bank_statement_openai(text_only):
    # initialize the models
    #logger.info("[PARSER] {}() resume word count:: {}".format("parse_resume", (len(resume_txt.split(" ")))))
    #resume_txt = (" ").join(resume_txt.split(" ")[:500])
    #logger.info("[PARSER] {}() resume word count:: {}".format("parse_resume", (len(resume_txt.split(" ")))))
    openai = OpenAI(model_name="text-davinci-003",openai_api_key=OPENAI_API_KEY,
                    temperature=0.0, max_tokens=2000)

    template = """Given a Bank Statement, your task is to parse it and extract the following information and convert to JSON format with keys:  bank_name (as String); bank_address (as String); statement_from_date (format YYYY/MM/DD);  statement_to_date (format YYYY/MM/DD);  account_holder_name (as a string);  account_number (as String);  transactions as a List of transaction_date (format YYYY/MM/DD), description (as String), ref_id (as String), withdrawal_amount (as Number), deposit_amount (as Number), balance_amount (as Number). 

 
    Bank Statement: {statement_txt}

    The JSON object:
    """

    prompt_template = PromptTemplate(input_variables=["statement_text"], template=template)
    statement_json = openai(prompt_template.format(statement_txt=text_only))
    #logger.info("[PARSER] {}() resume_json:: {}".format("parse_resume",resume_json.replace('\n',' ')))
    
    # Parse JSON text to object
    statement_json = json.loads(statement_json)
    return statement_json

@timer
def bank_statement_bard(text_only):
    # initialize the models
    #logger.info("[pan_card] {}() pan word count:: {}".format("pan_card_bard", (len(text_only.split(" ")))))
    #pan_txt = (" ").join(pan_txt.split(" ")[:1000])
    #logger.info("[pan_card] {}() pan word count:: {}".format("pan_card_bard", (len(text_only.split(" ")))))
    
    palm.configure(api_key=BARD_API_KEY)
    prompt = """Given a Bank Statement, your task is to parse it and extract  the following information and convert to JSON format with keys:  bank_name (as String); bank_address (as String); statement_from_date (format YYYY/MM/DD);  statement_to_date (format YYYY/MM/DD);  account_holder_name (as a string);  account_number (as String);  transactions as a List of transaction_date (format YYYY/MM/DD), description (as String), ref_id (as String), withdrawal_amount (as Number), deposit_amount (as Number), balance_amount (as Number). 

 
    Bank Statement: {}

    The JSON object:
    """.format(text_only)    

    logger.info("[bank_statement] {}() prompt:: {}".format("bank_statement_bard", prompt))
    palm_messages = palm.chat(model=BARD_MODEL,messages=prompt,temperature=0.0)
    logger.info("[aadhaar_card] {}() palm_messages:: {}".format("bank_statement_bard",palm_messages))
    _res = palm_messages.messages[1]['content']
    #pan_json = json.loads(_res.split('```json')[1].replace('```','').replace('\n',''))
    #return pan_json
    return _res



######################################  Invoice  ##############################################


@timer
def invoice_openai(text_only):
 
    openai = OpenAI(model_name="text-davinci-003",openai_api_key=OPENAI_API_KEY,
                    temperature=0.0, max_tokens=2000)

    template = """Given a invoice, your task is to parse it and extract the following information and convert to JSON format with keys:  invoice_no. (as String); date (format YYYY/MM/DD); due_date (format YYYY/MM/DD); amount (as a string);  bill_to company or preson details as a list of   bill_to_company/person_name (as a string),  bill_to_GST_number (as String),  bill_to_address (as a string);  seller details as a list of   seller_ company/person_name(as a string), seller_address(as a string), GST_no.(as a string); items as a list of title(as a string), quantity(as a string),  rate(as a string);   amount tax informations like  GST(as a string);  VAT(as a string) . 

 
    Invoice : {invoice_txt}

    The JSON object:
    """

    prompt_template = PromptTemplate(input_variables=["invoice_text"], template=template)
    invoice_json = openai(prompt_template.format(invoice_txt=text_only))
    #logger.info("[PARSER] {}() resume_json:: {}".format("parse_resume",resume_json.replace('\n',' ')))
    
    # Parse JSON text to object
    invoice_json = json.loads(invoice_json)
    return invoice_json

@timer
def invoice_bard(text_only):
    
    palm.configure(api_key=BARD_API_KEY)
    prompt = """Given a invoice, your task is to parse it and extract  the following information and convert to JSON format with keys:  bank_name (as String); bank_address (as String); statement_from_date (format YYYY/MM/DD);  statement_to_date (format YYYY/MM/DD);  account_holder_name (as a string);  account_number (as String);  transactions as a List of transaction_date (format YYYY/MM/DD), description (as String), ref_id (as String), withdrawal_amount (as Number), deposit_amount (as Number), balance_amount (as Number). 

 
    Invoice: {}

    The JSON object:
    """.format(text_only)    

    logger.info("[bank_statement] {}() prompt:: {}".format("invoice_bard", prompt))
    palm_messages = palm.chat(model=BARD_MODEL,messages=prompt,temperature=0.0)
    logger.info("[aadhaar_card] {}() palm_messages:: {}".format("invoice_bard",palm_messages))
    _res = palm_messages.messages[1]['content']
    #pan_json = json.loads(_res.split('```json')[1].replace('```','').replace('\n',''))
    #return pan_json
    return _res


############################
# ---- FLASK METHODS ----#
############################

app = Flask(__name__)


@timer
@app.route('/extract_pan_text_bard', methods=['POST'])
def extract_pan_text_bard():
    pan_card = request.files['pan_card']
    #logger.info("[pan_card] {}() req:: {}".format("extract_pan_text",req))

    text_only = extract_text_from_image(pan_card)
    pan_json = pan_card_bard(text_only)
    logger.info("[pan_card] {}() >> pan_json:: {}".format("pan_card", json.dumps(pan_json)))
    return json.dumps({'success': True, 'data':{'pan card':pan_json}}), 200, {'ContentType': 'application/json'}
   

@timer
@app.route('/extract_pan_text_openai', methods=['POST'])
def extract_pan_text_openai():
    pan_card = request.files['pan_card']
    #logger.info("[pan_card] {}() req:: {}".format("extract_pan_text",req))

    text_only = extract_text_from_image(pan_card)
    pan_json = pan_card_openai(text_only)
    logger.info("[pan_card] {}() >> pan_json:: {}".format("pan_card", json.dumps(pan_json)))
    return json.dumps({'success': True, 'data':{'pan card':pan_json}}), 200, {'ContentType': 'application/json'}
   

@timer
@app.route('/extract_aadhaar_text_bard', methods=['POST'])
def extract_aadhaar_text_bard():
    aadhaar_card = request.files['aadhaar_card']
    #logger.info("[pan_card] {}() req:: {}".format("extract_pan_text",req))

    text_only = extract_text_from_image(aadhaar_card)
    aadhaar_json = aadhaar_card_bard(text_only)
    logger.info("[aadhaar_card] {}() >> aadhaar_json:: {}".format("aadhaar_card", json.dumps(aadhaar_json)))
    return json.dumps({'success': True, 'data':{'aadhaar card':aadhaar_json}}), 200, {'ContentType': 'application/json'}
   

@timer
@app.route('/extract_aadhaar_text_openai', methods=['POST'])
def extract_aadhaar_text_openai():
    aadhaar_card = request.files['aadhaar_card']
    #logger.info("[pan_card] {}() req:: {}".format("extract_pan_text",req))

    text_only = extract_text_from_image(aadhaar_card)
    aadhaar_json = aadhaar_card_openai(text_only)
    logger.info("[aadhaar_card] {}() >> aadhaar_json:: {}".format("aadhaar_card", json.dumps(aadhaar_json)))
    return json.dumps({'success': True, 'data':{'pan card':aadhaar_json}}), 200, {'ContentType': 'application/json'}
   
  
@timer
@app.route('/extract_statement_text_bard', methods=['POST'])
def extract_statement_text_bard():
    bank_statement = request.files['bank_statement']
    #logger.info("[pan_card] {}() req:: {}".format("extract_pan_text",req))

    text_only = extract_text_from_image(bank_statement)
    statement_json = bank_statement_bard(text_only)
    logger.info("[bank_statement] {}() >> statement_json:: {}".format("bank_statement", json.dumps(statement_json)))
    return json.dumps({'success': True, 'data':{'bank statement':statement_json}}), 200, {'ContentType': 'application/json'}
   

@timer
@app.route('/extract_statement_text_openai', methods=['POST'])
def extract_statement_text_openai():
    bank_statement = request.files['bank_statement']
    #logger.info("[pan_card] {}() req:: {}".format("extract_pan_text",req))

    text_only = extract_text_from_image(bank_statement)
    statement_json = bank_statement_openai(text_only)
    logger.info("[bank_statement] {}() >> statement_json:: {}".format("bank_statement", json.dumps(statement_json)))
    return json.dumps({'success': True, 'data':{'bank statement':statement_json}}), 200, {'ContentType': 'application/json'}
   
  
@timer
@app.route('/extract_invoice_text_bard', methods=['POST'])
def extract_invoice_text_bard():
    invoice = request.files['invoice']
    #logger.info("[pan_card] {}() req:: {}".format("extract_pan_text",req))

    text_only = extract_text_from_image(invoice)
    invoice_json = invoice_bard(text_only)
    logger.info("[invoice] {}() >> invoice_json:: {}".format("invoice", json.dumps(invoice_json)))
    return json.dumps({'success': True, 'data':{'invoice':invoice_json}}), 200, {'ContentType': 'application/json'}
   

@timer
@app.route('/extract_invoice_text_openai', methods=['POST'])
def extract_invoice_text_openai():
    invoice = request.files['invoice']
    #logger.info("[pan_card] {}() req:: {}".format("extract_pan_text",req))

    text_only = extract_text_from_image(invoice)
    invoice_json = invoice_openai(text_only)
    logger.info("[invoice] {}() >> invoice_json:: {}".format("invoice", json.dumps(invoice_json)))
    return json.dumps({'success': True, 'data':{'invoice':invoice_json}}), 200, {'ContentType': 'application/json'}
   
  






if __name__ == '__main__':
    app.run(port=SERVICE_PORT)












