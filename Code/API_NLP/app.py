from flask import Flask,request, url_for, redirect, render_template, jsonify


import pickle
import numpy as np
from transformers import BertTokenizer, TFBertModel, TFBertForSequenceClassification, TFPreTrainedModel
import keras

app = Flask(__name__)

model_name='bert-base-cased'
tokenizer=BertTokenizer.from_pretrained(model_name)
modelPath='assets' # a remplacer par un wget
model=TFBertForSequenceClassification.from_pretrained(modelPath)

def formatage_text(test_sentence):
        
        return tokenizer.encode_plus(
                              test_sentence,                      
                              add_special_tokens = True,# add [CLS], [SEP]
                              truncation=True, 
                              max_length = 32, # max length of the text that can go to BERT
                              padding='max_length', # add [PAD] tokens
                              return_token_type_ids= True,
                              return_attention_mask = True,# add attention mask to not focus on pad tokens
        )
    
    
def prediction(features):
       
       y_pred = model.predict([features['input_ids'],features['attention_mask'], features['token_type_ids']])[0].argmax(axis=-1)[0]
       return y_pred
       
def makePred(sentence):
    
    features=formatage_text(sentence)
    pred=prediction(features).tolist()
    if (pred==2):
        pred="Storename"
    elif (pred==0):
        pred="Item"
    elif (pred==1):
        pred="Total"    
    elif (pred==3):
        pred="Tax"
    elif (pred==4):
        pred="Address" 
    elif (pred==5):
        pred="Misc." 
    elif (pred==6):
        pred="Telephone"    
    elif (pred==7):
        pred="Subtotal"  
    elif (pred==8):
        pred="Date"      
        
        
    return pred   

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    
    sentence = [x for x in request.form.values()]
    sentence=sentence[0]
    res=makePred(sentence)
    
    return render_template('home.html',pred='prediction is {}'.format(res))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    sentence = [x for x in request.form.values()]
    sentence=sentence[0]
    res=makePred(sentence)
    return jsonify(res)

if __name__ == '__main__':

    app.run(host="0.0.0.0", port=80)
