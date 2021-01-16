from flask import Flask, redirect, url_for
from flask_restful import Api, Resource, reqparse, abort, fields, marshal_with
from flask_sqlalchemy import SQLAlchemy
from transformers import BertTokenizer, TFBertModel, TFBertForSequenceClassification, TFPreTrainedModel
app = Flask(__name__)
api = Api(app)

line_post_args = reqparse.RequestParser()
line_post_args.add_argument("line", type=str,  help="Please send the line of the receipt that you want to predict", required= True)




class Predict(Resource):
    
    def __init__(self):
        
        
        model_name='bert-base-cased'
        self.tokenizer=BertTokenizer.from_pretrained(model_name)
        
        
        self.model=TFBertForSequenceClassification.from_pretrained('C:/Users/lbbre/Documents/ECAM 5/PRD/content/assets')
        self.max_seq_len=32
    def formatage_text(self, test_sentence):
        
        return self.tokenizer.encode_plus(
                              test_sentence,                      
                              add_special_tokens = True,# add [CLS], [SEP]
                              truncation=True, 
                              max_length = self.max_seq_len, # max length of the text that can go to BERT
                              padding='max_length', # add [PAD] tokens
                              return_token_type_ids= True,
                              return_attention_mask = True,# add attention mask to not focus on pad tokens
        )
    
    def prediction(self, features):
        
        y_pred = self.model.predict([features['input_ids'],features['attention_mask'], features['token_type_ids']])[0].argmax(axis=-1)[0]
        return y_pred
        
    def post(self):
        args = line_post_args.parse_args()
        sentence=args['line']
        features=self.formatage_text(sentence)
        pred=self.prediction(features).tolist()
        if (pred==2):
            pred="telephone"
        elif (pred==0):
            pred="misc"
        elif (pred==1):
            pred="items"    
        elif (pred==3):
            pred="address"
        elif (pred==4):
            pred="Storename" 
        elif (pred==5):
            pred="tax" 
        elif (pred==6):
            pred="subtotal"    
        elif (pred==7):
            pred="total"  
        elif (pred==8):
            pred="Date"      
            
            
        return {"prediction":pred}
    
api.add_resource(Predict,"/predict") 



   
@app.route("/home")
def home():
    return "hello this is the main page"

@app.route("/home/<name>")
def user(name):
    
    return (f"hello {name}!")

@app.route("/home/redirect")
def redir():
    return redirect(url_for("home"))

if __name__ == "__main__":
	app.run()