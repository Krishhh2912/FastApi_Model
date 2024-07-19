import uvicorn
from fastapi import FastAPI
from BankNotes import BankNote
import numpy as np
import pickle
import pandas as pd

app = FastAPI()

# Load the trained model
pickle_in = open('classifier.pkl', "rb")
model = pickle.load(pickle_in)

@app.get('/')
def index():
    return {'message': 'Hello, User'}

@app.get('/{name}')
def get_name(name: str):
    return {'message': f'Hello, {name}'}

@app.post('/predict')
def predict_banknote(data: BankNote):
    data = data.dict()
    print(data)
    
    # Use string keys to access dictionary values
    variance = data['variance']
    skewness = data['skewness']
    curtosis = data['curtosis']
    entropy = data['entropy']

    prediction = model.predict([[variance, skewness, curtosis, entropy]])
    if prediction[0] > 0.5:
        prediction = "It's a Fake Note"
    else:
        prediction = "It's a Bank Note"
    return {
       'prediction': prediction
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
