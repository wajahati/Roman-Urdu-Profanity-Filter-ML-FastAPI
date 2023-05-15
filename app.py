# -*- coding: utf-8 -*-
"""
Created on Sun May 14 02:11:08 2023

@author: mr.laptop
"""

# 1. Library imports
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from API import Profanity
from API import Price

import string
import emoji

import re
import pickle

app = FastAPI()

# Load the model from the pickle file
with open('profanity_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the model from the pickle file
with open('vector_model.pkl', 'rb') as f:
    vectorizer = pickle.load(f)




roman_urdu_stop_words = [   "ai", "ayi", "hy", "hai", "main", "ki", "tha", "koi", "ko", "sy", "woh", 
                            "bhi", "aur", "wo", "yeh", "rha", "hota", "ho", "ga", "ka", "le", "lye", 
                            "kr", "kar", "lye", "liye", "hotay", "waisay", "gya", "gaya", "kch", "ab",
                            "thy", "thay", "houn", "hain", "han", "to", "is", "hi", "jo", "kya", "thi",
                            "se", "pe", "phr", "wala", "waisay", "us", "na", "ny", "hun", "rha", "raha",
                            "ja", "rahay", "abi", "uski", "ne", "haan", "acha", "nai", "sent", "photo", 
                            "you", "kafi", "gai", "rhy", "kuch", "jata", "aye", "ya", "dono", "hoa", 
                            "aese", "de", "wohi", "jati", "jb", "krta", "lg", "rahi", "hui", "karna", 
                            "krna", "gi", "hova", "yehi", "jana", "jye", "chal", "mil", "tu", "hum", "par", 
                            "hay", "kis", "sb", "gy", "dain", "krny", "tou",
                            'keh', 'ko', 'ki', 'mein', 'aur', 'ya', 'lekin', 'jaisay', 'yah', 'to', 'agar', 'kuch', 'to', 'kya',   
                            'hota', 'hoti', 'hotay', 'bhi', 'kar', 'diya', 'jab', 'tak', 'tha', 'thi', 'thay', 'kartay',   
                            'karti', 'karta', 'apnay', 'aap', 'kuch', 'baaz', 'kabhi', 'hum', 'tum', 'aap', 'woh', 'unhon',   
                            'usay', 'usi', 'un', 'unhon', 'unhain', 'ho', 'gaya', 'gayi', 'gaye', 'karna', 'kartay', 'rahey',   
                            'rahi', 'raha', 'kuch', 'kaisay', 'yahan', 'wahan', 'ab', 'kahan', 'kab', 'kis', 'kaisay', 'kyun']


def preprocess_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove usernames and hashtags
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#[A-Za-z0-9_]+', '', text)
    
    text = text.lower()
    # Replace emojis with their descriptions
    text = emoji.demojize(text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stop words and stem the remaining words
    text = ' '.join([word for word in text.split() if word not in roman_urdu_stop_words])
    # Join the tokens back into a string
    return text


@app.post('/profanityCheck')
def profanityCheck(data:Profanity):
    data = data.dict()
    message=data['message']
    
    preprocessed_text = preprocess_text(message)

    # Transform the preprocessed text into a feature vector
    feature_vector = vectorizer.transform([preprocessed_text])

    # Make predictions using the trained model
    prediction = model.predict(feature_vector)
    output_dict = {"profanity": int(prediction[0])}
    return JSONResponse(content=output_dict)

@app.post('/predictCategory')
def predictCategory(data:Profanity):
    

   # print(classifier.predict([[variance,skewness,curtosis,entropy]]))
    prediction = {"aa":"asdasdd"}
    return JSONResponse(content=prediction)

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
