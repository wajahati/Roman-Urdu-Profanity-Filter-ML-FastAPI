# -*- coding: utf-8 -*-
"""
Created on Sun May 14 02:11:08 2023

@author: mr.laptop
"""

# 1. Library imports
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from Categories import Category
from Categories import Price

import pickle
from keras_preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
from keras.models import load_model
import operator

import openai
# 2. Create the app object
app = FastAPI()

tokenizer = 0 
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
model = load_model('model-LSTM.h5')

# Utility function for data cleaning, natural language processing concepts

def decontract(sentence):
    sentence = str(sentence)
    sentence = re.sub(r"won't", "will not", sentence)
    sentence = re.sub(r"can\'t", "can not", sentence)
    sentence = re.sub(r"n\'t", " not", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"\'s", " is", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'t", " not", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"\'m", " am", sentence)
    return sentence

def cleanPunc(sentence): 
    sentence = str(sentence)
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned

def keepAlpha(sentence):
    sentence = str(sentence)
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', '', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent

def removeStopWords(sentence):
    sentence = str(sentence)
    global re_stop_words
    return re_stop_words.sub("", sentence)

stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"])

re_stop_words = re.compile(r"\b(" + "|".join(stopwords) + ")\\W", re.I)

stemmer = SnowballStemmer("english")
def stemming(sentence):
    sentence = str(sentence)
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence

classes = ['3D Printer Filament',
 'A/V Cables & Connectors',
 'Action Camcorder Accessories',
 'Action Camcorder Mounts',
 'Activity Trackers & Pedometers',
 'Adapters, Cables & Chargers',
 'Air Conditioners',
 'Air Purifier Filters & Parts',
 'Air Purifiers',
 'All Cell Phones with Plans',
 'All Desktops',
 'All Flat-Panel TVs',
 'All Headphones',
 'All Laptops',
 'All Memory Cards',
 'All Microwaves',
 'All Point & Shoot Cameras',
 'All Printers',
 'All Refrigerators',
 'All TV Stands',
 'All Tablets',
 'All Unlocked Cell Phones',
 'Amps & Effects',
 'Antennas & Adapters',
 'Apple Watch',
 'Apple Watch Accessories',
 'Apple Watch Bands & Straps',
 'Appliance Parts & Accessories',
 'Appliances',
 'Audio',
 'Bakeware',
 'Best Buy Gift Cards',
 'Binoculars',
 'Binoculars, Telescopes & Optics',
 'Blenders',
 'Blenders & Juicers',
 'Bluetooth & Wireless Speakers',
 'Bookshelf Speakers',
 'Built-In Dishwashers',
 'CD & DVD Media Storage',
 'CD Players & Turntables',
 'Cables & Connectors',
 'Camcorder Accessories',
 'Camera Bags & Cases',
 'Camera Bags, Cases & Straps',
 'Camera Batteries',
 'Camera Batteries & Power',
 'Camera Lenses',
 'Cameras & Camcorders',
 'Car Amplifiers',
 'Car Audio',
 'Car Audio Installation Parts',
 'Car Chargers',
 'Car Electronics & GPS',
 'Car Installation Parts & Accessories',
 'Car Speakers',
 'Car Stereo Receivers',
 'Car Subwoofers',
 'Car Subwoofers & Enclosures',
 'Cases',
 'Cases, Covers & Keyboard Folios',
 'Cell Phone Accessories',
 'Cell Phone Batteries & Power',
 'Cell Phone Cases & Clips',
 'Cell Phones',
 'Coffee Makers',
 'Coffee Pods',
 'Coffee Pods & Beans',
 'Coffee, Tea & Espresso',
 'Computer Accessories & Peripherals',
 'Computer Cards & Components',
 'Computer Keyboards',
 'Computers & Tablets',
 'Connected Home',
 'Connected Home & Housewares',
 'Cooktops',
 'Cookware',
 'Cookware, Bakeware & Cutlery',
 'Cordless Telephones',
 'DJ & Lighting Equipment',
 'DSLR Lenses',
 'Dash Installation Kits',
 'Deck Harnesses',
 'Deck Installation Parts',
 'Desks',
 'Desktop & All-in-One Computers',
 'Digital Camera Accessories',
 'Digital Cameras',
 'Dishwashers',
 'Drones & Accessories',
 'Drums & Percussion',
 'Dryers',
 'Earbud & In-Ear Headphones',
 'Effects',
 'Electric Dryers',
 'Electric Griddles & Hotplates',
 'Electric Ranges',
 'Electric Tea Kettles',
 'External Hard Drives',
 'Fans',
 'Filters & Accessories',
 'Fireplaces',
 'Fitness & GPS Watches',
 'Flashes & Accessories',
 'Flashes, Lighting & Studio',
 'Food Preparation Utensils',
 'Freezers & Ice Makers',
 'Full-Size Blenders',
 'Furniture & Decor',
 'GPS Navigation & Accessories',
 'Game Room & Bar Furniture',
 'Gas Dryers',
 'Gas Ranges',
 'Grills',
 'Guitar Accessories',
 'Hair Care',
 'Hard Drives & Storage',
 'Headphones',
 'Health & Fitness Accessories',
 'Health, Fitness & Beauty',
 'Heaters',
 'Heating, Cooling & Air Quality',
 'Home Audio',
 'Home Audio Accessories',
 'Home Theater Systems',
 'Household Batteries',
 'Household Essentials',
 'Housewares',
 'Humidifiers',
 'In-Ceiling Speakers',
 'In-Wall & In-Ceiling Speakers',
 'Instrument Instructional Books',
 'Internal Batteries',
 'Keyboards',
 'Kitchen Gadgets',
 'LED Monitors',
 'Laptop Accessories',
 'Laptop Bags & Cases',
 'Laptop Batteries',
 'Laptop Chargers & Adapters',
 'Laptops',
 'Learning & Education',
 'Living Room Furniture',
 'Luggage',
 'Luggage, Bags & Travel',
 'Magnolia Accessories',
 'Magnolia Home Theater',
 'Magnolia TV Stands, Mounts & Furniture',
 'Marine & Powersports',
 'Memory Cards',
 'Mice',
 'Mice & Keyboards',
 'Microphones',
 'Microphones & Accessories',
 'Microphones & Live Sound',
 'Microwaves',
 'Mirrorless Lenses',
 'Mixers & Mixer Accessories',
 'Monitor & Screen Accessories',
 'Monitors',
 'Multi-Cup Coffee Makers',
 'Musical Instrument Accessories',
 'Musical Instruments',
 'Name Brands',
 'Networking & Wireless',
 'Nintendo 3DS',
 'Nintendo DS',
 'Nintendo DS Games',
 'Office & School Supplies',
 'Office Chairs',
 'Office Electronics',
 'Office Furniture & Storage',
 'On-Ear Headphones',
 'Outdoor Heating',
 'Outdoor Living',
 'Outdoor Seating',
 'Outdoor Speakers',
 'Over-Ear & On-Ear Headphones',
 'Over-Ear Headphones',
 'PC Gaming',
 'PC Laptops',
 'PS3 Games',
 'PS4 Games',
 'Patio Furniture & Decor',
 'Pedals',
 'Personal Care & Beauty',
 'Photography Accessories',
 'PlayStation 3',
 'PlayStation 4',
 'Point & Shoot Cameras',
 'Portable Chargers/Power Packs',
 'Pre-Owned Games',
 'Prime Lenses',
 'Printer Ink',
 'Printer Ink & Toner',
 'Printers, Ink & Toner',
 'Projector Mounts & Screens',
 'Projector Screens',
 'Projectors',
 'Projectors & Screens',
 'Range Hoods',
 'Ranges',
 'Ranges, Cooktops & Ovens',
 'Receivers & Amplifiers',
 'Recording Equipment',
 'Refrigerators',
 'Screen Protectors',
 'Security Camera Systems',
 'Security Cameras & Surveillance',
 'Shavers & Trimmers',
 'Sheet Music',
 'Sheet Music & DVDs',
 'Single Ovens',
 'Slow Cookers, Crock Pots & Roaster Ovens',
 'Small Kitchen Appliances',
 'Smartwatch Accessories',
 'Smartwatch Bands',
 'Smartwatches & Accessories',
 'Software',
 'Sound Bars',
 'Speaker Accessories',
 'Speakers',
 'Specialty Appliances',
 'Steamers, Rice Cookers & Pressure Cookers',
 'Surge Protectors & Power',
 'Systems',
 'TV & Home Theater',
 'TV & Home Theater Accessories',
 'TV Mounts',
 'TV Stands',
 'TV Stands, Mounts & Furniture',
 'TV, Movie & Character Toys',
 'TVs',
 'Tablets',
 'Tea Kettles',
 'Telephone Accessories',
 'Telephones & Communication',
 'Toasters',
 'Toner',
 'Toys to Life',
 'Toys, Games & Drones',
 'Tripods & Monopods',
 'USB Cables & Hubs',
 'USB Flash Drives',
 'Universal Camera Bags & Cases',
 'Unlocked Cell Phones',
 'Vacuum & Floor Care Accessories',
 'Vacuum Cleaners & Floor Care',
 'Video Game Accessories',
 'Video Games',
 'Wall Art',
 'Wall Chargers & Power Adapters',
 'Wall Mount Range Hoods',
 'Wall Ovens',
 'Washers & Dryers',
 'Washing Machines',
 'Wearable Technology',
 'Wii',
 'Xbox 360',
 'Xbox 360 Games',
 'Xbox One',
 'Xbox One Games',
 'iPad & Tablet Accessories',
 'iPhone 6s Cases',
 'iPhone 6s Plus Cases',
 'iPhone Accessories',
 'iPhone Cases & Clips',
 'iPod & MP3 Player Accessories',
 'Others']

def findCategory(name,description):
  name = name.lower()
  name = decontract(name)
  name = cleanPunc(name)
  name = keepAlpha(name)
  name = removeStopWords(name)
  name = stemming(name)

  description = description.lower()
  description = decontract(description)
  description = cleanPunc(description)
  description = keepAlpha(description)
  description = removeStopWords(description)
  description = stemming(description)

  information = name + description

  # loading
#  tokenizer = 0 
#  with open('tokenizer.pickle', 'rb') as handle:
#    tokenizer = pickle.load(handle)

  sequences = tokenizer.texts_to_sequences([information])
  # print(sequence)
  x = pad_sequences(sequences, maxlen=500)

#  model = load_model('model-LSTM.h5')
  prediction = model.predict(x)
  predScores = [score for pred in prediction for score in pred]
  predDict = {}
  for cla,score in zip(classes,predScores):
    predDict[cla] = score

  D = dict(sorted(predDict.items(), key=operator.itemgetter(1),reverse=True)[:10])
  # return dict(filter(lambda elem: elem[1] > 0.5, D.items()))
  data_array = [{'category': k} for k, v in D.items()]
  return data_array
def suggest(title,description):
    openai.api_key = "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"
    prompt = f"Based on the following product information in Pakistan in pakistani rupees, suggest a price range :\n\nTitle: {title}\nDescription: {description}\n\nSuggested Price:"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=64,
        n=1,
        stop=None,
        temperature=0.5,
        )

    response_text = response.choices[0].text.strip()
    print(response_text)
    # Define a regular expression pattern to match the price range
    price_pattern = r"^(PKR|(Rs\.) )?\d{1,3}(,\d{3})*(\.\d+)?\s*-\s*(Rs\. )?\d{1,3}(,\d{3})*(\.\d+)?$"

    # Search the response text for the price pattern
    match = re.search(price_pattern, response_text)

    # If a match is found, extract the price range
    if match:
        price_range = match.group()
        return price_range
    else:
        return("Could not find price range")
    

# # 3. Index route, opens automatically on http://127.0.0.1:8000
# @app.get('/')
# def index():
#     return {'message': 'Hello, World'}
    

@app.post('/suggestPrice')
def suggestPrice(data:Price):
    data = data.dict()
    title=data['title']
    description=data['description']
    
    
    data = {"suggestedprice": suggest(title,description)}
    return JSONResponse(content=data)

@app.post('/predictCategory')
def predictCategory(data:Category):
    data = data.dict()
    title=data['title']
    description=data['description']

   # print(classifier.predict([[variance,skewness,curtosis,entropy]]))
    prediction = findCategory(title,description)
    return JSONResponse(content=prediction)

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
