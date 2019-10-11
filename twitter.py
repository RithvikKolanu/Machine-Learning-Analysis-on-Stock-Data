import tweepy
from tweepy import api
import json
import pandas as pd
import numpy as np
auth = tweepy.OAuthHandler('vg2b6lu2Ly0dzZyinO59QrLM6', 'yMTEKdE6OaDRqalmt3iccVPoBiCEQOudd5xqssA1uWsg7aqtlA')
auth.set_access_token('1112897842312445952-xazweRwNuhVSIuoU5Bk6x1JcisHUo5', '0xvlUB50Sp6d5LwLAuOdYlwKFvonnFhz2IExElqf4Aeam')
api = tweepy.API(auth)
import keras.models
from keras.models import model_from_json
from keras.datasets import imdb
from nltk import word_tokenize
from keras.preprocessing import sequence
import re

json_file = open('Model_Save/model_json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("Model_Save/model.h5")
loaded_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
loaded_model.summary()

textdata = []
datedata = []
datajson = []
class MyStreamListener(tweepy.StreamListener):
    def on_status(self, status):
        print(status.text)
    
    def on_error(self, status_code):
        print(status_code)
        
    def on_data(self, raw_data):
        datajson = json.loads(raw_data)
        try:
            textdata.append(datajson['text'])
            datedata.append(datajson['created_at'])
            print(len(textdata))
        except Exception as e:
            print(e)
        if len(textdata)>=100:
            return False

def predictions(inputtext):    
    word2index = imdb.get_word_index()
    X=[]
    revised = []
    review = re.sub('[^a-zA-Z]', ' ', inputtext)
    review = review.lower()
    for word in word_tokenize(review):
        if word in word2index:
            revised.append(word)
    for word in revised:  
            X.append(word2index[word])
    X=sequence.pad_sequences([X],maxlen=500)
    
    y_pred_lstm = loaded_model.predict(X)
    return(y_pred_lstm)

class livetweet():
    Stream = MyStreamListener()
    myStream = tweepy.Stream(auth = auth, listener = Stream)
    tracked = myStream.filter(track=['trump'])

    global dataframe
    dataframe = pd.DataFrame({
        'text':textdata,
        'created_at':datedata
        })
    def getSentiments(self):
        sentiments = []
        for each in dataframe['text']:
            sentiments.append(predictions(each))
            print(predictions(each))
        dates = []
        for date in dataframe['created_at']:
            milli = re.sub('[^0-9]', '', date)
            milli = milli[2:8]
            dates.append(milli)
        return (sentiments, dates)

class cursortweet():
    global tweets, date
    tweets = []
    date = []
    for tweet in tweepy.Cursor(api.search, q='microsoft').items(10):
        tweets.append(tweet.text)
        date.append(tweet.created_at)
    def getSentiments(self):
        sentiments = []
        for each in tweets:
            sentiments.append(predictions(each))
        return sentiments
    def getDates(self):
        dates = []
        for each in date:
            milli = re.sub('[^0-9]', '', str(each))
            milli = milli[2:8]
            dates.append(milli)
        return(dates)
