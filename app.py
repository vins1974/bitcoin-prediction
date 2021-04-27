#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot
import cufflinks
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl', offline=True)
import re
import random
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# In[2]:


def clean_tweet(tweet):
    return ' '.join(re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)', ' ', tweet).split())


# In[3]:


#removes pattern in the input text
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for word in r:
        input_txt = re.sub(word, "", input_txt)
    return input_txt.lower()


# In[4]:


def analyze_sentiment(tweet):
    analysis=TextBlob(tweet)
    if analysis.sentiment.polarity>0:
        return "positive"
    elif analysis.sentiment.polarity==0:
        return "neutral"
    else:
        return "negative"


# In[5]:


def getpolarity(text):
    return TextBlob(text).sentiment.polarity
def getsubjectivity(text):
    return TextBlob(text).sentiment.subjectivity


# In[6]:


twibit = pd.read_csv('datasets/Final_Dataset.csv')


# In[7]:


from flask import Flask, request, jsonify, render_template, url_for
from flask_cors import cross_origin
import pickle


# In[8]:


app = Flask(__name__) #to create flask App
model = pickle.load(open('model.pkl', 'rb'))


# In[9]:


@app.route('/')
@cross_origin()
def home():
    return render_template('index.html')


# In[10]:


@app.route('/predict',methods=['GET','POST'])
@cross_origin()

def predict():
    if request.method == 'POST':
        user = request.form["TweetonBitcoin"]
        data = {'o_t':  [user]}
        df = pd.DataFrame (data, columns = ['o_t'])
        
        #removing the twitter handles @user
        df['clean_tweet'] = np.vectorize(remove_pattern)(df['o_t'], "@[\w]*")

        #using above functions
        df['clean_tweet'] = df['clean_tweet'].apply(lambda x : clean_tweet(x))

        #removing special characters, numbers and punctuations
        df['clean_tweet'] = df['clean_tweet'].str.replace("[^a-zA-Z]", " ")


        #remove short words
        df['clean_tweet'] = df['clean_tweet'].apply(lambda x: " ".join([w for w in x.split() if len(w)>3]))

        # Removing every thing other than text
        df['clean_tweet'] = df['clean_tweet'].apply( lambda x: re.sub(r'[^\w\s]',' ',x))  # Replacing Punctuations with space
        df['clean_tweet'] = df['clean_tweet'].apply( lambda x: re.sub(r'[^a-zA-Z]', ' ', x)) # Raplacing all the things with space other than text
        df['clean_tweet'] = df['clean_tweet'].apply( lambda x: re.sub(r"\s+"," ", x)) # Removing extra spaces

        
        # for performing NLP Functions i.e detection of Polarity and Subjectivity

        polarity=[]     #list that contains polarity of tweets
        subjectivity=[]    ##list that contains subjectivity of tweets

        for i in df.clean_tweet.values:
            try:
                analysis = TextBlob(i) # [i] records to the first data in dataset
                polarity.append(analysis.sentiment.polarity)
                subjectivity.append(analysis.sentiment.subjectivity)
            except:
                polarity.append(0)
                subjectivity.append(0)


        # adding sentiment polarity and subjectivity column to dataframe

        df['subjectivity'] = subjectivity
        df['polarity'] = polarity
        


        #Create a function to get the sentiment scores (using Sentiment Intensity Analyzer)
        
        def getSentiment(score):
            if score < 0:
                return 0 #indicates negative
            elif score == 0:
                return 1  #indicates neutral
            else:
                return 2 #indicates positived
            
        df['sentiment_score'] = df['polarity'].apply(getSentiment)
        
        list1 = []
        for i in range(twibit.shape[0]):
            list1.append(i)
        random_index = random.choice(list1)
        
        r=[]
        
        r.append(twibit.open[random_index])
        r.append(twibit.high[random_index])
        r.append(twibit.low[random_index])
        r.append(twibit.close[random_index])
        r.append(twibit.volume[random_index])

        r.append(df.polarity[0])
        r.append(df.subjectivity[0])
        r.append(df.sentiment_score[0])

        final_features = [np.array(r)]

        prediction = model.predict(final_features)

        output = int(prediction[0]) #1 stands for price up; 0 stands for price down
        
        if df['polarity'][0] > 0:   # Positive Sentiment
            if output == 1:
                prediction_text1='Your Tweet is {}'.format(user)
                prediction_text2='The Tweet sentiment is POSITIVE as it\'s polarity is {}'.format(df['polarity'][0])
                prediction_text3='Price will go up so it\'s a good time to sell.'
                combine = prediction_text1 + ' \n ' + prediction_text2 + ' \n ' + prediction_text3
                return render_template('index.html', prediction_text = combine)
            else:
                prediction_text1='Your Tweet is {}'.format(user)
                prediction_text2='The Tweet sentiment is POSITIVE as it\'s polarity is  {}'.format(df['polarity'][0])
                prediction_text3='Price will go down so it\'s a good time to buy more'
                combine = prediction_text1 + ' \n ' + prediction_text2 + ' \n ' + prediction_text3
                return render_template('index.html', prediction_text = combine)
            
        elif df['polarity'][0] < 0:  # Negative Sentiment
            if output == 1:
                prediction_text1='Your Tweet is {}'.format(user)
                prediction_text2='The Tweet sentiment is NEGATIVE as it\'s polarity is {}'.format(df['polarity'][0])
                prediction_text3='Price will go up so it\'s a good time to sell.'
                combine = prediction_text1 + ' \n ' + prediction_text2 + ' \n ' + prediction_text3
                return render_template('index.html', prediction_text = combine)
            else:
                prediction_text1='Your Tweet is {}'.format(user)
                prediction_text2='The Tweet sentiment is POSITIVE as it\'s polarity is  {}'.format(df['polarity'][0])
                prediction_text3='Price will go down so it\'s a good time to buy more'
                combine = prediction_text1 + ' \n ' + prediction_text2 + ' \n ' + prediction_text3
                return render_template('index.html', prediction_text = combine)
        else:    # Neutral Sentiment
            if output == 1:
                prediction_text1='Your Tweet is {}'.format(user)
                prediction_text2='The Tweet sentiment is NEUTRAL as it\'s polarity is {}'.format(df['polarity'][0])
                prediction_text3='Price will go up so it\'s a good time to sell.'
                combine = prediction_text1 + ' \n ' + prediction_text2 + ' \n ' + prediction_text3
                return render_template('index.html', prediction_text = combine)
            else:
                prediction_text1='Your Tweet is {}'.format(user)
                prediction_text2='The Tweet sentiment is NEUTRAL as it\'s polarity is  {}'.format(df['polarity'][0])
                prediction_text3='Price will go down so it\'s a good time to buy more'
                combine = prediction_text1 + ' \n ' + prediction_text2 + ' \n ' + prediction_text3
                return render_template('index.html', prediction_text = combine)     
            
    return render_template(index.html)


# In[11]:


if __name__ == "__main__":
    app.run(debug=False)


# In[12]:


#pip freeze > requirements.txt


# In[ ]:




