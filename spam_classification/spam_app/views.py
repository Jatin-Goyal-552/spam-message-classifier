from django.shortcuts import render
import joblib
from nltk.corpus import stopwords
import string
import numpy as np 
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

df=pd.read_csv("archive/spam.csv",encoding='latin-1')
# print(df.head(5)) 
def remove_stopwords(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
from sklearn.pipeline import Pipeline

pipeline2 = Pipeline([
    ('count', TfidfVectorizer(analyzer=remove_stopwords)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB()),
])
pipeline2.fit(df['v2'],df['v1'])
def home(request):
    prediction=""
    if request.method=='POST':
        spam=request.POST.get('spam')
        prediction=pipeline2.predict([spam])[0]
        print(prediction)
        if prediction=='ham':
            prediction="Not a spam text"
        else:
            prediction="Spam text"
    context={
        'prediction':prediction
        }
    return render(request, 'home.html',context)