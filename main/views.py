from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

#ml
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Create your views here.
fs = FileSystemStorage()
x=fs.open('x.csv')
x=pd.read_csv(x)
x=x['data'].tolist()
y=fs.open('y.csv')
y=pd.read_csv(y)
y_train = y.iloc[:, 1]
cv = CountVectorizer(max_features = 2500)
x = cv.fit_transform(x).toarray()
sc = StandardScaler()
x_train = sc.fit_transform(x)
model = RandomForestClassifier()
model.fit(x_train, y_train)

def pred(text):
    to_predict=[text]
    test_corpus = []
    review = re.sub('[^a-zA-Z]', ' ', to_predict[0])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    test_corpus.append(review)
    x_test = cv.transform(test_corpus).toarray()
    x_test = sc.transform(x_test)
    y_pred = model.predict(x_test)
    return y_pred

def home(request):
    return render(request, 'index.html')

def predict(request):
    if(request.method=='POST'):
        body=request.POST['body']
        res=pred(body)
        result=res[0]
        return render(request, 'result.html', {"result": result})

    return render(request, 'predict.html')