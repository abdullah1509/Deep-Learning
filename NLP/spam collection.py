# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 10:42:14 2021

@author: Abdullah
"""

import pandas as pd
messages = pd.read_csv('SMSSpamCollection', sep='\t', names=["label", "message"])

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# text cleaning
ps= PorterStemmer()
corpus=[]

for i in range(0, len(messages)):
    review= re.sub('[^a-zA-Z0-9]', ' ', messages['message'][i])
    review= review.lower()
    review= review.split()
    
    review= [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review= ' '.join(review)
    corpus.append(review)


# creating bag of word
from sklearn.feature_extraction.text import CountVectorizer
# max_feature choose top 5000 most frequent words inste4ad of all 6296 unique words
cv= CountVectorizer(max_features=5000)
X= cv.fit_transform(corpus).toarray()

# converting message to dummy values
y= pd.get_dummies(messages['label'])
# taking only spam column
y= y.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=0)


# Training model using NaiveBayes classifier
from sklearn.naive_bayes import MultinomialNB
model= MultinomialNB()
model.fit(X_train, y_train)

# prediction
y_pred= model.predict(X_test)

# accuracy
from sklearn.metrics import accuracy_score
accuracy= accuracy_score(y_test, y_pred)

# confusion matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test, y_pred)