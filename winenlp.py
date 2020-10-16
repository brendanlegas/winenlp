#Import dependencies
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

#Read winemag csv
df = pd.read_csv('winemag-data_first150k.csv')

#Clean up dataset by dropping duplicates and nulls
clean = df[df.duplicated('description', keep=False)]
clean.dropna(subset=['description', 'points'])

#Create simplified df of just description and points
simple = clean[['description', 'points']]

#Transform method to create buckets for point ranges
def transform_points_simplified(points):
    if points < 82:
        return 1
    elif points >= 82 and points < 87:
        return 2 
    elif points >= 87 and points < 92:
        return 3 
    elif points >= 92 and points < 97:
        return 4 
    else:
        return 5

simple = simple.assign(bucket = simple['points'].apply(transform_points_simplified))

#Description Vectorization

X = simple['description']
y = simple['bucket']

vectorizer = CountVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)

#Splitting 80/20 train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=113)
#Using RandomForestClassifier Class to fit model to training data subset
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

#Testing the model
predictions = rfc.predict(X_test)
print(classification_report(y_test, predictions))

from sklearn.externals import joblib
joblib.dump(rfc, 'wine_nlp_mod.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')