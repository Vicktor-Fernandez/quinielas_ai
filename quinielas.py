#!/usr/bin/python

import numpy as np
import pandas as pd
import sklearn.preprocessing

from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif

def compute1x2(str_in):
    str_in=str(str_in)
    result = 0
    if (str_in[0] > str_in[2]):
        result = 1
    elif (str_in[2] > str_in[0]):
        result = 2
    return result

cols = ['temporada','division','jornada','equipos','resultado','1X2']
datos_orig = pd.read_csv('datos_liga_2.csv', sep=',', header=None, names=cols, dtype='U')

datos_orig["1X2"] = map(compute1x2, datos_orig["resultado"])

datos_orig = datos_orig.iloc[:,[3,5]]

datos_train, datos_test = cross_validation.train_test_split(datos_orig, test_size=0.9, random_state=2)

print datos_train.head()

dicc_train = {}
for x in range(len(datos_train)):
    currentid = datos_train.iloc[x,0]
    currentvalue = datos_train.iloc[x,1]
    dicc_train.setdefault(currentid, [])
    dicc_train[currentid].append(currentvalue)

for x in range(10):
    print dicc_train.keys()[x], ' ', dicc_train[dicc_train.keys()[x]]

### test_size is the percentage of events assigned to the test set
### (remainder go into training)
#features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(
#    datos_orig['equipos'].values.astype('U'), datos_orig['1X2'].values.astype('U'), test_size=0.9999, random_state=42)
#features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(
#    datos_orig['equipos'], datos_orig['1X2'], test_size=0.9999, random_state=42)

### text vectorization--go from strings to lists of numbers
#vectorizer = TfidfVectorizer(sublinear_tf=True, decode_error='ignore')
#features_train_transformed = vectorizer.fit_transform(features_train).toarray()
#features_test_transformed  = vectorizer.transform(features_test).toarray()

#print features_test_transformed, ' ', labels_test

#vectorizer = CountVectorizer(decode_error='ignore')
#features_train_transformed = vectorizer.fit_transform(features_train).toarray()
#features_test_transformed  = vectorizer.transform(features_test).toarray()

dicc_test = {}

vectorizer = DictVectorizer(sparse=False)
dict_train_transformed = vectorizer.fit_transform(dicc_train)
dict_test_transformed  = vectorizer.transform(dicc_test)

#from sklearn_pandas import DataFrameMapper, cross_val_score

#mapper = DataFrameMapper([('equipos', sklearn.preprocessing.LabelBinarizer()),
#                          ('1X2', sklearn.preprocessing.LabelBinarizer())])

#print mapper.fit_transform(datos_orig.copy())

print vectorizer.get_feature_names()
print dict_train_transformed

from sklearn.naive_bayes import GaussianNB

clf=GaussianNB()
clf.fit(features_train_transformed, labels_train)

pred = clf.predict(features_test_transformed)
print 'Score : ', clf.score(features_test_transformed, labels_test)


print 'Predicciones ', clf.predict(vectorizer.transform([u'Oviedo           Mallorca', u'Zaragoza         Oviedo']).toarray())


