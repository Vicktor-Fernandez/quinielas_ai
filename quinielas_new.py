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
datos_orig['equipos'] = datos_orig['equipos'].str.replace('.','')
datos_orig['equipos'] = datos_orig['equipos'].str.replace('(','')
datos_orig['equipos'] = datos_orig['equipos'].str.replace(')','')
datos_orig['equipos'] = datos_orig['equipos'].str.replace(' ','')

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(
    datos_orig['equipos'], datos_orig['1X2'], test_size=0.01, random_state=2)

vectorizer = CountVectorizer(strip_accents='unicode')
features_train_transformed = vectorizer.fit_transform(features_train).toarray()
features_test_transformed  = vectorizer.transform(features_test).toarray()

##print 'features_train'
##print features_train
##print 'features_train_transformed'
##print features_train_transformed
##print 'labels_train'
##print labels_train

#print 'vectorizer.get_feature_names()'
#print vectorizer.get_feature_names()

from sklearn.naive_bayes import GaussianNB

clf=GaussianNB()
clf.fit(features_train_transformed, labels_train)

pred = clf.predict(features_test_transformed)
print 'Score : ', clf.score(features_test_transformed, labels_test)

print datos_orig.loc[datos_orig['equipos'] == 'OviedoMallorca']
print datos_orig.loc[datos_orig['equipos'] == 'ZaragozaOviedo']
print datos_orig.loc[datos_orig['equipos'] == 'BarcelonaOviedo']

print 'Predicciones ', clf.predict(vectorizer.transform(
    [u'OviedoMallorca', u'ZaragozaOviedo', u'BarcelonaOviedo']).toarray())
