#!/usr/bin/python

import numpy as np
import pandas as pd
import sklearn.preprocessing

from sklearn import cross_validation
from sklearn.feature_extraction.text import CountVectorizer

def compute1x2(str_in):
    str_in=str(str_in)
    result = 0
    if (str_in[0] > str_in[2]):
        result = -1
    elif (str_in[2] > str_in[0]):
        result = 1
    return result

cols = ['temporada','division','jornada','equipos','resultado','1X2']
datos_orig = pd.read_csv('datos_liga_2.csv', sep=',', header=None, names=cols, dtype='U')

datos_orig["1X2"] = map(compute1x2, datos_orig["resultado"])
datos_orig['equipos'] = datos_orig['equipos'].str.replace('.','')
datos_orig['equipos'] = datos_orig['equipos'].str.replace('(','')
datos_orig['equipos'] = datos_orig['equipos'].str.replace(')','')
datos_orig['equipos'] = datos_orig['equipos'].str.replace(' ','')

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(
    datos_orig['equipos'], datos_orig['1X2'], test_size=0.1, random_state=42)

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

print 'features_train_transformed x : ', len(features_train_transformed)
print 'features_train_transformed y : ', len(features_train_transformed[0])

from time import time
from sklearn import svm

#print 'svm instanciando'
clf = svm.SVC(kernel="rbf", verbose=True, C=1000)
#print 'svm fit'
t0 = time()
clf.fit(features_train_transformed, labels_train)
t0 = round(time() - t0, 3)
print "training time:", t0, "s ", t0/60, "min ", t0/3600, "h"

#print 'svm predict'
t1 = time()
pred = clf.predict(features_test_transformed)
t1 = round(time() - t1, 3)
print "prediction time:", t1, "s", t1/60, "min ", t1/3600, "h"

t2 = time()
print 'Score : ', clf.score(features_test_transformed, labels_test)
t2 = round(time() - t2, 3)
print "score time:", t2, "s", t2/60, "min ", t2/3600, "h"

#print datos_orig.loc[datos_orig['equipos'] == 'OviedoMallorca']
#print datos_orig.loc[datos_orig['equipos'] == 'ZaragozaOviedo']
#print datos_orig.loc[datos_orig['equipos'] == 'OviedoBarcelona']

print 'Predicciones ', u'OviedoMallorca', u'ZaragozaOviedo', u'OviedoBarcelona', clf.predict(
    vectorizer.transform([u'OviedoMallorca', u'ZaragozaOviedo', u'OviedoBarcelona']).toarray())

