from app.result import Result
import pandas as pd
import requests
from flask import jsonify

from app.cleaner import Cleaner
from app.config import *
from app.embeddings.ngrams import Ngrams
from app.classifiers.nbClassifier import NbClassifier
from app.classifiers.rbClassifier import RbClassifier
from app.result import Result

'''Handler class for all the requests
   This class decides what happens with any request'''
class Handler:
    def __init__(self):
        super().__init__()
    


    '''Handler function for clean data request
        @params: url of the csv file
        @returns: dictionary of top words by count, to be used in EDA'''
    def cleanRequest(self, url):
        #req = requests.get(url)
        df = pd.read_csv(REV_DATA_FILE)
        cleaner = Cleaner()
        topWords = cleaner.cleanData(df)
        edaRes = {
            'res-type': 'EDA',
            'topWords': topWords,
        }
        return jsonify(edaRes)


    def classifyRequest(self, choices):
        df = pd.read_csv(CLEAN_FNAME)
        df_embeddings = pd.DataFrame()
        y_pred = pd.DataFrame()
        embedding_algo = choices['feature']
        classifier = choices['classifier']

        if embedding_algo == 'ngrams':
            ngram = Ngrams()
            df_embeddings = ngram.getEmbeddings(df)
        elif embedding_algo == 'w2vec':
            pass             

        if classifier == 'nb':
            nbClassifier = NbClassifier()
            y_pred = nbClassifier.classify(df_embeddings)
        elif classifier == 'rb':
            rbClassifier = RbClassifier()
            y_pred = rbClassifier.classify(df)

        df['class'] = y_pred
        result = Result()
        metrics = result.calMetrics(df)
        insights = result.calInsights(df)

        response = {
            'embedding_algo': embedding_algo,
            'classifier': classifier,
            'metrics' : metrics,
            'insights' : insights
        }

        return jsonify(response)
