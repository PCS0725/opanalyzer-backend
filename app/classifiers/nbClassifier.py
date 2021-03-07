import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
import pickle

from app.config import *

class NbClassifier:

    def scaleData(self, df):
        df.fillna(0, inplace = True)
        scaler = MinMaxScaler()
        
        dfArray = df.values
        dfScaled = scaler.fit_transform(dfArray)

        return pd.DataFrame(dfScaled)


    '''Returns the predicted columm only, no metrics'''
    def classify(self, df):
        #dependent and independents
        X = df.drop('sentiment', axis = 1)
        #y = df['sentiment']

        x = self.scaleData(X)

        #training and predicting will be different scenarios, take care
        #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=5)

        #model = MultinomialNB()
        #model.fit(x_train, y_train)

        #padding extra columns to match with the trained model
        for i in range(0, 2116 - x.shape[1]):
            x[str(i)] = pd.Series([0 for x in range(len(df.index))], index=x.index)

        model = pickle.load(open(NB_MODEL, 'rb'))
        y_pred = model.predict(x)
        
        return y_pred