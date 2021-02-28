import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from app.config import *


class Ngrams:
    
    def getStringsArray(self, df):
        strings = []
        for row in df.clean:
            string = ''
            rowNew = row.strip('][').split(', ') 
            for word in rowNew:
                string += word.replace("'", "")
                string += ' '
            strings.append(string)
            string = ''
        return strings


    def getEmbeddings(self, df):
        strings = self.getStringsArray(df)

        tfidfVectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df = TF_IDF_MINDF)
        embedsList = tfidfVectorizer.fit_transform(strings).toarray()

        embedsDf = pd.DataFrame(embedsList)
        embedsDf['sentiment'] = df['sentiment']

        #embedsDf.to_csv(FEATURES_NGRAMS_FNAME, index = False)

        return embedsDf



