from app.config import WORD2VEC_MODEL
import pandas as pd
import gensim
from gensim.models import Word2Vec
import pickle


class Word2Vecs:


    def getEmbeddings(self, df):
        glove_vectors = pickle.load(open(WORD2VEC_MODEL, 'rb'))

        X = []
        for row in df.clean:
            sent = []
            lst = row.strip('][').split(', ')
            for word in lst:
                wd = word.replace("'", "")
                if(wd in glove_vectors.vocab):
                    vec = glove_vectors[wd]
                    sent.append(round(sum(vec)/len(vec), 5))
                else:
                    sent.append(0)
            X.append(sent)

        embeds_df = pd.DataFrame(X)

        embeds_df['sentiment'] = df['sentiment']

        return embeds_df        
