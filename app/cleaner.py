import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from collections import Counter

from app.config import *


'''Data cleaning, tokenization and lemmatization
    @author: Prabhat Chand Sharma
    @Last modified: 28 Feb 2021'''
class Cleaner:
    def __init__(self):
        super().__init__()

    '''Removes punctuations, numbers, symbols, etc.
    @params: A single row from dataframe, dirty text
    @returns: Clean row'''
    def clean_dirt(self, text):
        text = text.lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\w*\d\w*', '', text)
        text = re.sub('[‘’“”…]', '', text)
        text = re.sub('\n', '', text)
        return text

    '''Tags the word with nltk tagger'''
    def tagWords(self, text):
        return pos_tag(text)


    '''Tags the words with wordnet tagger, to be used for lemmatization'''
    def newPos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            # As default pos in lemmatization is Noun
            return wordnet.NOUN


    def tagWordnet(self, text):
        tokens =[]
        for pair in text:
            tup = (pair[0], self.newPos(pair[1]))
            tokens.append(tup)
        return tokens


    '''Converts the word into its root form based on its POS tag'''
    def lemmaMaker(self, text):
        lemmatizer = WordNetLemmatizer()
        tokens =[]
        for pair in text:
            new = lemmatizer.lemmatize(pair[0], pair[1])
            tokens.append(new)
        return tokens


    '''For training purposes only'''
    def changeRating(self, rating):
        if(rating < 3 ):
            return 'Negative'
        if(rating == 3):
            return 'Neutral'
        if(rating > 3):
            return 'Positive'


    '''Find topmost words from a dataframe
        @params: df, the dataframe
        @returns: dictionary of top 'X' words by count'''
    def getTopWords(self, df):
        words = []
        for row in df.clean:
            for word in row:
                words.append(word)
        word_counts = Counter(words)
        topWords = word_counts.most_common(MAX_TOP_WORDS)
        return topWords


    '''Main function for cleaning the data'''
    def cleanData(self, d):

        #get the relevant columns and rename them
        df = d[['reviews.text', 'reviews.rating']].copy()
        df.rename(columns = {'reviews.text':'review_body'}, inplace = True) 
        df.rename(columns = {'reviews.rating':'rating'}, inplace = True) 

        df2 = pd.DataFrame([df['rating'], df['review_body']]).transpose()
        round1 = lambda x: self.clean_dirt(x)
        df['review_body'] = df['review_body'].apply(str)
        data_clean = pd.DataFrame(df.review_body.apply(round1))

        stop = stopwords.words('english')
        data_clean['clean'] = data_clean['review_body'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

        data_clean['clean'] = data_clean['clean'].apply(word_tokenize)

        data_clean['clean'] = data_clean.clean.apply(self.tagWords)
        data_clean['clean'] = data_clean.clean.apply(self.tagWordnet)

        data_clean['clean'] = data_clean.clean.apply(self.lemmaMaker)

        df2['rating'] = df2.rating.apply(self.changeRating)

        data_clean['sentiment'] = df2['rating']

        data_clean.to_csv(CLEAN_FNAME, index = False)

        return self.getTopWords(data_clean)




