from textblob import TextBlob

class RbClassifier:

    def getPolarity(self, sentiment):
        if sentiment.polarity > 0:
            return 'Positive'
        elif sentiment.polarity == 0:
            return 'Neutral'
        else:
            return 'Negative'
    
    def classify(self, df):
        y_pred = []
        for i in range(0, df.shape[0]):
            rev = str(df.iloc[i]['review_body'])
            tb = TextBlob(rev)
            y_pred.append(self.getPolarity(tb.sentiment))
        
        return y_pred