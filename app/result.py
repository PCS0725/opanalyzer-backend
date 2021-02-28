import pandas as pd
from sklearn import metrics
from collections import Counter

class Result:

    def calMetrics(self, df):
        accuracy = metrics.accuracy_score(df['sentiment'], df['class'])*100

        #cm = metrics.confusion_matrix(y_test, y_pred)
        #check the official documentation for the meaning od the parameters
        met = metrics.precision_recall_fscore_support(df['sentiment'], df['class'], average = 'weighted')
        
        res = {
            'accuracy': accuracy,
            'precison': met[0]*100,
            'recall': met[1]*100,
            'F1-Score': met[2]*100
        }
        return res
    

    def calInsights(self, df):
        strPos = []
        strNeg = []
        strNeut = []
        totPos = 0
        totNeg = 0
        totNeut = 0
        tot = df.shape[0]

        for i in range(0, df.shape[0]):
            revClass = df.iloc[i]['class']

            if revClass == 'Positive':
                totPos += 1
                lst = df.iloc[i]['clean'].strip('][').split(', ')
                for word in lst:
                    word.replace("'", "")
                    strPos.append(word)
            
            if revClass == 'Neutral':
                totNeut += 1
                lst = df.iloc[i]['clean'].strip('][').split(', ')
                for word in lst:
                    word.replace("'", "")
                    strNeut.append(word)
            
            if revClass == 'Negative':
                totNeg += 1
                lst = df.iloc[i]['clean'].strip('][').split(', ')
                for word in lst:
                    word.replace("'", "")
                    strNeg.append(word)
        
        countPos = Counter(strPos)
        countNeg = Counter(strNeg)
        countNeut = Counter(strNeut)

        insights = {
            'total': tot,
            'total_pos': totPos,
            'total_neg': totNeg,
            'total_neut': totNeut,
            'top_pos': countPos.most_common(10),
            'tot_neg': countNeg.most_common(10),
            'tot_neut': countNeut.most_common(10) 
        }
        
        return insights
