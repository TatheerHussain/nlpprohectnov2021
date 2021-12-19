import pandas as pd
import numpy as np
import re
import emoji
import string
import csv
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


data = pd.read_csv(r'Hate Speech Roman Urdu.csv')
data.head()

data["Sentence"][0]

STOPWORDS = ['ai','ayi','hy','hai','main','ki','tha','koi','ko','sy','woh','bhi','aur',
             'wo','yeh','rha','hota','ho','ga','ka','le','lye','kr','kar','lye','liye',
             'hotay','aisey','gya','gaya','kch','ab','thy','thay','houn','hain','ho','jo',
             'han','to','is','hi','jo','kya','thi','se','pe','phr','phir','wala','waisay',
             'us','na','ny','hun','rha','raha','ja','rahay','abi','uski','ne','haan',
             'acha','nai','ney','ye','sent','photo','you','kafi','gai','rhy','kuch','jata','aye',
             'ya','dono','hoa','aese','de','wohi','jati','jb','krta','lg','rahi','hui',
             'karna','krna','gi','hova','yehi','jana','jye','chal','mil','tu','tum','hum','par',
             'hay','kis','sb','gy','dain','krny','tou']


def stopword_removal(new_post):
    tokens = word_tokenize(new_post)
    stopword_remove_row = [word for word in tokens if word not in STOPWORDS]
    return stopword_remove_row

def puncuation_removal(list):
    list2 = []
    for i in list:
        new_str = re.sub(r'[a-zA-Z0-9\n\',.#@_:…।?/|!$*-]', r'',i)
        list2.append(new_str)
    return list2


post = data["Sentence"]
#new_post = emojis_removal(post)
#post1 = puncuation_removal(new_post)
