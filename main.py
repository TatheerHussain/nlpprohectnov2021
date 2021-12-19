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


data_x = post
data_y = data["Neutral (N) / Hostile (H)"]


data_dict = { 
    
    "Sentence": data_x,
    "Neutral (N) / Hostile (H)": data_y,   
}
data_new = pd.DataFrame(data_dict)

l = []
for i in data_y:
    a = i.split(",")
    for k in a:
        l.append(k)

        
print("Hate", l.count("H"))
print("Neutral",l.count("N"))
# print("offensive", l.count("offensive"))
# print("non-hostile", l.count("non-hostile"))
# print("defamation", l.count("defamation"))
# print("fake", l.count("fake"))




df_dict = {
    "post":data_x,
    "H":np.zeros(5000),
    "N":np.zeros(5000)
#     "non-hostile":np.zeros(5728),
#     "defamation":np.zeros(5728),
#     "fake":np.zeros(5728)
}




df_new = pd.DataFrame(df_dict)



data_x = post
data_y = data["Neutral (N) / Hostile (H)"]


# len(data_x)
# len(data_y)


for i in range(0,len(data_new["Neutral (N) / Hostile (H)"])):
    a = data_new["Neutral (N) / Hostile (H)"][i].split(",")
    for k in a:
        df_new[k][i] = 1


X = df_new["post"]
ini_array1 = np.array(df_new[df_new.columns[1:]])



import nltk 
nltk.download('punkt')



# TF count vectorizor 

cv = CountVectorizer(analyzer=stopword_removal,)

tfvetorizar  = cv.fit_transform(X)
# print(tfvetorizar.shape)


df_tfvetorizar= pd.DataFrame(tfvetorizar.toarray(), columns = cv.get_feature_names())

# TF-IDF count vectorizer 


tfidf = TfidfVectorizer(analyzer=stopword_removal)

tfidfvetorizar = tfidf.fit_transform(df_new["post"])

df_tfidfvetorizar = pd.DataFrame(tfidfvetorizar.toarray(), columns = tfidf.get_feature_names())



# splitting the data into test train 

X_train,X_test,Y_train,Y_test = train_test_split(df_tfidfvetorizar,ini_array1,test_size=0.33, random_state=88)



Y = ini_array1.ravel()
target_data=Y[0:len(Y_train)]


