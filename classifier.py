import pandas as pd
import numpy as np
import os
import nltk
#nltk.download('stopwords')
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
import re
from nltk.corpus import stopwords
import spacy
from spacy import displacy
from sklearn.model_selection import train_test_split

import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
LabeledSentence = gensim.models.doc2vec.TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer

from tqdm import tqdm
from nltk.tokenize import  word_tokenize
tqdm.pandas(desc="progress-bar")


nlp = spacy.load("en_core_web_sm")

nltk.download('averaged_perceptron_tagger')

os.chdir("C://Users/arimo/OneDrive/Documents/Ariel/Education/ESSEC Centrale/Cours/CentraleSupelec/Elective Classes/NLP/Assignments/Assignment2/Ressources/exercise2/data") # Directory de Ariel
#os.chdir("C://Users/33652/Documents/cours_centrale/Second_semestre/nlp/NLP_2/exercise2/data") # Directory de Raphaël
#os.chdir("/Users/michaelallouche/Google Drive/Ecoles/CentraleSupelec/Data Science Electives/Natural Language Processing/Assignment 2/exercise2/data") # Directory de Michaël
os.getcwd()

pd.set_option('display.max_colwidth', -1)

train_data=pd.read_csv("traindata.csv", sep='\t', header=None)# error_bad_lines=False
dev_data=pd.read_csv("devdata.csv", sep='\t', header=None)

train_data.head(10)
dev_data.head()
train_data=train_data.rename(columns={0: 'label',1:'category',2:'subject',3:'index',4:'commentary'})

class Classifier:
    """The Classifier"""


    #############################################
    def train(self, trainfile):
        """Trains the classifier model on the training set stored in file trainfile"""


    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """

############################ FIRST STEP ####################################################
#Separate the column category into two parts+ maybe one hot encoding
# For the column subject do a one hot encoding
        
new = train_data["category"].str.split("#", n = 1, expand = True) 
                           
train_data["category"]= new[0] 
train_data["subcategory"]= new[1] 

train_data = train_data[["label","category", "subcategory", "subject", "index", "commentary"]]

train_data.head()


#Separate the column index into two parts
        
new = train_data["index"].str.split(":", n = 1, expand = True) 
                           
train_data["index_b"]= new[0] 
train_data["index_e"]= new[1] 

train_data = train_data[["label","category", "subcategory", "subject", "index_b","index_e", "commentary"]]

train_data["index_b"]=train_data["index_b"].apply(int)
train_data["index_e"]=train_data["index_e"].apply(int)

train_data.head()

############################ SECOND STEP ####################################################
#Pre-processing : cleaning
        

def clean_columns(string):    
    # create a list of tuples where the first element of each tuple is a review
    # the second element is the label
    # documents.append( (p, "pos") )
    
    # put everything in lowercases
    string = string.lower()
    
    # remove punctuations
    string = re.sub("([^\w]|[\d_])+", " ",  string)
    
    
    
    ##https://spacy.io/usage/linguistic-features 
    # Testing chunk , try to remove verbs
    # Apply spacy on the sentence 
    
    #tokenized_string = nlp(string)

    #displacy.serve(tokenized_string, style="dep")
    #for token in tokenized_string:
    #    print(token.pos_)
    #    print(token)
    #    print(type(token.pos_))
    tokenized_string=word_tokenize(string)

    #for chunk in tokenized_string.noun_chunks:
        #print(chunk.text)
    
    
    # remove stopwords 
    #stopped = [w for w in tokenized if not w in stop_words]
    
    # parts of speech tagging for each word 
    #pos = nltk.pos_tag(stopped)
    
    # make a list of  all adjectives identified by the allowed word types list above
    #for w in pos:
        #if w[1][0] in allowed_word_types:
            #all_words.append(w[0].lower())
    
    return tokenized_string

list_columns = ['commentary']        

#train_data['commentary'][:10].apply(clean_columns) 


def clean_column(list_columns):
    for name in list_columns:
        print(name)
        train_data[name] = train_data[name].apply(clean_columns) 
        
    

clean_column(list_columns)

train_data.subject = train_data.subject.apply(lambda x: x.lower())

train_data.head()



#train_data["subject"].nunique()

train_data.label.replace(["negative","neutral", "positive"], [0,1,2], inplace=True)



################################# Word2Vec ######################################################

#parameters
n_dim=200
size_corpus=1503
min_count = 1
n_epochs = 50
min_df = 10


def labelizeCommentary(X, label_type):
    labelized = []
    for i,v in tqdm(enumerate(X)):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized


X_train_tagged = labelizeCommentary(train_data.commentary, 'TRAIN')
#X_test_tagged = labelizeCommentary(X_test_dl, 'TEST')

word_w2v = Word2Vec(size = n_dim, min_count = min_count)
word_w2v.build_vocab([x.words for x in tqdm(X_train_tagged)])
word_w2v.train([x.words for x in tqdm(X_train_tagged)], epochs = n_epochs, total_examples = size_corpus)


vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df = min_df)
matrix = vectorizer.fit_transform([x.words for x in X_train_tagged])
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))


def buildWordVector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += word_w2v[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec


from sklearn.preprocessing import scale
train_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, X_train_tagged))])
train_vecs_w2v = scale(train_vecs_w2v)

train_vecs_w2v = pd.DataFrame(train_vecs_w2v)

#test_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, X_test_tagged))])
#test_vecs_w2v = scale(test_vecs_w2v)



################################# One hot encoding ######################################################

from sklearn.preprocessing import OneHotEncoder
# creating instance of one-hot-encoder

enc = OneHotEncoder(handle_unknown='ignore')



# passing bridge-types-cat column (label encoded values of bridge_types)
enc_df = pd.DataFrame(enc.fit_transform(train_data[['category', 'subcategory']]).toarray())

enc.get_feature_names(['category', 'subcategory'])

enc_df.columns = enc.get_feature_names(['category', 'subcategory'])

# merge with main df bridge_df on key values

train_vecs_w2v = pd.concat([enc_df,train_vecs_w2v],axis=1)

train_vecs_w2v = pd.concat([train_data[["index_b","index_e"]],train_vecs_w2v],axis=1)
#train_vecs_w2v["subject"] = train_data.subject

train_vecs_w2v.head()



############################ Tfidf matrice ####################################################

# Write code to import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Write code to create a TfidfVectorizer object
tfidf = TfidfVectorizer()

# Write code to vectorize the sample text
X_tfidf_sample = tfidf.fit_transform(train_data["commentary"])

print("Shape of the TF-IDF Matrix:")
print(X_tfidf_sample.shape)
print("TF-IDF Matrix:")
print(X_tfidf_sample.todense())
print(tfidf.get_feature_names())

y_train_set=train_data["label"]

df_idf=pd.DataFrame(X_tfidf_sample.toarray())

X_train_set=pd.concat([train_data, df_idf], axis=1)

X_train_set.drop(['label', 'commentary'], axis=1, inplace = True)


X_train_set.head()





#################################Models######################################################

from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer



from sklearn.metrics import accuracy_score


accuracy_scorer = make_scorer(accuracy_score)

#Random Forest
# Set the parameters for grid_search
random_state = [42] #add more parameters here (default None)
max_depth = [4,6,8,12,50,None] #add more parameters here (default None)
tuned_parameters_rf = [{'random_state': random_state, 'max_depth':max_depth}]



grid_random_forest = GridSearchCV(
    RandomForestClassifier() , tuned_parameters_rf,cv=5,scoring=accuracy_scorer
  )



result_rf=grid_random_forest.fit(train_vecs_w2v,train_data.label)
print(result_rf.best_params_)
print(result_rf.best_score_)

# Naive bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB,BernoulliNB


alpha=[1.0] #add more parameters here (default 1.0)
fit_prior=[True] #add more parameters here (default True)
class_prior=[None] #add more parameters here (default None)
tuned_parameters_nb = [{'alpha': alpha, 'fit_prior':fit_prior,'class_prior':class_prior}]


grid_Naive_Bayes_multi = GridSearchCV(
    MultinomialNB() , tuned_parameters_nb,cv=5,scoring=accuracy_scorer
  )



result_nb_multi=grid_Naive_Bayes_multi.fit(X_train_set,y_train_set)
print(result_nb_multi.best_params_)
print(result_nb_multi.best_score_)



#XGboost
# Set the parameters for grid_search
import xgboost as xgb
# Set the parameters for grid_search
parameters = {
    'eta': [0.2,0.1,0.05], 
    'max_depth': [8], 
    "n_estimators" : [1000],
    'objective': ['multi:softprob'], 
    'num_classes' : [3], 
    'nthread' : [-1],
    'subsample' : [0.8],
    'colsample_bytree' : [0.8],
    'colsample_bylevel' : [1], 
 
    } 


grid_xgb = GridSearchCV(xgb.XGBClassifier(), 
                           parameters, 
                           cv = 3, 
                           scoring = accuracy_scorer, 
                               verbose = 10, 
                           n_jobs = -1)


result_xgb = grid_xgb.fit(train_vecs_w2v,train_data.label)

print(result_xgb.best_params_)
print(result_xgb.best_score_)

#Keras linear NN
X_train_dl, X_test_dl, y_train_dl, y_test_dl = train_test_split(train_vecs_w2v,
                                                   train_data.label, test_size=0.2)

from keras.models import Sequential
from keras.layers import Dense

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

input_dim = 213
num_epochs = 30 
batch_size = 32
verbose = 10

model = Sequential()

model.add(Dense(32, activation = 'relu', input_dim = input_dim))

model.add(Dense(32, activation = 'relu'))
model.add(Dense(3, activation = 'sigmoid'))

model.summary()

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

results = model.fit(X_train_dl, y_train_dl, 
          epochs = num_epochs, 
          batch_size = batch_size, 
          verbose = verbose, 
          workers = 4)

score = model.evaluate(X_test_dl, y_test_dl, batch_size = 32, verbose = verbose, validation_split=0.2)
print(score[1])
     
#display loss and accuracy curves
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()






