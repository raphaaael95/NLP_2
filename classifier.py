import pandas as pd
import numpy as np
import os
import nltk
nltk.download('stopwords')
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

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")


nlp = spacy.load("en_core_web_sm")

nltk.download('averaged_perceptron_tagger')

#os.chdir("C://Users/arimo/OneDrive/Documents/Ariel/Education/ESSEC Centrale/Cours/CentraleSupelec/Elective Classes/NLP/Assignments/Assignment2/Ressources/exercise2/data") # Directory de Ariel
#os.chdir("C://Users/33652/Documents/cours_centrale/Second_semestre/nlp/NLP_2/exercise2/data") # Directory de Raphaël
os.chdir("/Users/michaelallouche/Google Drive/Ecoles/CentraleSupelec/Data Science Electives/Natural Language Processing/Assignment 2/exercise2/data") # Directory de Michaël
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
    tokenized_string = nlp(string)
    #displacy.serve(tokenized_string, style="dep")
    for token in tokenized_string:
        print(token.pos_)
        print(token)
        print(type(token.pos_))
    
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

list_columns = ['category', 'subcategory', 'subject', 'commentary']        

train_data['commentary'][:10].apply(clean_columns) 

def clean_column(list_columns):
    for name in list_columns:
        print(name)
        train_data[name] = train_data[name].apply(clean_columns) 

clean_column(list_columns)
train_data.head()

train_data["subject"].nunique()

train_data.label.replace(["negative","neutral", "positive"], [-1,0,1], inplace=True)



################################# Word2Vec ######################################################


X_train_dl, X_test_dl, y_train_dl, y_test_dl = train_test_split(train_data.commentary,
                                                    train_data.label, test_size=0.2)

def labelizeCommentary(X_train_dl, label_type):
    labelized = []
    for i,v in tqdm(enumerate(X_train_dl)):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized

labelizeCommentary(X_train_dl, 'TRAIN')
labelizeCommentary(X_test_dl, 'TEST')


































################################# One hot encoding ######################################################

from sklearn.preprocessing import OneHotEncoder
# creating instance of one-hot-encoder

enc = OneHotEncoder(handle_unknown='ignore')

# passing bridge-types-cat column (label encoded values of bridge_types)
enc_df = pd.DataFrame(enc.fit_transform(train_data[['category', 'subcategory']]).toarray())

# merge with main df bridge_df on key values

train_data = train_data.join(enc_df)

train_data.head()
train_data.drop(['category','subcategory','subject'],axis = 1, inplace=True)
train_data.head()

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
random_state = [None] #add more parameters here (default None)
max_depth = [4,6,8,12,50,None] #add more parameters here (default None)
tuned_parameters_rf = [{'random_state': random_state, 'max_depth':max_depth}]



grid_random_forest = GridSearchCV(
    RandomForestClassifier() , tuned_parameters_rf,cv=5,scoring=accuracy_scorer
  )



result_rf=grid_random_forest.fit(X_train_set,y_train_set)
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

