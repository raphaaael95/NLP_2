import pandas as pd
import os
import nltk
nltk.download('punkt')
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
nltk.download('averaged_perceptron_tagger')

os.chdir("C://Users/33652/Documents/cours_centrale/Second_semestre/nlp/NLP_2/exercise2/data") # Change working directory
os.getcwd()
pd.set_option('display.max_columns', 500)


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


all_words = []
documents = []


stop_words = list(set(stopwords.words('english')))

#  j is adject, r is adverb, and v is verb
#allowed_word_types = ["J","R","V"]
allowed_word_types = ["J"]



#files_lines = [open('train/pos/'+f, 'r').read() for f in files_pos]

comment=train_data["commentary"]
comment_list=list(comment)

all_words = []
documents = []
for p in  comment_list:
def clean_columns(p):    
    # create a list of tuples where the first element of each tuple is a review
    # the second element is the label
    #documents.append( (p, "pos") )
    
    # remove punctuations
    cleaned = re.sub(r'[^(a-zA-Z)\s]','', p)
    
    # tokenize 
    tokenized = word_tokenize(cleaned)
    
    # remove stopwords 
    stopped = [w for w in tokenized if not w in stop_words]
    
    # parts of speech tagging for each word 
    pos = nltk.pos_tag(stopped)
    
    # make a list of  all adjectives identified by the allowed word types list above
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())
            
            
train_data["commentary"].apply(clean_columns)
   

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


#################################Models######################################################
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from tqdm import tqdm_notebook as tqdm
tqdm().pandas()

y_train_set=train_data["label"]


f1_scorer = make_scorer(f1_score, average="weighted")
df_idf=pd.DataFrame(X_tfidf_sample.toarray())
X_train_set=pd.concat([train_data, df_idf], axis=1)
X_train_set.drop(['label', 'commentary'], axis=1)




# Set the parameters for grid_search
random_state = [None] #add more parameters here (default None)
max_depth = [4,6,8,12,50,None] #add more parameters here (default None)
tuned_parameters_rf = [{'random_state': random_state, 'max_depth':max_depth}]



grid_random_forest = GridSearchCV(
    RandomForestClassifier() , tuned_parameters_rf,cv=5,scoring=f1_scorer
  )
result_rf=grid_random_forest.fit(X_train_set,y_train_set)
print(result_rf.best_params_)
print(result_rf.best_score_)
