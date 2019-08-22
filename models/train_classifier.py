import sys
import numpy as np
import pandas as pd
import re
import sqlalchemy 
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('stopwords')
nltk.download('wordnet')
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import pickle


def load_data(database_filepath):
    '''
    input: (
        database_filepath: path to database
            )
    Loads data from sqlite database 
    output: (
        X: features dataframe
        y: target dataframe
        category_names: list of target names
        )
    '''
    engine = sqlalchemy.create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster', engine) 
    X = df.loc[:,'message']
    y = df.iloc[:,4:]
    category_names = list(y.columns.values)
    return X, y, category_names
    


def tokenize(text):
    '''
    input: (
        text: raw text data
            )
    output: (
        returns cleaned tokens in list 
            )
    Function normalizes, tokenizes, and lemmatizes the text and
    removes stopwords.
    '''
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # tokenize text
    tokens = word_tokenize(text)
    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens

def build_model():
    '''
    Creates pipeline for the model with the best parameters
    discovered by GridSearch. 
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, ngram_range=(1,2))),
        ('tfidf', TfidfTransformer(norm='l2', sublinear_tf=False, smooth_idf=False)),
        ('clf', MultiOutputClassifier(RandomForestClassifier( n_estimators=250,\
            min_samples_split=2, max_features='log2', n_jobs=-1)))
    ])
    
    return pipeline
    
def evaluate_model(model, X_test, y_test, category_names):
    '''
    Input: (
        model: a pipeline as defined by build_model(),
        X_test: values of a dataframe defined by train_test_split() below,
        y_test: values of a dataframe,
        category names: list defined in load_data() function
        )
    Output: (
        y_pred: predicted values for X_test,
        a confusion matrix of the results
        )
        
    '''
    y_pred = model.predict(X_test)
    for i, label in enumerate(category_names):
        print(label)
        print(confusion_matrix(y_pred[:,i] ,y_test.values[:,i]))


def save_model(model, model_filepath):
    '''
    input: (
        model: trained model 
        model_filepath: filepath to save model in flattened, serialized form 
            )
    Saves the model to a Python pickle file
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()