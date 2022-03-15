import sys

import re
import pandas as pd
from sqlalchemy import create_engine
import joblib

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet'])

from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score


def load_data(database_filepath):
    '''
    Laoding data from SQL database and splits it into Datasets of predictor and criterion and a list of the categorie names.
    
    Args:
    database_filepath (str): database filepath
    
    Returns: 
    X (DataFrame):
    Y (DataFrame):
    category_names (list of str): 
    
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('MessagesAndCategories', engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns

    return X,Y, category_names

def tokenize(text, url_place_holder_string='url_place_holder_string'):
    '''Tokenize text and replacing all urls.
    
    Args:
    text (str): Text that should be tokenized
    url_place_holder_string (str, default:'url_place_holder_string'): The string all URLs are changed to. 
    
    Retruns:
    clean_tokens (list of str): List of tokens of the given text.
    '''
    # Replace all urls with a urlplaceholder string
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Extract all the urls from the provided text 
    detected_urls = re.findall(url_regex, text)
    
    # Replace url with a url placeholder string
    for detected_url in detected_urls:
        text = text.replace(detected_url, url_place_holder_string)

    # Extract the word tokens from the provided text
    tokens = nltk.word_tokenize(text)
    
    #Lemmanitizer to remove inflectional and derivationally related forms of a word
    lemmatizer = nltk.WordNetLemmatizer()

    # List of clean tokens
    clean_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]

    return clean_tokens

def build_model():
    '''
    Build a GridSearch Model with a defined pipeline and parameters. 
    The Pipeline is compiled by a CountVectorizer using tokenize, TFidfTransformer and a MultiOutputClassifier with RandomForrestClassifier as Estimator.

    Args:
    None

    Returns:
    cv (sklear GridSearchCV): created gridsearch object
    '''

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'clf__estimator__min_samples_split': [2, 3, 4]  
    }

    cv = GridSearchCV(pipeline, param_grid = parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate a model with multiclass-multioutput with classification reports and accurancies of each categorie.
    Results are printed in the console.

    Args:
    model (sklearn GridSearch, MultiPutputClassfier, Pipeline): The Multiclass-multioutput Model which is tested
    X_test (DataFrame): A pandas DataFrame with the test data
    Y_test (DataFrame): A pandas DataFrame with the test results to check the model against. 
    category_names (list of str): List of the category names of the categories which should be evaluated. 
    '''

    Y_prediction_test = model.predict(X_test)

    for category_name in category_names:
        print('Category: {} '.format(category_name))
        print(classification_report(Y_test[category_name].values, Y_prediction_test[:,list(category_names).index(category_name)]))
        print('Accuracy {}\n\n'.format(accuracy_score(Y_test[category_name].values, Y_prediction_test[:,list(category_names).index(category_name)])))


def save_model(model, model_filepath):
    '''
    Saving model.

    Args: 
    model (sklearn GridSearch, MultiPutputClassfier, Pipeline): Model to be safed.
    model_filepath (str): Path and file name with file extension where and how the model is safed (pickle files are recommended)  
    '''
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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