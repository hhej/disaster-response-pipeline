import re
import sys
import nltk
import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+' 

def load_data(database_filepath):
    """loads X and Y and gets category names
    Args:
        database_filepath (str): string filepath of the sqlite database
    Returns:
        X (pandas dataframe): messages
        Y (pandas dataframe): labels
        category_names (list): list of the category names for classification
    """
    # load the database into a pandas dataframe
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('labeled_messages', engine)
    engine.dispose()
    # make feature and label for model
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = Y.columns.tolist()

    return X, Y, category_names


def tokenize(text):
    """tokenizer that removes url, punctuation, and stopwords then lemmatizes
    Args:
        text (string): input message to tokenize
    Returns:
        tokens (list): list of cleaned tokens in the message
    """
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]

    words = [w for w in clean_tokens if w not in stopwords.words("english")]
    
    return words


def build_model():
    '''
    function for building pipeline and GridSearch
    Args: 
        None
    Returns: 
        cv (scikit-learn GridSearchCV): Grid search model object
    '''
    #pipeline for transforming data, fitting to model and predicting
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(max_depth=None,
                                                             min_samples_leaf=1,
                                                             min_samples_split=5,
                                                             n_estimators=20)
                                     ))
    ])
    # parameters for GridSearchCV
    parameters = {'clf__estimator__min_samples_leaf': [1, 2],
              'clf__estimator__min_samples_split': [2, 5],
              'clf__estimator__n_estimators': [10, 20]}
    # GridSearch with the above parameters
    cv = GridSearchCV(pipeline, parameters, 
                      scoring='f1_micro', verbose=10, n_jobs=-1)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """prints multi-output classification results
    Args:
        model (pandas dataframe): scikit-learn trained model
        X_text (pandas dataframe): X test data
        Y_test (pandas dataframe): Y test data
        category_names (list): the category names
    Returns:
        None
    """
    # Generate predictions
    Y_pred = model.predict(X_test)
    # Print out the full classification report
    print(classification_report(Y_test, Y_pred, target_names=category_names))
    

def save_model(model, model_filepath):
    """dumps the model to the given filepath
    Args:
        model (scikit-learn model): The fitted model
        model_filepath (string): the filepath to save the model to
    Returns:
        None
    """
    joblib.dump(model.best_estimator_, model_filepath)


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