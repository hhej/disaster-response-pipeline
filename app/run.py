import json
import plotly
import joblib
import pandas as pd
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('labeled_messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    gen_count = df.groupby('genre').count()['message']
    gen_per = round(100*gen_count/gen_count.sum(), 2)
    genre_related = df[df['related']==1].groupby('genre').count()['message']
    genre_not_rel = df[df['related']==0].groupby('genre').count()['message']
    gen = list(gen_count.index)
    cat_num = df.drop(['id', 'message', 'original', 'genre'], axis = 1).sum()
    cat_num = cat_num.sort_values(ascending = False)
    cat = list(cat_num.index)
    
    # create visuals
    graphs = [
        {
          # circle chart for visualize genre distribution
            "data": [
              {
                "type": "pie",
                "uid": "f4de1f",
                "hole": 0.4,
                "name": "Genre",
                "pull": 0,
                "domain": {
                  "x": gen_per,
                  "y": gen
                },
                "marker": {
                  "colors": [
                    "#FDD26E",
                    "#5A8D84",
                    "#D9D9D6"
                   ]
                },
                "textinfo": "label+value",
                "hoverinfo": "all",
                "labels": gen,
                "values": gen_count
              }
            ],
            "layout": {
              "title": "Count and Percent of Messages by Genre"
            }
        },
        {
          # bar cahrt for count category num
            "data": [
              {
                "type": "bar",
                "x": cat,
                "y": cat_num,
                "marker": {
                  "color": '#869CAE'}
                }
            ],
            "layout": {
              "title": "Count of Messages by Category",
              'yaxis': {
                  'title': "Count"
              },
              'xaxis': {
                  'title': "Genre"
              },
              'barmode': 'group'
            }
        },
        {
          # bar chart for compare related and unrelated message in each genre
            'data': [
                Bar(
                    x=gen,
                    y=genre_related,
                    name = 'Related',
                    marker_color = '#8C8985'
                ),
                
                Bar(
                    x=gen,
                    y=genre_not_rel,
                    name = 'Not Related',
                    marker_color = '#FAAA8D'
                )
            ],

            'layout': {
                'title': 'Distribution of Messages by Genre and Related Status',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                },
                'barmode': 'group'
            }
        }

    ]

    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()