# Disaster Response Web App

## Installation
The code contained in this repository was written in HTML and Python 3, and requires the following </br> Python packages: json, plotly, pandas, nltk, flask, sklearn, sqlalchemy, sys, numpy, re, pickle, warnings.

## Project Overview
This repository contains code for a web app which an emergency worker could use during a disaster event (e.g. an earthquake or hurricane), to classify a disaster message into several categories, in order that the message can be directed to the appropriate aid agencies. 

The app uses a ML model to categorize any new messages received, and the repository also contains the code used to train the model and to prepare any new datasets for model training purposes.

## File Descriptions
* **process_data.py**: This code takes as its input csv files containing message data and message categories (labels), and creates an SQLite database containing a merged and cleaned version of this data.
* **train_classifier.py**: This code takes the SQLite database produced by process_data.py as an input and uses the data contained within it to train and tune a ML model for categorizing messages. The output is a pickle file containing the fitted model. Test evaluation metrics are also printed as part of the training process.
* **data**: This folder contains sample messages and categories datasets in csv format.
* **app**: This folder contains all of the files necessary to run and render the web app.

## Running Instructions
### ***Run process_data.py***
1. Save the data folder in the current working directory and process_data.py in the data folder.
2. From the current working directory, run the following command:
`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

### ***Run train_classifier.py***
1. In the current working directory, create a folder called 'models' and save train_classifier.py in this.
2. From the current working directory, run the following command:
`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

### ***Run the web app***
1. Save the app folder in the current working directory.
2. Run the following command in the app directory:
    `python run.py`
3. Go to http://0.0.0.0:3001/

## Web App Preview

***Screenshot: Web App Front Page***


![Screenshot 1](https://github.com/hhej/disaster-response-pipeline/blob/3ef66f71e75ac5bb370c508d62f5dbd7b6eb65e0/ScreenShotFrontPage.png)

***Screenshot: Web App Results Page***


![Screenshot 2](https://github.com/hhej/disaster-response-pipeline/blob/3ef66f71e75ac5bb370c508d62f5dbd7b6eb65e0/ScreenShotResultPage.png)


## Licensing, Authors, Acknowledgements
***Authors:*** Panjapol Ampornratana

This app was completed as part of the [Udacity Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025). 

Code templates and data were provided by Udacity. The data was originally sourced by Udacity from Figure Eight.
