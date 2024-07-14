# Sentiment-Analysis
Overview
This project performs sentiment analysis on a dataset of text reviews. The primary goal is to classify reviews as positive or negative using Natural Language Processing (NLP) and Machine Learning techniques. The project includes data preprocessing, visualization, model training, evaluation, and persistence steps.

Key Features
Data Preprocessing: Cleaning text data by removing stopwords using NLTK.
Visualization: Generating word clouds for positive and negative reviews to visualize the most frequent words.
Model Training: Using TF-IDF Vectorization and Logistic Regression for training a sentiment classification model.
Model Evaluation: Evaluating model performance with accuracy scores, classification reports, and confusion matrices.
Model Persistence: Saving the trained model and vectorizer for future use with Pickle.
Libraries Used
pandas: For data manipulation and analysis.
matplotlib: For visualizing word clouds and confusion matrices.
nltk: For natural language processing tasks like removing stopwords.
wordcloud: For generating word cloud visualizations.
sklearn: For machine learning tasks including vectorization, model training, evaluation, and splitting data.
pickle: For saving and loading the trained model and vectorizer.
How to Run
Install Dependencies:

bash
Copy code
pip install pandas matplotlib nltk wordcloud scikit-learn pickle-mixin
Download NLTK Stopwords:

python
Copy code
import nltk
nltk.download('stopwords')
Run the Script:

Ensure your data file sentiment_dataset.csv is in the same directory as the script.
Execute the script to preprocess data, visualize word clouds, train the model, and evaluate performance.
Usage
Data Preprocessing: Cleans text by removing stopwords.
Visualization: Generates and displays word clouds for positive and negative reviews.
Model Training and Evaluation: Trains a logistic regression model and evaluates its performance on test data.
Model Persistence: Saves the trained model and vectorizer for future predictions.
