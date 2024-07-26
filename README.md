# Sentiment Analysis on Text Data

## Overview

This project performs sentiment analysis on a dataset of text reviews to classify them as positive or negative. By leveraging Natural Language Processing (NLP) and Machine Learning techniques, this project aims to understand customer sentiments and provide valuable insights from review data. It includes steps for data preprocessing, visualization, model training, evaluation, and persistence.

## Objectives

- **Understand Sentiment:** Analyze customer reviews to classify them as positive or negative, gaining insights into overall sentiment.
- **Visualize Data:** Create visual representations, such as word clouds, to illustrate the most frequent words associated with positive and negative sentiments.
- **Build a Model:** Develop a robust machine learning model that can accurately predict sentiment based on text data.

## Key Features

- **Data Preprocessing:** Clean and preprocess text data by removing stopwords using the NLTK library.
- **Visualization:** Generate and display word clouds to highlight the most significant words in positive and negative reviews.
- **Model Training:** Utilize TF-IDF Vectorization and Logistic Regression to train a sentiment classification model.
- **Model Evaluation:** Assess the model's performance with metrics like accuracy, classification reports, and confusion matrices.
- **Model Persistence:** Save the trained model and vectorizer for future use using Pickle.

## Dataset

- **Source:** The dataset `sentiment_dataset.csv` contains text reviews along with sentiment labels (positive or negative).
- **Content:** Includes review text and sentiment labels, providing a foundation for sentiment classification.

## Methodology

1. **Data Preprocessing:** Load the dataset, handle missing values, and clean review text by removing stopwords.
2. **Visualization:** Create word clouds for positive and negative reviews to visualize word frequency and sentiment trends.
3. **Model Training:** Transform text data into numerical features using TF-IDF Vectorization and train a Logistic Regression model.
4. **Model Evaluation:** Evaluate the model's performance on test data. The model achieves an accuracy of approximately 90-91%, indicating strong performance.
5. **Model Persistence:** Save the trained model and vectorizer for future predictions and easy deployment.

## How to Run

1. **Install Dependencies:** Ensure you have the required Python libraries installed using pip.
2. **Download NLTK Stopwords:** Download the stopwords dataset from NLTK for text preprocessing.
3. **Run the Script:** Place `sentiment_dataset.csv` in the same directory as the script and execute it to preprocess data, visualize word clouds, train the model, and evaluate performance.

## Usage

### Data Preprocessing

Text data is cleaned by removing stopwords to focus on meaningful content in the reviews.

### Visualization

Word clouds are generated for positive and negative reviews to show the most frequent and significant words associated with each sentiment.

### Model Training and Evaluation

The Logistic Regression model is trained on the processed data and evaluated on test data, achieving an accuracy of around 90-91%. The model's performance is analyzed using accuracy scores and detailed classification reports.

### Model Persistence

The trained model and TF-IDF vectorizer are saved using Pickle, enabling future use without the need for retraining.

## Libraries Used

- **pandas:** For data manipulation and analysis.
- **matplotlib:** For creating visualizations such as word clouds and confusion matrices.
- **nltk:** For processing text data, including stopword removal.
- **wordcloud:** For generating word cloud visualizations.
- **sklearn:** For machine learning tasks including vectorization, model training, and evaluation.
- **pickle:** For saving and loading the trained model and vectorizer.

## Future Improvements

- **Expand Dataset:** Incorporate more diverse and extensive datasets to improve model accuracy and generalization.
- **Advanced Models:** Explore other machine learning models and techniques, including Deep Learning, to enhance performance.
- **Additional Features:** Include features such as review metadata or user ratings to improve prediction accuracy.
- **Interactive Visualization:** Develop interactive tools to better explore sentiment trends and word frequencies.

## Contributions

Contributions are welcome! Please fork the repository and submit a pull request with your improvements or suggestions. For any issues or feedback, feel free to open an issue on the GitHub repository.
