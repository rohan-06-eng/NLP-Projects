# Kindle Review Sentiment Analysis Using NLP

## Overview
This project involves sentiment analysis of product reviews from the Amazon Kindle Store category, specifically focusing on analyzing the sentiment behind the reviews and predicting product ratings. The dataset spans from May 1996 to July 2014, with a total of 982,619 entries. The goal is to extract insights from the reviews, such as determining the usefulness of reviews, identifying fake reviews, and understanding factors influencing helpfulness ratings.

## Dataset Description

The dataset contains product reviews from Amazon's Kindle Store, where each reviewer has written at least five reviews, and each product has at least five reviews. The columns in the dataset include:

1. **asin** - Product ID (e.g., `B000FA64PK`).
2. **helpful** - Helpfulness rating of the review (e.g., `2/3` indicates two users found it helpful out of three).
3. **overall** - Rating of the product, typically ranging from 1 to 5 stars.
4. **reviewText** - Text content of the review (the main body of the review).
5. **reviewTime** - Raw time when the review was submitted.
6. **reviewerID** - ID of the reviewer (e.g., `A3SPTOKDG7WBLN`).
7. **reviewerName** - Name of the reviewer.
8. **summary** - Summary or brief description of the review.
9. **unixReviewTime** - Unix timestamp of when the review was posted.

### Data Size:
- Total entries: 982,619
- Each reviewer has at least 5 reviews.
- Each product has at least 5 reviews.

## Preprocessing and Cleaning

Before training machine learning models, preprocessing steps are crucial:
1. **Text Cleaning**: Removing unnecessary characters, punctuation, and converting text to lowercase.
2. **Tokenization**: Breaking text into individual words.
3. **Removing Stop Words**: Filtering out common words like "and," "the," "of," etc., that do not contribute to the sentiment.
4. **Lemmatization/Stemming**: Reducing words to their base form (e.g., "running" to "run").
5. **Handling Missing Values**: Imputing or removing rows with missing data.

## Feature Extraction

To convert textual data into a format that machine learning algorithms can understand, several techniques are used:

1. **Bag of Words (BoW)**: A method of representing text data by counting the frequency of each word in the text.
2. **TF-IDF (Term Frequency-Inverse Document Frequency)**: This technique weights the words based on their importance in a document relative to the corpus.
3. **Word2Vec**: A deep learning-based approach to represent words as vectors, capturing semantic meaning of words by analyzing their context.

## Train-Test Split
The dataset is split into training and testing sets to evaluate model performance. A typical split might be 80% for training and 20% for testing, ensuring the models are trained on one portion and evaluated on an unseen portion of the data.

## Machine Learning Models

Several machine learning algorithms are used to classify the sentiment of the reviews:

1. **Naive Bayes**: A probabilistic classifier based on Bayes' Theorem. It works well with text classification problems and assumes feature independence.
2. **Logistic Regression**: A statistical method for binary classification that can be used to predict the probability of a given review being positive or negative.
3. **Random Forest**: An ensemble learning method that uses multiple decision trees to make predictions, often improving performance compared to individual models.

### Sentiment Prediction Using BoW, TF-IDF, and Word2Vec
- **BoW**: Used for simple word frequency-based feature extraction. Often, this method is effective when the text is relatively short and the vocabulary is manageable.
- **TF-IDF**: Provides better performance over BoW as it weighs words based on their importance, helping in scenarios where there is a large vocabulary and the context of words matters.
- **Word2Vec**: Captures semantic relationships between words, providing a deeper understanding of the text. It can capture relationships like "king" is to "queen" as "man" is to "woman."

## Use Cases and Applications
1. **Sentiment Analysis**: Classifying reviews as positive, negative, or neutral based on the sentiment expressed in the text.
2. **Understanding Usefulness of Reviews**: Analyzing factors such as review length, rating, and reviewer's history to determine what makes a review helpful.
3. **Fake Review Detection**: Identifying suspicious patterns in review behavior, such as extremely high ratings from new users or contradictory information in the reviews.
4. **Product Comparison**: Analyzing top-rated products and drawing comparisons between them based on their reviews.
5. **Recommendation Systems**: Using sentiment analysis to recommend products that have received positive reviews.
6. **Product Improvement Insights**: Understanding the common themes in negative reviews to help vendors improve their products.

## Acknowledgements
- **Dataset**: The dataset is publicly available for research purposes and was provided by **Julian McAuley, UCSD**. You can access it from [this link](http://jmcauley.ucsd.edu/data/amazon/).
  
## Inspiration
The project is inspired by the need for understanding product reviews in the e-commerce space, providing insights into:
- How customers perceive products.
- What makes a review helpful.
- Detection of fake reviews.
- Comparing product quality based on sentiment analysis of reviews.

### Conclusion
This project provides valuable insights into customer feedback from the Kindle Store. By applying sentiment analysis and machine learning models to the review data, you can derive actionable information about product quality, customer satisfaction, and even identify fake or misleading reviews. The techniques used in this project, including text preprocessing, feature extraction, and machine learning algorithms, are standard approaches in natural language processing (NLP) for text-based classification tasks.