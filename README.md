# Brand_reputation_analysis
A comprehensive sentiment analysis of Starbucks customer reviews to assess brand reputation using machine learning techniques to assess and monitor brand reputation over time. This project aims to evaluate the **brand reputation of Starbucks** by performing **sentiment analysis** on customer reviews collected from various platforms. By leveraging Natural Language Processing (NLP) and Machine Learning (ML), we analyze the emotional tone of customer feedback and generate actionable insights about customer satisfaction and brand perception. This study aims to develop an automated sentiment analysis system to assess emotions in customer reviews belonging to the Starbucks Reviews Dataset and categorize them as Positive, Neutral, or Negative.

## Table of Contents
- [Dataset](#dataset)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Preprocessing](#data-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Data Resampling](#data-resampling)
- [Model Training - Traditional Models](#model-training)
- [Prediction & Evaluation](#prediction-and-evaluation)
- [Optimisation](#optmisation)
- [Ensemble Teachniques](#ensemble-techniques)
- [Deployment](#deployment)


## Dataset
The dataset, named `reviews_data.csv & downloaded from Kaggle, consists of  850 records and the following 6 columns
- `name`: Name of the reviewer
- `location`: Location of the reviewer
- `date`: Date of the review
- `rating`: The rating given by the reviewer (1 to 5)
- `review`: The text of the review
- `image_links`: Links to images associated with the review


## Exploratory Data Analysis

1. Initial analysis is performed to understand the structure and contents of the dataset. Key steps include:
2. Displaying dataset information: data types, column names, shape, and null value counts using df.info() and df.describe().
3. Rating Distribution: A distribution plot and pie chart are used to visualize the spread of customer ratings.
4. Review Trends: Bar charts show the volume of reviews by City, Day of the week, Year
5. Sentiment Over Time: A line graph illustrates how sentiment (positive, neutral, negative) varies across years, helping to track reputation changes over time.


## Data Preprocessing

Preprocessing ensures the text data is clean and standardized for analysis. The steps include:

1. Handling Missing Values - Rows with missing text reviews are dropped. Missing values in other columns are filled with the column median.
2. Text Cleaning - Removal of special characters, extra whitespaces, and punctuations.
3. Conversion of all text to lowercase for consistency.
   
This preprocessing stage prepares the textual data for effective feature extraction and modeling.


## Feature Engineering

To convert text into a format suitable for machine learning models, the following steps are applied:

1. Data Splitting: The dataset is divided into training and test sets (80% training, 20% testing).
2. **TF-IDF** Vectorization: Converts textual data into numerical feature vectors. Words that are too common (max_df=0.85) or too rare (min_df=5) are filtered out to reduce noise.

TF-IDF helps capture the importance of terms in a document relative to the entire corpus.

## Data Resampling

To address class imbalance and improve model fairness, a two-step resampling strategy is used:

1. SMOTE (Synthetic Minority Over-sampling Technique): Creates synthetic examples of minority classes to enhance representation.
2. RandomUnderSampler: Reduces the size of the majority class to balance the dataset.

The combination results in a resampled training dataset (`X_train_bal`, `y_train_bal`) that enables more equitable model training across sentiment categories.

## Model Training

1. Convert text data into numerical vectors using various techniques:
   - Bag of Words (Count Vectorizer)
   - TF-IDF Vectorizer
   - Continuous Bag of Words (CBOW)
   - Skip-gram
   - Pretrained Word Embeddings "word2vec-google-news-300"
   
2. Train a classification model using logistic regression on the vectorized data.
3. Evaluate the model's performance, achieving an accuracy score of 0.58 for multiclass classification.
4. Save the model using the `pickle` library.
5. Convert the problem into binary classification by creating a new column, `B_Rating`, where 1 and 2 are mapped to "Bad" and 3, 4, 5 are mapped to "Good."

## Balancing the Binary Dataset

To balance the binary dataset, down-sample "Bad" and "Good" reviews to match each other. The resulting balanced dataset is saved as `balanced_df1`.

## Model Training (Binary Classification)

1. Reapply the text vectorization techniques.
2. Split the data into training and testing sets.
3. Normalize the feature data using Min-Max scaling.
4. Train a logistic regression model for binary classification.
5. Achieve an accuracy score of 0.84 for the binary classification problem.
6. Generate a confusion matrix, heatmap, and classification report.

## Flask Web Application

The Flask web application (`app.py`) is created to deploy the trained predictive model. It allows users to input their review and receive predictions what rating they likely to give. The web application consists of two main HTML templates:
- `index.html`: The homepage where users input review.
- `prediction.html`: The page displaying the prediction.

## Running the Flask Web App

To run the Flask web app, follow these steps:
1. Install the required libraries listed in `requirements.txt` using `pip install -r requirements.txt`.
2. Run the Flask app by executing `python app.py`.
3. Open the web app in your browser by navigating to `http://localhost:5000`.

For any questions or suggestions, please feel free to contact me on LinkedIn.

## Webpage Glimpse:

![Index](index.png)
![Prediction](prediction.png)
