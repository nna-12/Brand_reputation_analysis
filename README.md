# Starbucks Brand Reputation Analysis
A comprehensive sentiment analysis of Starbucks customer reviews to assess brand reputation using machine learning techniques to assess and monitor brand reputation over time. This project aims to evaluate the brand reputation of Starbucks by performing sentiment analysis on customer reviews collected from various platforms. By leveraging Natural Language Processing (NLP) and Machine Learning (ML), we analyze the emotional tone of customer feedback and generate actionable insights about customer satisfaction and brand perception. This study aims to develop an automated sentiment analysis system to assess emotions in customer reviews belonging to the Starbucks Reviews Dataset and categorize them as Positive, Neutral, or Negative.


## Table of Contents
- [Dataset](#dataset)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Preprocessing](#data-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Data Resampling](#data-resampling)
- [Model Training - Traditional Models](#model-training---traditional-models)
- [Prediction & Evaluation](#prediction--evaluation)
- [Optimisation](#optimisation)
- [Ensemble Techniques](#ensemble-techniques)
- [Deployment](#deployment)


## Dataset
The dataset, named `reviews_data.csv` & downloaded from Kaggle, consists of  850 records and the following 6 columns
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
2. TF-IDF Vectorization: Converts textual data into numerical feature vectors. Words that are too common (max_df=0.85) or too rare (min_df=5) are filtered out to reduce noise.

TF-IDF helps capture the importance of terms in a document relative to the entire corpus.


## Data Resampling

To address class imbalance and improve model fairness, a two-step resampling strategy is used:

1. SMOTE (Synthetic Minority Over-sampling Technique): Creates synthetic examples of minority classes to enhance representation.
2. RandomUnderSampler: Reduces the size of the majority class to balance the dataset.

The combination results in a resampled training dataset (`X_train_bal`, `y_train_bal`) that enables more equitable model training across sentiment categories.


## Model Training - Traditional Models

A set of traditional machine learning classifiers is defined and trained on the balanced dataset obtained through SMOTE and under-sampling techniques. Each model is evaluated using a consistent evaluation function to compare performance metrics across models. 
Models Trained:
- Logistic Regression
- Naïve Bayes
- Support Vector Machines (SVM)
- k-Nearest Neighbors (k-NN)
- Decision Tree

The models were tested on unseen data, and their performance metrics were compared to identify the best-performing classifiers in this category.


## Prediction & Evaluation

Each trained model's performance is assessed using various classification metrics, and the most promising models from both traditional and ensemble categories are selected for in-depth evaluation. Metrics Used:

- Accuracy – Correct predictions over total predictions.
- Precision – True positives over predicted positives.
- Recall – True positives over actual positives.
- F1-Score – Harmonic mean of precision and recall.
- Matthews Correlation Coefficient (MCC) – A balanced measure that takes into account true and false positives and negatives.
- Cohen’s Kappa Score – Measures agreement between actual and predicted labels, correcting for chance.

Confusion matrices are plotted to visualize the distribution of predictions across sentiment classes (Positive, Negative, Neutral), helping to identify any model biases or misclassifications.


## Optimisation

To improve model performance, both dimensionality reduction and hyperparameter tuning techniques are applied:

### Dimensionality Reduction:
1. Principal Component Analysis (PCA): Reduces the number of features while retaining variance.
2. Linear Discriminant Analysis (LDA): Projects features in a way that maximizes class separability.

### Hyperparameter Tuning:
1. Grid Search: Exhaustive search over manually specified parameter values.
2. Random Search: Random combinations for faster yet effective tuning.

These techniques enhance model generalization and computational efficiency.


## Ensemble Techniques

To further enhance model robustness and accuracy, ensemble methods are employed. These techniques combine the predictions of multiple models to reduce variance and avoid overfitting.
Ensemble Models:
1. Random Forest
2. AdaBoost Classifier
3. Bagging Classifier

These models generally outperform individual classifiers, especially in terms of accuracy and F1-score, making them well-suited for production-level sentiment analysis systems.


## Deployment

The final, best-performing model is deployed as a RESTful API using FastAPI, providing real-time sentiment predictions for new reviews.
1. Cloud Hosting: Deployed to Google Cloud Platform (GCP) using services such as Cloud Run.
2. Version Control: Full source code and deployment instructions are available on GitHub.

