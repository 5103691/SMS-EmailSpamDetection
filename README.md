# SMS Spam Detection Using Machine Learning

## Project Overview
This project implements a spam detection system for SMS messages using various machine learning techniques. It preprocesses SMS data, performs exploratory analysis, and trains multiple classifiers, including ensemble methods, to classify messages as either spam or ham (not spam). The model is saved for future use and deployed on Streamlit to create an interactive web application for better user experience.

## Libraries to Install
To run this project, you need to install the following libraries:

```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn wordcloud streamlit
```

## Project Structure

### Data Loading and Preprocessing
1. **Import Necessary Libraries**: 
   ```python
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   import seaborn as sns
   import nltk
   ```
2. **Load the Dataset**:
   ```python
   df = pd.read_csv("spam.csv", encoding='ISO-8859-1')
   ```
3. **Data Cleaning**: Drop unnecessary columns, rename them, and handle duplicates.

### Exploratory Data Analysis (EDA)
- Visualizations to understand the distribution of spam and ham messages.
- Analysis of message lengths, word counts, and sentence counts.

### Feature Engineering
- Text cleaning using NLP techniques to preprocess the messages.
- Converting text data into numerical vectors using `CountVectorizer` and `TfidfVectorizer`.

### Model Training
- Split the dataset into training and testing sets.
- Train various classifiers including:
  - Naive Bayes (Gaussian, Multinomial, Bernoulli)
  - Logistic Regression
  - Support Vector Machines (SVM)
  - Random Forest
  - Extra Trees Classifier

### Model Evaluation
- Evaluate models based on accuracy, precision, and recall.
- Utilize ensemble methods like Voting Classifier and Stacking Classifier to improve predictions.

### Model Saving
- Save the trained model and vectorizer using `pickle` for future use:
  ```python
  import pickle as pkl
  pkl.dump(tfidf, open("Vectorizer.pkl", "wb"))
  pkl.dump(clf, open("Model.pkl", "wb"))
  ```

### Streamlit Application
- Build a web application using Streamlit to allow users to input SMS messages for classification.
- The app preprocesses the input, vectorizes it, and provides a prediction on whether it is spam or not.

## Conclusion
The spam detection system was developed using a comprehensive dataset, leveraging multiple machine learning algorithms and ensemble techniques for improved accuracy. Visualizations were used to highlight the strengths and weaknesses of each model, and the final model achieved high performance with the Stacking Classifier.

To run the Streamlit application, use the command:
```bash
streamlit run app.py
```

This project showcases a practical application of machine learning in natural language processing, emphasizing the importance of feature extraction and model selection. Future work could explore advanced deep learning techniques for even better performance.
