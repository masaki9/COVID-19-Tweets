import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import utils as utils
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression

data = 'data/covid_vaccine_tweets_canada.csv'
df = pd.read_csv(data, header=0, sep=',')

df['text_processed'] = df["full_text"].apply(utils.remove_punctuations)
df['text_processed'] = df['text_processed'].apply(utils.remove_stopwords)
df['text_processed'] = df['text_processed'].apply(utils.stem_words)

# Transforms words into numerical data for use in machine learning
vectorized = utils.vectorize_words(df['text_processed'])

# Create labels based on sentiment scores
df['sentiment_label'] = df['sentiment_score'].apply(utils.label_score)


def model_and_eval(X, y):
    ''' Create and evaluate a model. '''
    # Create training set with 75% of data and test set with 25% of data.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y.values.ravel(), train_size=0.75, test_size=0.25, random_state=40
    )

    model = LogisticRegression(multi_class='auto')

    mean_cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
    print("The mean accuracy score using 5-fold CV is: {}".format(mean_cv_score))

    mean_cv_recall = cross_val_score(model, X_train, y_train, cv=5, scoring='recall_macro').mean()
    print("The mean recall score using 5-fold CV is: {}".format(mean_cv_recall))

    mean_cv_precision = cross_val_score(model, X_train, y_train, cv=5, scoring='precision_macro').mean()
    print("The mean recall score using 5-fold CV is: {}".format(mean_cv_precision))

    model.fit(X_train, y_train)  # Train with the train set
    y_pred = model.predict(X_test)  # Predict target values.

    acc = model.score(X_test, y_test)
    print("\nThe accuracy score (test set) is: {}".format(acc))

    recall = recall_score(y_pred, y_test, average='macro')
    print("The recall score (test set) is: {}".format(recall))

    precision = precision_score(y_pred, y_test, average='macro')
    print("The precision score (test set) is: {}".format(precision))

    return y_test, y_pred


# Model using vectorized tweets as X and sentiment labels as y
y_test, y_pred = model_and_eval(vectorized, df[['sentiment_label']])


def show_confusion_matrix(y_test, y_predicted):
    ''' Plot a confusion matrix. '''
    cm = confusion_matrix(y_test, y_predicted)
    inds = ['Negative', 'Neutral', 'Positive']
    cols = ['Negative', 'Neutral', 'Positive']
    df = pd.DataFrame(cm, index=inds, columns=cols)

    plt.figure(figsize=(12, 9))

    ax = sns.heatmap(df, cmap='Blues', annot=True, fmt='g')
    ax.set(title="Sentiment Predictions - Actual vs Predicted")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment='right')

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


show_confusion_matrix(y_test, y_pred)
