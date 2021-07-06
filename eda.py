import matplotlib.pyplot as plt
from matplotlib.transforms import BboxBase
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import seaborn as sns
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.util import ngrams
from collections import Counter
import string
from sklearn.feature_extraction.text import CountVectorizer

pd.set_option("display.max_rows", 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 100)
pd.set_option('display.max_colwidth', 100)

data = 'data/covid_vaccine_tweets_canada.csv'
df = pd.read_csv(data, header=0, sep=',')
print(df.shape)

avg_score = np.mean(df["sentiment_score"])
print("Average Score: {}".format(avg_score))

max_score = np.max(df["sentiment_score"])
print("Maximum Score: {}".format(max_score))

min_score = np.min(df["sentiment_score"])
print("Minimum Score: {}".format(min_score))

dist_plot = sns.displot(data=df['sentiment_score'], kde=True, height=12, aspect=20/12)
dist_plot.fig.subplots_adjust(top=0.95)
plt.xlabel('Sentiment Score')
plt.title('Sentiment Score Distribution')
plt.show()


def add_bar_value_labels(ax, spacing=5, decimal=4, size=10):
    # For each bar, place a label
    for rect in ax.patches:
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2
        data_label = np.round(rect.get_height(), decimals=decimal)
        ax.annotate(data_label, (x_value, y_value), xytext=(0, spacing), size=size,
                    textcoords="offset points", ha='center', va='bottom')


df_num_tweets_by_month = df[['id', 'yyyy-mm']].groupby('yyyy-mm').count().reset_index()
df_num_tweets_by_month = df_num_tweets_by_month.rename(columns={'id': 'Number of Tweets'})

plt.figure(figsize=(20, 12))
x_axis = df_num_tweets_by_month['yyyy-mm']
y_axis = df_num_tweets_by_month['Number of Tweets']
plt.plot(x_axis, y_axis)
plt.bar(x_axis, y_axis, color='lightblue')
plt.xlabel('Month (yyyy-mm)')
plt.ylabel('Number of Tweets')
plt.title('Number of COVID-19 Vaccine Related Tweets in Canada by Month')
ax = plt.gca()
add_bar_value_labels(ax)
plt.show()

df_avg_scores_by_month = df[['yyyy-mm', 'sentiment_score']].groupby('yyyy-mm').mean().reset_index()
df_avg_scores_by_month = df_avg_scores_by_month.rename(columns={'sentiment_score': 'Average Sentiment Score'})

plt.figure(figsize=(20, 12))
x_axis = df_num_tweets_by_month['yyyy-mm']
y_axis = df_avg_scores_by_month['Average Sentiment Score']
plt.bar(x_axis, y_axis, color='lightblue')
plt.xlabel('Month (yyyy-mm)')
plt.ylabel('Average Sentiment Score')
plt.title('Average Sentiment Score by Month')
ax = plt.gca()
add_bar_value_labels(ax)
plt.show()
