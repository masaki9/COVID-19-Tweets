import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import utils as utils

pd.set_option("display.max_rows", 200)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 100)
pd.set_option('display.max_colwidth', 100)

data = 'data/covid_vaccine_tweets_canada.csv'
df = pd.read_csv(data, header=0, sep=',')
print("Dataset Size: {}".format(df.shape))

avg_score = np.mean(df["sentiment_score"])
print("Average Sentiment Score: {:.4f}".format(avg_score))

max_score = np.max(df["sentiment_score"])
print("Maximum Sentiment Score: {}".format(max_score))

min_score = np.min(df["sentiment_score"])
print("Minimum Sentiment Score: {}".format(min_score))

dist_plot = sns.displot(data=df['sentiment_score'], kde=True, height=12, aspect=20/12)
dist_plot.fig.subplots_adjust(top=0.95)
plt.xlabel('Sentiment Score', size=14)
plt.ylabel('Number of Tweets', size=14)
plt.title('Sentiment Score Distribution', size=18)
ax = plt.gca()
utils.add_bar_value_labels(ax)
# plt.show()


df_num_tweets_by_label = df[['sentiment_label']].value_counts().to_frame()\
    .reset_index().rename(columns={0: 'Number of Tweets'})
df_num_tweets_by_label = df_num_tweets_by_label.astype({'sentiment_label': str})

plt.figure(figsize=(20, 12))
x_axis = df_num_tweets_by_label['sentiment_label']
y_axis = df_num_tweets_by_label['Number of Tweets']
plt.bar(x_axis, y_axis, color='lightblue')
plt.xlabel('Sentiment Class', size=14)
plt.ylabel('Number of Tweets', size=14)
plt.title('Number of Tweets by Sentiment Class', size=18)
ax = plt.gca()
senti_labels = ['Positive', 'Neutral', 'Negative']
ax.set_xticklabels(senti_labels)
utils.add_bar_value_labels(ax)
# plt.show()


df_num_tweets_by_month = df[['id', 'yyyy-mm']].groupby('yyyy-mm').count().reset_index()
df_num_tweets_by_month = df_num_tweets_by_month.rename(columns={'id': 'Number of Tweets'})

plt.figure(figsize=(20, 12))
x_axis = df_num_tweets_by_month['yyyy-mm']
y_axis = df_num_tweets_by_month['Number of Tweets']
# plt.plot(x_axis, y_axis)
plt.bar(x_axis, y_axis, color='lightblue')
plt.xlabel('Month (yyyy-mm)', size=14)
plt.ylabel('Number of Tweets', size=14)
plt.title('Number of COVID-19 Vaccine Related Tweets in Canada by Month', size=18)
ax = plt.gca()
utils.add_bar_value_labels(ax)
# plt.show()


df_avg_scores_by_month = df[['yyyy-mm', 'sentiment_score']].groupby('yyyy-mm').mean().reset_index()
df_avg_scores_by_month = df_avg_scores_by_month.rename(columns={'sentiment_score': 'Average Sentiment Score'})

plt.figure(figsize=(20, 12))
x_axis = df_num_tweets_by_month['yyyy-mm']
y_axis = df_avg_scores_by_month['Average Sentiment Score']
plt.bar(x_axis, y_axis, color='lightblue')
plt.xlabel('Month (yyyy-mm)', size=14)
plt.ylabel('Average Sentiment Score', size=14)
plt.title('Average Sentiment Score by Month', size=18)
ax = plt.gca()
utils.add_bar_value_labels(ax)
# plt.show()

print('\nPositive Sentiment Unigrams')
unigrams_pos_sentiment = utils.get_ngrams_df(df, 'text_processed', 'sentiment_label', 1, 1)
print(unigrams_pos_sentiment.head(n=50))

plt.figure(figsize=(20, 12))
x_axis = unigrams_pos_sentiment['N-Gram'][:30]
y_axis = unigrams_pos_sentiment['Frequency'][:30]
plt.bar(x_axis, y_axis, color='lightgreen')
plt.xlabel('Unigram', size=14)
plt.xticks(rotation=45)
plt.ylabel('Frequency', size=14)
plt.title('Top 30 Unigrams Related to Positive Sentiments', size=18)
ax = plt.gca()
utils.add_bar_value_labels(ax)

print('\nPositive Sentiment Bigrams')
bigrams_pos_sentiment = utils.get_ngrams_df(df, 'text_processed', 'sentiment_label', 1, 2)
print(bigrams_pos_sentiment.head(n=50))

plt.figure(figsize=(20, 12))
x_axis = bigrams_pos_sentiment['N-Gram'][:30]
y_axis = bigrams_pos_sentiment['Frequency'][:30]
plt.bar(x_axis, y_axis, color='lightgreen')
plt.xlabel('Bigram', size=14)
plt.xticks(rotation=45)
plt.ylabel('Frequency', size=14)
plt.title('Top 30 Bigrams Related to Positive Sentiments', size=18)
ax = plt.gca()
utils.add_bar_value_labels(ax)

trigrams_pos_sentiment = utils.get_ngrams_df(df, 'text_processed', 'sentiment_label', 1, 3)
print('\nPositive Sentiment Trigrams')
print(trigrams_pos_sentiment.head(n=50))

plt.figure(figsize=(20, 12))
x_axis = trigrams_pos_sentiment['N-Gram'][:30]
y_axis = trigrams_pos_sentiment['Frequency'][:30]
plt.bar(x_axis, y_axis, color='lightgreen')
plt.xlabel('Trigram', size=14)
plt.xticks(rotation=45)
plt.ylabel('Frequency', size=14)
plt.title('Top 30 Trigrams Related to Positive Sentiments', size=18)
ax = plt.gca()
utils.add_bar_value_labels(ax)

print('\nNegative Sentiment Unigrams')
unigrams_neg_sentiment = utils.get_ngrams_df(df, 'text_processed', 'sentiment_label', -1, 1)
print(unigrams_neg_sentiment.head(n=50))

plt.figure(figsize=(20, 12))
x_axis = unigrams_neg_sentiment['N-Gram'][:30]
y_axis = unigrams_neg_sentiment['Frequency'][:30]
plt.bar(x_axis, y_axis, color='lightpink')
plt.xlabel('Unigram', size=14)
plt.xticks(rotation=45)
plt.ylabel('Frequency', size=14)
plt.title('Top 30 Unigrams Related to Negative Sentiments', size=18)
ax = plt.gca()
utils.add_bar_value_labels(ax)

print('\nNegative Sentiment Bigrams')
bigrams_neg_sentiment = utils.get_ngrams_df(df, 'text_processed', 'sentiment_label', -1, 2)
print(bigrams_neg_sentiment.head(n=50))

plt.figure(figsize=(20, 12))
x_axis = bigrams_neg_sentiment['N-Gram'][:30]
y_axis = bigrams_neg_sentiment['Frequency'][:30]
plt.bar(x_axis, y_axis, color='lightpink')
plt.xlabel('Bigram', size=14)
plt.xticks(rotation=45)
plt.ylabel('Frequency', size=14)
plt.title('Top 30 Bigrams Related to Negative Sentiments', size=18)
ax = plt.gca()
utils.add_bar_value_labels(ax)

trigrams_neg_sentiment = utils.get_ngrams_df(df, 'text_processed', 'sentiment_label', -1, 3)
print('\nNegative Sentiment Trigrams')
print(trigrams_neg_sentiment.head(n=50))

plt.figure(figsize=(20, 12))
x_axis = trigrams_neg_sentiment['N-Gram'][:30]
y_axis = trigrams_neg_sentiment['Frequency'][:30]
plt.bar(x_axis, y_axis, color='lightpink')
plt.xlabel('Trigram', size=14)
plt.xticks(rotation=45)
plt.ylabel('Frequency', size=14)
plt.title('Top 30 Trigrams Related to Negative Sentiments', size=18)
ax = plt.gca()
utils.add_bar_value_labels(ax)

plt.show()
