import matplotlib.pyplot as plt
# from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import seaborn as sns
import utils as utils

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
utils.add_bar_value_labels(ax)
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
utils.add_bar_value_labels(ax)
plt.show()

# df['tokenized'] = df['full_text'].apply(word_tokenize)

df['text_processed'] = df["full_text"].apply(utils.remove_punctuations)
df['text_processed'] = df['text_processed'].apply(utils.remove_stopwords)
df['text_processed'] = df['text_processed'].apply(utils.stem_words)

unigrams = utils.list_ngrams(df, 'text_processed', 'sentiment_score', 0.00000001, 1.0, 2)
print()
unigrams = utils.list_ngrams(df, 'text_processed', 'sentiment_score', -1.0, -0.00000001, 2)
