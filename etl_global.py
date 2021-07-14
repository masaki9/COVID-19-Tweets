import glob
import pandas as pd
import utils as utils

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 100)
pd.set_option('display.max_colwidth', 30)

json_file_paths = [path for path in glob.glob('data_hydrated/geo-tagged/*.jsonl')]
score_file_paths = [path for path in glob.glob('data/geo-tagged/*.csv')]

# Read all files and concatenate them into one dataframe
dfs = []
for file in json_file_paths:
    df = pd.read_json(file, lines=True)
    dfs.append(df)
df_tweet = pd.concat(dfs, ignore_index=True)

df_tweet.sort_values(by=['created_at'], inplace=True, ascending=False)

dfs = []
for file in score_file_paths:
    df = pd.read_csv(file, header=None)
    dfs.append(df)
df_id_and_score = pd.concat(dfs, ignore_index=True)

df_id_and_score.columns = ['id', 'sentiment_score']
df_id_and_score.drop_duplicates(inplace=True)

df_tweet['date'] = pd.to_datetime(df_tweet['created_at']).dt.date
df_tweet['yyyy-mm'] = df_tweet['date'].apply(utils.get_yyyy_mm)

df_tweet = pd.merge(df_tweet, df_id_and_score, left_on='id', right_on='id', how='inner')

df_tweet = df_tweet[df_tweet['retweeted'] == False]
df_tweet.drop('retweeted', axis=1, inplace=True)

df_tweet = df_tweet[df_tweet['lang'] == 'en']
df_tweet.drop('lang', axis=1, inplace=True)

keywords = ['Pfizer', 'Moderna', 'AstraZeneca', 'Astra Zeneca', 'Janssen', 'Johnson & Johnson',
            'Johnson and Johnson', 'Covishield', 'mRNA', 'messenger RNA', 'vaccine',
            'vaccinate', 'vaccination', 'jab', 'inoculate', 'inoculation', 'injection',
            'dose', 'antibody', 'antigen', 'efficacy', 'vax', 'immune', 'immunity',
            'immunization', 'adverse event', 'adjuvant', 'booster', 'novavax', 'side effect',
            '1st shot', '2nd shot', 'first shot', 'second shot', 'Sputnik V', 'Sinopharm',
            'Sinovac', 'CoronaVac']
keywords = [i.lower() for i in keywords]

# Convert text to lowercase for text processing.
df_tweet['full_text'] = df_tweet['full_text'].str.lower()

# Get tweets that contain any of the keywords above.
df_tweet = df_tweet[df_tweet['full_text'].str.contains('|'.join(keywords))]

df_tweet['full_text'] = df_tweet['full_text']\
    .str.replace('http\S+|www\S+', '', case=False, regex=True)  # Remove URLs

# Remove hash tags and username mentions
df_tweet['full_text'] = df_tweet['full_text']\
    .str.replace('#\S+|@\S+', '', case=False, regex=True)

# Remove spams
spam_pattern = 'fighting stigma:|fighting stigma :|fighting stigma —|view article...|amgmt|ca \\||gmt —|cad —|utc -7'
df_tweet = df_tweet[df_tweet['full_text'].str.contains(spam_pattern) == False]

# Perform text processing

df_tweet['text_processed'] = df_tweet['full_text'].apply(utils.remove_emojis)
df_tweet['text_processed'] = df_tweet['text_processed'].str.replace('&amp;|amp;', '', regex=True)

# Replace 'doses' with 'dose' because
# 'doses' does not get lemmatized properly (https://github.com/nltk/nltk/issues/2567).
df_tweet['text_processed'] = df_tweet['text_processed'].str.replace('doses', 'dose', regex=False)

df_tweet['text_processed'] = df_tweet['text_processed'].str.replace('per cent', 'percent', regex=False)

df_tweet['text_processed'] = df_tweet['text_processed'].apply(utils.remove_punctuations)
df_tweet['text_processed'] = df_tweet['text_processed'].apply(utils.correct_spellings)
df_tweet['text_processed'] = df_tweet['text_processed'].apply(utils.remove_stopwords)
df_tweet['text_processed'] = df_tweet['text_processed'].apply(utils.lemmatize_words)

# Create labels based on sentiment scores
df_tweet['sentiment_label'] = df_tweet['sentiment_score'].apply(utils.label_score)

df_tweet = df_tweet[['id', 'date', 'yyyy-mm', 'text_processed',
                           'sentiment_score', 'sentiment_label']]

print('Processed Dataset Size: {}'.format(df_tweet.shape))
df_tweet.to_csv('data/covid_vaccine_tweets_global.csv', index=False)
