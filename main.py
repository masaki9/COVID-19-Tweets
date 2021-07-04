import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.set_option("display.max_rows", 100)
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

dfs = []
for file in score_file_paths:
    df = pd.read_csv(file, header=None)
    dfs.append(df)
df_id_and_score = pd.concat(dfs, ignore_index=True)

df_id_and_score.columns = ['id', 'sentiment_score']
df_id_and_score.drop_duplicates(inplace=True)

df_place = df_tweet['place']
full_place_names = []
countries = []

for i in range(len(df_place)):
    try:
        name = df_place.iloc[i].get("full_name")
    except AttributeError:
        name = ""

    try:
        country = df_place.iloc[i].get("country")
    except AttributeError:
        country = ""

    full_place_names.append(name)
    countries.append(country)

df_tweet['full_place_name'] = full_place_names
df_tweet['country'] = countries
df_tweet.drop('place', axis=1, inplace=True)

df_tweet["full_text"] = df_tweet["full_text"].str.lower()

df_tweet['date'] = pd.to_datetime(df_tweet['created_at']).dt.date
df_tweet['time'] = pd.to_datetime(df_tweet['created_at']).dt.time
df_tweet = df_tweet[['id', 'date', 'time', 'full_text', 'retweeted', 'lang', 'full_place_name', 'country']]

df_tweet = pd.merge(df_tweet, df_id_and_score, left_on='id', right_on='id', how='inner')

df_tweet = df_tweet[df_tweet['retweeted'] == False]
df_tweet.drop('retweeted', axis=1, inplace=True)

df_tweet = df_tweet[df_tweet['lang'] == 'en']
df_tweet.drop('lang', axis=1, inplace=True)

df_tweet_ca = df_tweet[df_tweet['country'] == 'Canada']

keywords = ['Pfizer', 'Moderna', 'AstraZeneca', 'Janssen', 'Johnson & Johnson',
            'Johnson and Johnson', 'Covishield', 'mRNA', 'messenger RNA', 'vaccine',
            'vaccinate', 'vaccination', 'jab', 'inoculate', 'inoculation', 'injection',
            'dose', 'antibody', 'antigen', 'efficacy', 'immune', 'immunity', 'immunization']
keywords = [i.lower() for i in keywords]

df_tweet_ca = df_tweet_ca[df_tweet_ca['full_text'].str.contains('|'.join(keywords))]
