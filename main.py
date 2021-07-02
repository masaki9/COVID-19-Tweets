import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.set_option("display.max_rows", 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 100)
pd.set_option('display.max_colwidth', 30)

df_id_and_score = pd.read_csv("data/geo-tagged/2020_december1_december2.csv", header=None)
df_id_and_score.columns = ['id', 'sentiment_score']

df_tweet = pd.read_json("data_hydrated/geo-tagged/2020_december1_december2.jsonl", lines=True)
df_tweet = df_tweet[['id', 'created_at', 'full_text', 'retweeted', 'lang', 'place']]

df_place = df_tweet['place']
full_place_names = []
countries = []

for i in range(len(df_place)) :
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

# print(df_tweet.head())
print(df_tweet.shape)
print(df_tweet.dtypes)

df_tweet['date'] = pd.to_datetime(df_tweet['created_at']).dt.date
df_tweet['time'] = pd.to_datetime(df_tweet['created_at']).dt.time
df_tweet = df_tweet[['id', 'date', 'time', 'full_text', 'retweeted', 'lang', 'full_place_name', 'country']]

# print(df_tweet.head())
print(df_tweet.dtypes)
print(df_tweet.shape)

df = pd.merge(df_tweet, df_id_and_score, left_on='id', right_on='id', how='inner')
print(df.head())
print(df.dtypes)
print(df.shape)
