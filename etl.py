import glob
import pandas as pd
from emoji import UNICODE_EMOJI_ENGLISH

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

df_tweet.sort_values(by=['created_at'], inplace=True, ascending=False)

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
df_tweet = df_tweet[['id', 'date', 'time', 'full_text', 'retweeted',
                     'lang', 'full_place_name', 'country']]

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

df_tweet_ca['full_text'] = df_tweet_ca['full_text']\
    .str.replace('http\S+|www\S+', '', case=False, regex=True)  # Remove URLs

# Remove hash tags and username mentions
df_tweet_ca['full_text'] = df_tweet_ca['full_text']\
    .str.replace('#\S+|@\S+', '', case=False, regex=True)

# Remove spams
spam_pattern = 'fighting stigma:|fighting stigma :|fighting stigma —|view article...'
df_tweet_ca = df_tweet_ca[df_tweet_ca["full_text"].str.contains(spam_pattern) == False]


def convert_emojis(text):
    converted = []
    for char in text:
        if char in UNICODE_EMOJI_ENGLISH:
            converted.append(UNICODE_EMOJI_ENGLISH[char])
        else:
            converted.append(char)
    return ''.join(converted)


def remove_emojis(text):
    text_wo_emoji = []
    for char in text:
        if char in UNICODE_EMOJI_ENGLISH:
            text_wo_emoji.append('')
        else:
            text_wo_emoji.append(char)
    return ''.join(text_wo_emoji)


# df_tweet_ca['full_text'] = df_tweet_ca['full_text'].apply(convert_emojis)
df_tweet_ca['full_text'] = df_tweet_ca['full_text'].apply(remove_emojis)
print("COVID Vac Size: {}".format(df_tweet_ca.shape))
df_tweet_ca.to_csv('data/covid_vaccine_tweets_canada.csv', index=False)
