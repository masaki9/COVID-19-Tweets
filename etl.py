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

df_tweet['user'] = df_tweet['user'].apply(lambda x: {} if pd.isna(x) else x)
df_user = pd.json_normalize(df_tweet['user'])
df_tweet['user_id'] = df_user['id']

df_place = df_tweet['place']
full_place_names = []
countries = []

for i in range(len(df_place)):
    try:
        name = df_place.iloc[i].get('full_name')
    except AttributeError:
        name = ''

    try:
        country = df_place.iloc[i].get('country')
    except AttributeError:
        country = ''

    full_place_names.append(name)
    countries.append(country)

df_tweet['full_place_name'] = full_place_names
df_tweet['country'] = countries
df_tweet.drop('place', axis=1, inplace=True)

df_tweet['date'] = pd.to_datetime(df_tweet['created_at']).dt.date
df_tweet['yyyy-mm'] = df_tweet['date'].apply(utils.get_yyyy_mm)

df_tweet = pd.merge(df_tweet, df_id_and_score, left_on='id', right_on='id', how='inner')

df_tweet = df_tweet[df_tweet['retweeted'] == False]
df_tweet.drop('retweeted', axis=1, inplace=True)

df_tweet = df_tweet[df_tweet['lang'] == 'en']
df_tweet.drop('lang', axis=1, inplace=True)

df_tweet_ca = df_tweet[df_tweet['country'] == 'Canada']

keywords = ['Pfizer', 'Moderna', 'AstraZeneca', 'Astra Zeneca', 'Janssen', 'Johnson & Johnson',
            'Johnson and Johnson', 'Covishield', 'mRNA', 'messenger RNA', 'vaccine',
            'vaccinate', 'vaccination', 'jab', 'inoculate', 'inoculation', 'injection',
            'dose', 'antibody', 'antigen', 'efficacy', 'vax', 'immune', 'immunity',
            'immunization', 'adverse event', 'adjuvant', 'booster', 'novavax', 'side effect',
            '1st shot', '2nd shot', 'first shot', 'second shot']
keywords = [i.lower() for i in keywords]

# Convert text to lowercase for text processing.
df_tweet_ca['full_text'] = df_tweet_ca['full_text'].str.lower()

# Get tweets that contain any of the keywords above.
df_tweet_ca = df_tweet_ca[df_tweet_ca['full_text'].str.contains('|'.join(keywords))]

df_tweet_ca['full_text'] = df_tweet_ca['full_text']\
    .str.replace('http\S+|www\S+', '', case=False, regex=True)  # Remove URLs

# Remove hash tags and username mentions
df_tweet_ca['full_text'] = df_tweet_ca['full_text']\
    .str.replace('#\S+|@\S+', '', case=False, regex=True)

# TODO: Find problematic users and remove them.
# Remove spams and tweets unrelated to Canada
spam_pattern = 'fighting stigma:|fighting stigma :|fighting stigma —|view article...|amgmt|ca \\||gmt —|cad —|utc -7 \\||sukhbir badal|amarinder singh|punjab chief minister|bengal chief minister|delhi chief minister|nadu chief minister|narendra modi|india|delhi|goa|bharat biotech|biological e|punjab|maha|pradesh|himachal|rahul gandhi|chandigarh|kangana ranaut|haryana|sputnik|sinovac|sinopharm|milkha singh|rasika dugal|raghu sharma|vivek oberoi|arshad warsi|vishal dadlani|sevak sharma|bengaluru|karnataka'
df_tweet_ca = df_tweet_ca[df_tweet_ca['full_text'].str.contains(spam_pattern) == False]

# Remove a problematic user's tweets
df_tweet_ca = df_tweet_ca[df_tweet_ca['user_id'] != 127943720]

# Perform text processing

df_tweet_ca['text_processed'] = df_tweet_ca['full_text'].apply(utils.remove_emojis)
df_tweet_ca['text_processed'] = df_tweet_ca['text_processed'].str.replace('&amp;|amp;', '', regex=True)

# Replace 'doses' with 'dose' because
# 'doses' does not get lemmatized properly (https://github.com/nltk/nltk/issues/2567).
df_tweet_ca['text_processed'] = df_tweet_ca['text_processed'].str.replace('doses', 'dose', regex=False)

df_tweet_ca['text_processed'] = df_tweet_ca['text_processed'].str.replace('per cent', 'percent', regex=False)

df_tweet_ca['text_processed'] = df_tweet_ca['text_processed'].apply(utils.remove_punctuations)
df_tweet_ca['text_processed'] = df_tweet_ca['text_processed'].apply(utils.correct_spellings)
df_tweet_ca['text_processed'] = df_tweet_ca['text_processed'].apply(utils.remove_stopwords)
df_tweet_ca['text_processed'] = df_tweet_ca['text_processed'].apply(utils.lemmatize_words)

# Create labels based on sentiment scores
df_tweet_ca['sentiment_label'] = df_tweet_ca['sentiment_score'].apply(utils.label_score)

df_tweet_ca = df_tweet_ca[['id', 'date', 'yyyy-mm', 'text_processed',
                           'sentiment_score', 'sentiment_label']]

print('Processed Dataset Size: {}'.format(df_tweet_ca.shape))
df_tweet_ca.to_csv('data/covid_vaccine_tweets_canada.csv', index=False)
