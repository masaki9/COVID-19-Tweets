from collections import Counter
from emoji import UNICODE_EMOJI_ENGLISH
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import string


def get_yyyy_mm(yyyy_mm_dd):
    ''' Return date in yyyy-mm format. '''
    return str(yyyy_mm_dd).split('-')[0] + '-' + str(yyyy_mm_dd).split('-')[1]


def convert_emojis(text):
    ''' Convert emojis found in text. '''
    converted = []
    for char in text:
        if char in UNICODE_EMOJI_ENGLISH:
            converted.append(UNICODE_EMOJI_ENGLISH[char])
        else:
            converted.append(char)
    return ''.join(converted)


def remove_emojis(text):
    ''' Remove emojis from text. '''
    text_wo_emoji = []
    for char in text:
        if char in UNICODE_EMOJI_ENGLISH:
            text_wo_emoji.append('')
        else:
            text_wo_emoji.append(char)
    return ''.join(text_wo_emoji)


def correct_spellings(text):
    ''' Correct some common typos. '''
    dict = {'dos': 'dose', 'vaccin': 'vaccine',
            '1st': 'first', '2nd': 'second', '1': 'one', '2': 'two',
            'pvt': 'private', 'govt': 'government'}

    word_tokens = word_tokenize(text)

    corrected = []
    for word in word_tokens:
        word = word.strip()
        if word in dict.keys():
            word = dict.get(word)
        corrected.append(word)

    return ' '.join(corrected)


def remove_stopwords(text):
    ''' Remove stop words that do not carry useful information. '''
    stop_words = set(stopwords.words('english'))
    stop_words.update(['r', 'v', 'u', 'ur', 'us', 'im', 'ive', 'sup', 'le',
                       'nt', 'cuz', 'thats', 'that\'s', 'around', 'besides',
                       'across', 'along', 'always', 'still', 'till', 'among',
                       'please', 'inc', 'weve'])

    word_tokens = word_tokenize(text)

    filtered = []
    for word in word_tokens:
        word = word.strip()
        if (word not in stop_words):
            filtered.append(word)

    return filtered


def remove_punctuations(text):
    ''' Remove punctuations from text '''
    # "’", "‘", "—", "…", "“", "”", and "–" are added in addition for removal.
    pattern = r"[{}{}]".format(string.punctuation, '’‘—…“”–')
    return text.translate(str.maketrans('', '', pattern))


def stem_words(text):
    ''' Remove affixes from words using PorterStemmer. '''
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text]
    return ' '.join(text)


def lemmatize_words(text):
    ''' Convert words to the base form
    while ensuring the convered words are part of the language. '''
    wnl = WordNetLemmatizer()
    text = [wnl.lemmatize(word) for word in text]
    return ' '.join(text)


def vectorize_words(text):
    ''' Creates a matrix of word vectors. '''
    cv = CountVectorizer(binary=True)
    cv.fit(text)
    matrix = cv.transform(text)
    print("\nNumber vector size: {}".format(matrix.shape))

    return matrix


def get_ngrams_df(df, text_col, sentiment_label_col, sentiment_label, n_gram_size):
    ''' Get a dataframe containing n-grams and frequencies for the sentiment label. '''
    # Create df for the sentiment label.
    df = df[(df[sentiment_label_col] == sentiment_label)]

    sentences = [sentence.split() for sentence in df[text_col]]

    words = []
    for i in range(0, len(sentences)):
        words += sentences[i]

    # Create a df containing n-grams and frequencies.
    df = pd.Series(ngrams(words, n_gram_size)).value_counts()
    df = df.to_frame().reset_index()
    df = df.rename(columns={'index': 'N-Gram', 0: 'Frequency'})
    df = df.astype({'N-Gram': str})

    # Remove brackets, quotes, and commas from n-grams.
    pattern = ',|\'|\\(|\\)'
    df['N-Gram'] = df['N-Gram'].str.replace(pattern, '', regex=True)

    return df


def add_bar_value_labels(ax, spacing=5, decimal=4, size=10):
    ''' Add data labels to bar charts. '''
    # For each bar, place a label
    for rect in ax.patches:
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2
        data_label = np.round(rect.get_height(), decimals=decimal)
        ax.annotate(data_label, (x_value, y_value), xytext=(0, spacing), size=size,
                    textcoords="offset points", ha='center', va='bottom')


def label_score(score):
    ''' Lable continous score value as -1, 0, or 1
    where -1, 0, and 1 mean negative, neutral, and positive respectively. '''
    if score < 0:
        value = -1
    elif score == 0:
        value = 0
    else:
        value = 1
    return value
