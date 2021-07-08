from collections import Counter
from emoji import UNICODE_EMOJI_ENGLISH
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
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


def remove_stopwords(text):
    ''' Remove stop words that do not carry useful information. '''
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)

    filtered = []
    for word in word_tokens:
        if word not in stop_words:
            filtered.append(word)

    return filtered


def remove_punctuations(text):
    ''' Remove punctuations from text '''
    # "’" and "‘" are not in string.punctuation so they are added.
    pattern = r"[{}{}]".format(string.punctuation, '’‘')
    return text.translate(str.maketrans('', '', pattern))


def stem_words(text):
    ''' Remove affixes from words using PorterStemmer. '''
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text]
    return ' '.join(text)


def vectorize_words(text):
    ''' Creates a matrix of word vectors. '''
    cv = CountVectorizer(binary=True)
    cv.fit(text)
    matrix = cv.transform(text)
    print("\nNumber vector size: {}".format(matrix.shape))

    return matrix


def list_ngrams(df, text_col, score_col, score_start, score_end,
                n_gram_size, most_common=50):
    ''' Create a list of n-grams '''
    # Create df based on the score range.
    df = df[(df[score_col] >= score_start) & (df[score_col] <= score_end)]

    sentences = [sentence.split() for sentence in df[text_col]]

    words = []
    for i in range(0, len(sentences)):
        words += sentences[i]

    counter_list = Counter(ngrams(words, n_gram_size)).most_common(most_common)

    print("\n{} N-Grams".format(n_gram_size))
    for i in range(0, len(counter_list)):
        print("Occurrences: ", str(counter_list[i][1]), end=" ")
        delimiter = ' '
        print("N-Gram: ", delimiter.join(counter_list[i][0]))

    return counter_list


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
