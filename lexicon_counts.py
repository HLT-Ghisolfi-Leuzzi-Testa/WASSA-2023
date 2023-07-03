import pandas as pd
import nltk
import re
import contractions
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from textblob import TextBlob

nltk.download('wordnet')
nltk.download('omw-1.4')

CATEGORIES = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'hope']

TRAIN_DATA_PATH_IN = "./datasets/WASSA23_essay_level_with_labels_train.tsv"
TRAIN_DATA_PATH_OUT = "datasets/train_essay_level_preproc.tsv"
DEV_DATA_PATH_IN = "./datasets/WASSA23_essay_level_dev_preproc.tsv"
DEV_DATA_PATH_OUT = "datasets/dev_essay_level_preproc.tsv"
TEST_DATA_PATH_IN = "datasets/WASSA23_essay_level_test_preproc.tsv"
TEST_DATA_PATH_OUT = "datasets/test_essay_level_preproc.tsv"

def read_lexicon_df():
    categories_dfs = {}
    for category in CATEGORIES:
        categories_dfs[category] = pd.read_csv(f"./lexicon/{category}-NRC-Emotion-Lexicon.txt", header=None, names=['word', category], sep=None, engine='python')
    lexicon = pd.DataFrame(columns=['word'])
    for category in CATEGORIES:
        lexicon = pd.merge(lexicon, categories_dfs[category], on='word', how='outer')
    lexicon.dropna(inplace=True) # row with empty string
    lexicon.sort_values(by='word', inplace=True, ignore_index=True)
    lexicon.set_index('word', inplace=True)
    return lexicon

def get_stemmed_lexicon(lexicon):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    stemmed_lexicon = {}
    lexicon['stemma'] = ['' for _ in range(len(lexicon))]
    for word, _ in lexicon.iterrows():
        lemma = lemmatizer.lemmatize(word)
        stemma = stemmer.stem(lemma)
        lexicon.loc[word, 'stemma'] = stemma
        if stemma in stemmed_lexicon:
            stemmed_lexicon[stemma] += lexicon.loc[word, CATEGORIES]
        else:
            stemmed_lexicon[stemma] = lexicon.loc[word, CATEGORIES]
    stemmed_lexicon = pd.DataFrame(stemmed_lexicon).T
    stemmed_lexicon[stemmed_lexicon > 0] = 1
    stemmed_lexicon = stemmed_lexicon.loc[(stemmed_lexicon!=0).any(axis=1)]
    stemmed_lexicon.to_csv('./lexicon/stemmed_lexicon.csv')
    lexicon.to_csv('./lexicon/lexicon.csv')
    return stemmed_lexicon, lexicon

def remove_punctuations(text):
    return re.sub(r'[^\w\s]', '', text)

def remove_digit(text):
    return re.sub('\d+', '', text)

def expand_contractions(text):
    return contractions.fix(text)

def correct_spelling(text):
    textblob = TextBlob(text)
    return textblob.correct()

def add_lexica_counts(df, lexicon, stemmed_lexicon):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    
    for category in CATEGORIES:
        df[f'{category}_count'] = [0 for _ in range(len(df))]

    for index, row in df.iterrows():
        # remove punctuations and digits
        essay = remove_punctuations(row['essay'])
        essay = remove_digit(essay)

        # count emotion flags of essay tokens in lexicon
        for word in essay.split():
            if word in lexicon.index:
                for category in CATEGORIES:
                    df.loc[index, f'{category}_count'] += lexicon.loc[word][category]
        
        # expand contractions, correct spelling, lemmatize and stemmed essay tokens
        essay = expand_contractions(essay)
        essay = correct_spelling(essay)
        lemmas = [lemmatizer.lemmatize(token) for token in essay.split()]
        stemmas = [stemmer.stem(lemma) for lemma in lemmas]
        
        # sum the emotion flags counts of stemmed essay tokens in stemmed_lexicon
        for stemma in stemmas:
            if stemma in stemmed_lexicon.index:
                for category in CATEGORIES:
                    df.loc[index, f'{category}_count'] += stemmed_lexicon.loc[stemma][category]
        
        # nomalize counts by essay length and multiply by 100 (longest essay has around 200 words, max count 1600)
        word_count = len(row['essay'].split())
        for category in CATEGORIES:
            df.loc[index, f'{category}_count'] /= (word_count/100)
    
    return df

lexicon = read_lexicon_df()
stemmed_lexicon, lexicon = get_stemmed_lexicon(lexicon)

train_df = pd.read_csv(TRAIN_DATA_PATH_IN, sep='\t')
train_df = add_lexica_counts(train_df, lexicon, stemmed_lexicon)
train_df.to_csv(TRAIN_DATA_PATH_OUT, index=False, sep='\t')

dev_df = pd.read_csv(DEV_DATA_PATH_IN, sep='\t')
dev_df = add_lexica_counts(dev_df, lexicon, stemmed_lexicon)
dev_df.to_csv(DEV_DATA_PATH_OUT, index=False, sep='\t')

test_df = pd.read_csv(TEST_DATA_PATH_IN, sep='\t')
test_df = add_lexica_counts(test_df, lexicon, stemmed_lexicon)
test_df.to_csv(TEST_DATA_PATH_OUT, index=False, sep='\t')