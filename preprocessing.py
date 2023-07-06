import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
import re
import contractions
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from textblob import TextBlob
import json
import numpy as np
import os
    
def write_dict_to_json(dict, path):
    '''
    This function saves a dictionary to a json file.
    
    :param dict: dictionary to save
    :param path: path where to save the dictionary
    '''
    
    with open(path, 'w') as fp:
        json.dump(dict, fp, indent = 1)

# nltk.download('wordnet')
# nltk.download('omw-1.4')

TRAIN_DATA_PATH = "datasets/WASSA23_essay_level_with_labels_train.tsv"
DEV_DATA_PATH = "datasets/WASSA23_essay_level_dev.tsv"
TEST_DATA_PATH = "datasets/WASSA23_essay_level_test.tsv"
DEV_LABELS = "datasets/goldstandard_dev.tsv"

TRAIN22_DATA_PATH = "datasets/WASSA22_essay_level_with_labels_train.tsv"
DEV22_DATA_PATH = "datasets/WASSA22_essay_level_dev.tsv"
TEST22_DATA_PATH = "datasets/WASSA22_essay_level_test.tsv"
DEV22_LABELS = "datasets/goldstandard_dev.tsv"

DEV_COL_NAMES = [
    "empathy",
    "distress",
    "emotion",
    "personality_conscientiousness",
    "personality_openess",
    "personality_extraversion",
    "personality_agreeableness",
    "personality_stability",
    "iri_perspective_taking",
    "iri_personal_distress",
    "iri_fantasy",
    "iri_empathatic_concern"
]

VAL_SIZE = 0.2
RANDOM_STATE = 42

def get_stemmed_EMO_lexicon(lexicon, categories):
    
    if (os.path.exists('./lexicon/lexicon_EMO.csv') and os.path.exists('./lexicon/stemmed_lexicon_EMO.csv')):
        stemmed_lexicon = pd.read_csv('./lexicon/stemmed_lexicon_EMO.csv', index_col='word')
        lexicon = pd.read_csv('./lexicon/lexicon_EMO.csv', index_col='word')
    else:
        lemmatizer = WordNetLemmatizer()
        stemmer = PorterStemmer()
        stemmed_lexicon = {}
        lexicon['stemma'] = ['' for _ in range(len(lexicon))]
        for word, _ in lexicon.iterrows():
            lemma = lemmatizer.lemmatize(word)
            stemma = stemmer.stem(lemma)
            lexicon.loc[word, 'stemma'] = stemma
            if stemma in stemmed_lexicon:
                stemmed_lexicon[stemma] += lexicon.loc[word, categories]
            else:
                stemmed_lexicon[stemma] = lexicon.loc[word, categories]
        stemmed_lexicon = pd.DataFrame(stemmed_lexicon).T
        stemmed_lexicon[stemmed_lexicon > 0] = 1
        stemmed_lexicon = stemmed_lexicon.loc[(stemmed_lexicon!=0).any(axis=1)]
        stemmed_lexicon.to_csv('./lexicon/stemmed_lexicon_EMO.csv')
        lexicon.to_csv('./lexicon/lexicon_EMO.csv')
    return stemmed_lexicon, lexicon

def get_stemmed_EMP_lexicon(lexicon):
    
    if (os.path.exists('./lexicon/lexicon_EMP.csv') and os.path.exists('./lexicon/stemmed_lexicon_EMP.csv')):
        stemmed_lexicon = pd.read_csv('./lexicon/stemmed_lexicon_EMP.csv', index_col='word')
        lexicon = pd.read_csv('./lexicon/lexicon_EMP.csv', index_col='word') 
    else:
        lemmatizer = WordNetLemmatizer()
        stemmer = PorterStemmer()
        stemmed_lexicon = {}
        lexicon['stemma'] = ['' for _ in range(len(lexicon))]
        for word, _ in lexicon.iterrows():
            lemma = lemmatizer.lemmatize(word)
            stemma = stemmer.stem(lemma)
            lexicon.loc[word, 'stemma'] = stemma
            if stemma in stemmed_lexicon:
                stemmed_lexicon[stemma] = [
                    lexicon.loc[word, 'empathy']+stemmed_lexicon[stemma][0],
                    lexicon.loc[word, 'distress']+stemmed_lexicon[stemma][1],
                    stemmed_lexicon[stemma][-1]+1
                ]
            else:
                stemmed_lexicon[stemma] = [
                    lexicon.loc[word, 'empathy'],
                    lexicon.loc[word, 'distress'],
                    1
                ]
        stemmed_lexicon =  pd.DataFrame(stemmed_lexicon).T
        stemmed_lexicon.rename(columns={0:'empathy', 1:'distress', 2:'count'}, inplace=True)
        stemmed_lexicon['empathy'] = stemmed_lexicon['empathy'].astype(float) / stemmed_lexicon['count']
        stemmed_lexicon['distress'] = stemmed_lexicon['distress'].astype(float) / stemmed_lexicon['count']
        stemmed_lexicon.to_csv('./lexicon/stemmed_lexicon_EMP.csv')
        lexicon.to_csv('./lexicon/lexicon_EMP.csv')
    
    return stemmed_lexicon, lexicon

def read_lexicon_df(categories):
    categories_dfs = {}
    for category in categories:
        categories_dfs[category] = pd.read_csv(f"lexicon/{category}.txt", header=None, names=['word', category], sep=None, engine='python')
    
    lexicon = pd.DataFrame(columns=['word'])
    
    for category in categories:
        lexicon = pd.merge(lexicon, categories_dfs[category], on='word', how='outer')

    lexicon.dropna(inplace=True) # row with empty string
    lexicon.sort_values(by='word', inplace=True, ignore_index=True)
    lexicon.set_index('word', inplace=True)
    return lexicon

def remove_punctuations(text):
    return re.sub(r'[^\w\s]', '', text)

def remove_digit(text):
    return re.sub('\d+', '', text)

def expand_contractions(text):
    return contractions.fix(text)

def correct_spelling(text):
    textblob = TextBlob(text)
    return textblob.correct()

def add_counts(df, split, lexicon, stemmed_lexicon, categories):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    
    for category in categories:
        df[f'{category}_count'] = [0 for _ in range(len(df))]
    
    per_word_values = {}
    
    
    essays_per_word_values = {}
    for index, row in df.iterrows():
        # remove punctuations and digits
        per_word_values = {}
    
        for category in categories:
            per_word_values[category] = [0.0 for _ in range(len(row['essay'].split()))]
        
        
        essay = remove_punctuations(row['essay'])
        essay = remove_digit(essay)
        count = 0
        
        # count emotion flags of essay tokens in lexicon
        for i, word in enumerate(essay.split()):
            counter_word_in_lexicon = 0
            word = word.lower()
            if word in lexicon.index:
                counter_word_in_lexicon += 1
                count += 1
                for category in categories:
                    df.loc[index, f'{category}_count'] += lexicon.loc[word][category].astype(float)
                    per_word_values[category][i] = lexicon.loc[word][category].astype(float)
                                        
            # expand contractions, correct spelling, lemmatize and stemm essay tokens
            word = expand_contractions(word)
            word = correct_spelling(word)
            lemmas = [lemmatizer.lemmatize(token) for token in word.split()]
            stemmas = [stemmer.stem(lemma) for lemma in lemmas]
            
            # sum the emotion flags counts of stemmed essay tokens in stemmed_lexicon
            for stemma in stemmas:
                if stemma in stemmed_lexicon.index:
                    counter_word_in_lexicon += 1
                    count += 0.5
                    for category in categories:
                        df.loc[index, f'{category}_count'] += stemmed_lexicon.loc[stemma][category].astype(float)
                        per_word_values[category][i] += stemmed_lexicon.loc[stemma][category].astype(float)
                else:
                    if (len(categories) == 2):    
                        per_word_values['empathy'][i] = 4.0 
            
            if(len(categories) == 2):
                for category in categories:
                    if (counter_word_in_lexicon != 0):
                        per_word_values[category][i] /= counter_word_in_lexicon
            else:
                for category in categories:
                    if (per_word_values[category][i] > 1):
                        per_word_values[category][i] = 1.0
        # nomalize counts
        word_count = len(row['essay'].split())
        for category in categories:
            if (len(categories) == 2):
                df.loc[index, f'{category}_count'] /= (word_count * count/100)
            else:
                df.loc[index, f'{category}_count'] /= (word_count/100)
        
        essays_per_word_values[row['essay_id']] = per_word_values
    
    if (len(categories) == 2):
        write_dict_to_json(essays_per_word_values, f"./lexicon/{split}_per_word_lexicon_EMP.json")
    else:
        write_dict_to_json(essays_per_word_values, f"./lexicon/{split}_per_word_lexicon_EMO.json")
    
    return df

ANTICIPATION_LEXICON_PATH = "./lexicon/anticipation.txt"
POSITIVE_LEXICON_PATH = "./lexicon/positive.txt"
JOY_LEXICON_PATH = "./lexicon/joy.txt"
SUBJECTIVITY_LEXICON_PATH = "./lexicon/subjclueslen1-HLTEMNLP05.tff"
HOPE_LEXICON_PATH = "./lexicon/hope.txt"

def read_NRC_lexicon_file(file_name):
    lexicon = {}
    with open(file_name, 'r') as file:
        for line in file:
            word, value = line.strip().split()
            lexicon[word] = value
    return lexicon

def build_hope_lexicon():
    # read NRC lexicon files
    anticipation_lexicon = read_NRC_lexicon_file(ANTICIPATION_LEXICON_PATH)
    joy_lexicon = read_NRC_lexicon_file(JOY_LEXICON_PATH)
    positive_lexicon = read_NRC_lexicon_file(POSITIVE_LEXICON_PATH)

    # read subjectivity lexicon file
    subjectivity_df = pd.read_csv(SUBJECTIVITY_LEXICON_PATH, header=None, usecols=[2], names=["word"], sep=" ")
    subjectivity_lexicon = subjectivity_df['word'].str.replace('word1=', '').tolist()

    # build hope lexicon
    hope_lexicon = {}
    for word, value in anticipation_lexicon.items():
        if (value == 1 and (word in  subjectivity_lexicon or TextBlob(word).sentiment.subjectivity >= 0.5) 
            and (positive_lexicon[word]==1 or joy_lexicon[word]==1)):
            hope_lexicon[word] = 1
        else:
            hope_lexicon[word] = 0

    hope_lexicon = pd.DataFrame.from_dict(hope_lexicon, orient='index', columns=['value'])
    hope_lexicon.index.name = 'word'
    hope_lexicon.sort_values(by=['value', 'word'], ascending=[False, True], inplace=True)
    hope_lexicon.to_csv(HOPE_LEXICON_PATH, sep='\t', header=False)

def add_lexicon_features(train_df, dev_df, test_df):
    
    categories_EMO = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'hope']
    categories_EMP = ['empathy', 'distress']
    
    lexicon_EMO = read_lexicon_df(categories_EMO)
    lexicon_EMP = read_lexicon_df(categories_EMP)
    
    stemmed_lexicon_EMO, lexicon_EMO = get_stemmed_EMO_lexicon(lexicon_EMO, categories_EMO)
    stemmed_lexicon_EMP, lexicon_EMP = get_stemmed_EMP_lexicon(lexicon_EMP)

    train_df = add_counts(train_df, "train", lexicon_EMO, stemmed_lexicon_EMO, categories_EMO)
    train_df = add_counts(train_df, "train",lexicon_EMP, stemmed_lexicon_EMP, categories_EMP)

    dev_df = add_counts(dev_df, "dev",lexicon_EMO, stemmed_lexicon_EMO, categories_EMO)
    dev_df = add_counts(dev_df, "dev",lexicon_EMP, stemmed_lexicon_EMP, categories_EMP)

    test_df = add_counts(test_df, "test",lexicon_EMO, stemmed_lexicon_EMO, categories_EMO)
    test_df = add_counts(test_df, "test",lexicon_EMP, stemmed_lexicon_EMP, categories_EMP)

    return train_df, dev_df, test_df

def drop_rows_with_unknown(dataframe):
    '''
    This function drops the rows with 'unknow' value in the dataframe passed as parameter.

    :param dataframe: pandas dataframe
    :return: pandas dataframe without rows with 'unknow' value
    '''

    for column in dataframe.columns:
        dataframe = dataframe[dataframe[column] != 'unknown']
    return dataframe


def preprocess(year):

    train_path = f"datasets/WASSA{year}_essay_level_train.tsv"
    dev_path = f"datasets/WASSA{year}_essay_level_dev.tsv"
    test_path = f"datasets/WASSA{year}_essay_level_test.tsv"
    dev_labels_path = f"datasets/WASSA{year}_goldstandard_dev.tsv"

    train_df = pd.read_csv(train_path, sep='\t')
    dev_df = pd.read_csv(dev_path, sep='\t')
    test_df = pd.read_csv (test_path, sep='\t')
    dev_lbl_df = pd.read_csv(dev_labels_path, sep='\t', names=DEV_COL_NAMES)
    
    build_hope_lexicon()

    train_df, dev_df, test_df = add_lexicon_features(train_df, dev_df, test_df)

    # merging dev labels with data
    dev_df = dev_df.merge(dev_lbl_df, left_index=True, right_index=True, how='outer')
    
    # splitting train data into train and validation with a stratified approach
    emotions = train_df['emotion'].unique().tolist()
    internal_train_df = pd.DataFrame()
    internal_val_df = pd.DataFrame()
    for emotion in emotions:
        emotion_df = train_df.loc[train_df['emotion']==emotion]
        if emotion_df.shape[0] < 2 : # if a class has a single sample it is added to the train set
            internal_train_df = pd.concat([internal_train_df, emotion_df])
        else:
            t_df, v_df = train_test_split(emotion_df, test_size=VAL_SIZE, stratify=emotion_df['emotion'], shuffle=True)
            internal_train_df = pd.concat([internal_train_df, t_df])
            internal_val_df = pd.concat([internal_val_df, v_df])

    # drop unknown values
    train_df = drop_rows_with_unknown(train_df)
    internal_train_df = drop_rows_with_unknown(internal_train_df)
    internal_val_df = drop_rows_with_unknown(internal_val_df)
    dev_df = drop_rows_with_unknown(dev_df)
    test_df = drop_rows_with_unknown(test_df)

    # get pre-processed train data (ordered by internal_train and internal_val)
    train_df = pd.concat([internal_train_df, internal_val_df])
    essay_level = pd.concat([train_df, dev_df])

    # saving pre-processed data
    train_df.to_csv(f"datasets/WASSA{year}_essay_level_train_preproc.tsv", index=False, sep='\t')
    internal_train_df.to_csv(f"datasets/WASSA{year}_essay_level_internal_train_preproc.tsv", index=False, sep='\t')
    internal_val_df.to_csv(f"datasets/WASSA{year}_essay_level_internal_val_preproc.tsv", index=False, sep='\t')
    dev_df.to_csv(f"datasets/WASSA{year}_essay_level_dev_preproc.tsv", index=False, sep='\t')
    test_df.to_csv(f"datasets/WASSA{year}_essay_level_test_preproc.tsv", index=False, sep='\t')
    essay_level.to_csv(f"datasets/WASSA{year}_essay_level_preproc.tsv", index=False, sep='\t')

def main():
    # preprocess WASSA 22 dataset
    # preprocess(22)
    # # preprocess WASSA 23 dataset
    # preprocess(23)
    year = 23
    train_path = f"datasets/WASSA{year}_essay_level_train.tsv"
    dev_path = f"datasets/WASSA{year}_essay_level_dev.tsv"
    test_path = f"datasets/WASSA{year}_essay_level_test.tsv"
    dev_labels_path = f"datasets/WASSA{year}_goldstandard_dev.tsv"

    train_df = pd.read_csv(train_path, sep='\t')#.head(5)
    dev_df = pd.read_csv(dev_path, sep='\t')#.head(5)
    test_df = pd.read_csv (test_path, sep='\t')#.head(5)
    #dev_lbl_df = pd.read_csv(dev_labels_path, sep='\t', names=DEV_COL_NAMES)
    add_lexicon_features(train_df, dev_df, test_df)

if __name__ == "__main__":
    main()
