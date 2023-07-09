import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
import re
import contractions
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from textblob import TextBlob
from nrclex import NRCLex
from utils import EMPlexicon, generate_prompt
import json
import numpy as np
import os

ANTICIPATION_LEXICON_PATH = "./lexicon/anticipation.txt"
POSITIVE_LEXICON_PATH = "./lexicon/positive.txt"
JOY_LEXICON_PATH = "./lexicon/joy.txt"
SUBJECTIVITY_LEXICON_PATH = "./lexicon/subjclueslen1-HLTEMNLP05.tff"
HOPE_LEXICON_PATH = "./lexicon/hope.txt"

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

NRC_emotions = [
    'fear',
    'anger',
    'anticip',
    'trust',
    'surprise',
    'positive',
    'negative',
    'sadness',
    'disgust',
    'joy'
]

VAL_SIZE = 0.2
RANDOM_STATE = 42

# nltk.download('wordnet')
# nltk.download('omw-1.4')

    
def write_dict_to_json(dict, path):
    '''
    This function saves a dictionary to a json file.
    
    :param dict: dictionary to save
    :param path: path where to save the dictionary
    '''
    
    with open(path, 'w') as fp:
        json.dump(dict, fp, indent = 1)

def get_stemmed_EMO_lexicon(dataset):
    for emotion in NRC_emotions:
        dataset[f'{emotion}_count'] = ""

    for essay in dataset['essay']:
        NRC_obj = NRCLex(essay)
        for emotion in NRC_emotions:
            dataset[f'{emotion}_count'] = NRC_obj.affect_frequencies[f'{emotion}']
    
    return dataset

def get_stemmed_EMO_lexicon_per_word(dataset, split):
    lexicon = {}

    for idx, row in dataset.iterrows():
        essay = row['essay']
        local_lexicon = {'fear': [], 'anger': [], 'anticip': [], 'trust': [], 'surprise': [], 'positive': [], 
                        'negative': [], 'sadness': [], 'disgust': [], 'joy': []}

        for word in essay.split():
            NRC_obj = NRCLex(word)
            for emotion in NRC_emotions:
                local_lexicon[f'{emotion}'].append(NRC_obj.affect_frequencies[f'{emotion}'])


            """lexicon_fear.append(emotions_in_word['fear'] if 'fear' in emotions_in_word else 0)
            lexicon_anger.append(emotions_in_word['anger'] if 'anger' in emotions_in_word else 0)
            lexicon_anticip.append(emotions_in_word['anticip'] if 'anticip' in emotions_in_word else 0)
            lexicon_trust.append(emotions_in_word['trust'] if 'trust' in emotions_in_word else 0)
            lexicon_surprise.append(emotions_in_word['surprise'] if 'surprise' in emotions_in_word else 0)
            lexicon_positive.append(emotions_in_word['positive'] if 'positive' in emotions_in_word else 0)
            lexicon_negative.append(emotions_in_word['negative'] if 'negative' in emotions_in_word else 0)
            lexicon_sadness.append(emotions_in_word['sadness'] if 'sadness' in emotions_in_word else 0)
            lexicon_disgust.append(emotions_in_word['disgust'] if 'disgust' in emotions_in_word else 0)
            lexicon_joy.append(emotions_in_word['joy'] if 'joy' in emotions_in_word else 0)

        lexicon[row['essay_id']] = {'fear': lexicon_fear, 'anger': lexicon_anger, 'anticip': lexicon_anticip, 
                            'trust': lexicon_trust, 'surprise': lexicon_surprise, 'positive': lexicon_positive, 
                            'negative': lexicon_negative, 'sadness': lexicon_sadness, 'disgust': lexicon_disgust, 
                            'joy': lexicon_joy}"""
        
        lexicon[row['essay_id']] = local_lexicon
    
    path = './datasets/EMO_lexicon_per_word_' + split + '.json'
    write_dict_to_json(lexicon, path)

def get_stemmed_EMP_lexicon(dataset):
    EMP_lexicon_obj = EMPlexicon()
    dataset['empathy_count'] = ""
    dataset['distress_count'] = ""

    for essay in dataset['essay']:
        EMP_lexicon_obj.load_raw_text(essay)
        dataset['empathy_count'] = EMP_lexicon_obj.empathy_sentence_mean['empathy']
        dataset['distress_count'] = EMP_lexicon_obj.empathy_sentence_mean['distress']

    return dataset

def get_stemmed_EMP_lexicon_per_word(dataset, split):
    EMP_lexicon_obj = EMPlexicon()
    lexicon = {}

    for idx, row in dataset.iterrows():
        essay = row['essay']
        local_lexicon = {}

        EMP_lexicon_obj.load_raw_text(essay)
        local_lexicon['empathy'] = EMP_lexicon_obj.empathy_list
        local_lexicon['distress'] = EMP_lexicon_obj.distress_list

        lexicon[row['essay_id']] = local_lexicon
    
    path = './datasets/EMP_lexicon_per_word_' + split + '.json'
    write_dict_to_json(lexicon, path)

def read_lexicon_df(categories):
    categories_dfs = {}
    for category in categories:
        categories_dfs[category] = pd.read_csv(f"lexicon/{category}.txt", header=None, 
                                            names=['word', category], sep=None, engine='python')
    
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

"""def add_counts(df, split, lexicon, stemmed_lexicon, categories):
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
    
    return df"""

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
        if word=='abundance':
            print("ciao")
        if (value == '1' and (word in  subjectivity_lexicon or TextBlob(word).sentiment.subjectivity >= 0.5) 
            and (positive_lexicon[word]=='1' or joy_lexicon[word]=='1')):
            hope_lexicon[word] = 1
        else:
            hope_lexicon[word] = 0

    hope_lexicon = pd.DataFrame.from_dict(hope_lexicon, orient='index', columns=['value'])
    hope_lexicon.index.name = 'word'
    hope_lexicon.sort_values(by=['value', 'word'], ascending=[False, True], inplace=True)
    hope_lexicon.to_csv(HOPE_LEXICON_PATH, sep='\t', header=False)

def add_lexicon_features(internal_train_df, internal_val_df, dev_df, test_df):
    
    # add lexicon features for emotions
    internal_train_df = get_stemmed_EMO_lexicon(internal_train_df)
    internal_val_df = get_stemmed_EMO_lexicon(internal_val_df)
    dev_df = get_stemmed_EMO_lexicon(dev_df)
    test_df = get_stemmed_EMO_lexicon(test_df)

    # add lexicon features for empathy and distress
    internal_train_df = get_stemmed_EMP_lexicon(internal_train_df)
    internal_val_df = get_stemmed_EMP_lexicon(internal_val_df)
    dev_df = get_stemmed_EMP_lexicon(dev_df)
    test_df = get_stemmed_EMP_lexicon(test_df)

    # create dictionary with emotions values per word
    train_df = pd.concat([internal_train_df, internal_val_df])
    get_stemmed_EMO_lexicon_per_word(train_df, split='train')
    get_stemmed_EMO_lexicon_per_word(dev_df, split='dev')
    get_stemmed_EMO_lexicon_per_word(test_df, split='test')

    # create dictionary with empathy and distress values per word
    get_stemmed_EMP_lexicon_per_word(train_df, split='train')
    get_stemmed_EMP_lexicon_per_word(dev_df, split='dev')
    get_stemmed_EMP_lexicon_per_word(test_df, split='test')
    
    return internal_train_df, internal_val_df, dev_df, test_df

def drop_rows_with_unknown(dataframe):
    '''
    This function drops the rows with 'unknow' value in the dataframe passed as parameter.

    :param dataframe: pandas dataframe
    :return: pandas dataframe without rows with 'unknow' value
    '''

    for column in dataframe.columns:
        dataframe = dataframe[dataframe[column] != 'unknown']
    return dataframe

def add_word_count(dataframe):
    dataframe['essay_word_count'] = dataframe['essay'].apply(lambda x: len(x.split()))
    return dataframe

def add_prompt(dataframe, empathy=True):
    dataframe["prompt"] = ""
    for idx, row in dataframe.iterrows():
        text_prompt = generate_prompt(
            row['essay'],
            row['gender'],
            row['education'],
            row['race'],
            row['age'],
            row['income'],
            row['empathy'] if empathy else None,
            row['distress'] if empathy else None
            )
        dataframe["prompt"][idx] = text_prompt
    return dataframe

def remove_space_from_essay(dataframe):
    dataframe['essay'] = dataframe['essay'].str.replace("\r\n", " ")
    return dataframe

def preprocess(year):

    internal_train_path = f"datasets/WASSA{year}_essay_level_internal_train.tsv"
    internal_val_path = f"datasets/WASSA{year}_essay_level_internal_val.tsv"
    dev_path = f"datasets/WASSA{year}_essay_level_dev.tsv"
    test_path = f"datasets/WASSA{year}_essay_level_test.tsv"
    dev_labels_path = f"datasets/WASSA{year}_goldstandard_dev.tsv"

    internal_train_df = pd.read_csv(internal_train_path, sep='\t')
    internal_val_df = pd.read_csv(internal_val_path, sep='\t')
    dev_df = pd.read_csv(dev_path, sep='\t')
    test_df = pd.read_csv (test_path, sep='\t')
    dev_lbl_df = pd.read_csv(dev_labels_path, sep='\t', names=DEV_COL_NAMES)
    
    #build_hope_lexicon() #TODO: serve ancora?

    # merging dev labels with data
    dev_df = dev_df.merge(dev_lbl_df, left_index=True, right_index=True, how='outer')

    # drop unknown values
    internal_train_df = drop_rows_with_unknown(internal_train_df)
    internal_val_df = drop_rows_with_unknown(internal_val_df)
    dev_df = drop_rows_with_unknown(dev_df)
    test_df = drop_rows_with_unknown(test_df)

    # add essay word count
    internal_train_df = add_word_count(internal_train_df)
    internal_val_df = add_word_count(internal_val_df)
    dev_df = add_word_count(dev_df)
    test_df = add_word_count(test_df)

    # remouve unuselful space from essay
    test_df = remove_space_from_essay(test_df)

    # add essay_id to internal train and validation sets
    internal_train_df['essay_id'] = ""
    count = 0
    for idx, row in internal_train_df.iterrows():
        row['essay_id'] = count
        count += 1
    for idx, row in internal_val_df.iterrows():
        row['essay_id'] = count
        count += 1
    
    # add lexicon features
    internal_train_df, internal_val_df, dev_df, test_df = add_lexicon_features(internal_train_df, 
                                                            internal_val_df, dev_df, test_df)

    # add prompt with anagraphic data
    internal_train_df = add_prompt(internal_train_df)
    internal_val_df = add_prompt(internal_val_df)
    dev_df = add_prompt(dev_df)
    test_df = add_prompt(test_df, empathy=False)

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
    #preprocess(22) # TODO: non hanno gli essay_id, se si usa anche 22 bisogna dividere

    # preprocess WASSA 23 dataset
    preprocess(23)


if __name__ == "__main__":
    main()
