import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
import re
import contractions
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from textblob import TextBlob
from nrclex import NRCLex
from utils import EMPlexicon, generate_prompt, NRC_emotions, read_NRC_lexicon_file, hope_essay_frequency
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

VAL_SIZE = 0.2
RANDOM_STATE = 42

# nltk.download('wordnet')
# nltk.download('omw-1.4')

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

def write_dict_to_json(dict, path):
    '''
    This function saves a dictionary to a json file.
    
    :param dict: dictionary to save
    :param path: path where to save the dictionary
    '''
    
    with open(path, 'w') as fp:
        json.dump(dict, fp, indent = 1)

def get_stemmed_EMO_lexicon(dataset, hope_lexicon):
    for emotion in NRC_emotions:
        dataset[f'{emotion}_count'] = ""
    dataset['hope_count'] = ""

    for idx, row in dataset.iterrows():
        essay = row['essay']
        NRC_obj = NRCLex(essay)
        emo_frequencies = NRC_obj.affect_frequencies
        if 'anticip' in emo_frequencies.keys():
            emo_frequencies.pop('anticip')
        if 'anticipation' not in emo_frequencies.keys():
            emo_frequencies['anticipation'] = 0.0
        for emotion in NRC_emotions:
            dataset.at[idx, f'{emotion}_count'] = emo_frequencies[f'{emotion}']
        dataset.at[idx, 'hope_count'] = hope_essay_frequency(essay, hope_lexicon)
    
    return dataset

def hope_frequencies(word, hope_lexicon):
    if word in hope_lexicon:
        return int(hope_lexicon[word])
    else:
        return 0

def get_stemmed_EMO_lexicon_per_word(dataset, split, hope_lexicon, year):
    lexicon = {}

    for idx, row in dataset.iterrows():
        essay = row['essay']
        local_lexicon = {'fear': [], 'anger': [], 'anticipation': [], 'trust': [], 'surprise': [], 'positive': [], 
                        'negative': [], 'sadness': [], 'disgust': [], 'joy': [], 'hope': []}

        for word in essay.split():
            NRC_obj = NRCLex(word)
            emo_frequencies = NRC_obj.affect_frequencies
            for emotion in NRC_emotions:
                if 'anticip' in emo_frequencies.keys():
                    emo_frequencies.pop('anticip')
                if 'anticipation' not in emo_frequencies.keys():
                    emo_frequencies['anticipation'] = 0.0
                local_lexicon[f'{emotion}'].append(1 if emo_frequencies[f'{emotion}'] > 0 else 0)
            local_lexicon['hope'].append(hope_frequencies(word, hope_lexicon))
        
        lexicon[row['essay_id']] = local_lexicon
    
    path = f'./datasets/EMO{year}_lexicon_per_word_' + split + '.json'
    write_dict_to_json(lexicon, path)

def get_stemmed_EMP_lexicon(dataset):
    EMP_lexicon_obj = EMPlexicon()
    dataset['empathy_count'] = ""
    dataset['distress_count'] = ""
    dataset['empathy_level'] = ""
    dataset['distress_level'] = ""
    
    for idx, row in dataset.iterrows():
        essay = row['essay']
        EMP_lexicon_obj.load_raw_text(essay)
        dataset.at[idx, 'empathy_count'] = EMP_lexicon_obj.empathy_sentence_mean['empathy']
        dataset.at[idx, 'distress_count'] = EMP_lexicon_obj.empathy_sentence_mean['distress']
        dataset.at[idx, 'empathy_level'] = EMP_lexicon_obj.empathy_sentence_mean['empathy_level']
        dataset.at[idx, 'distress_level'] = EMP_lexicon_obj.empathy_sentence_mean['distress_level']
        """dataset['empathy_count'][idx] = EMP_lexicon_obj.empathy_sentence_mean['empathy']
        dataset['distress_count'][idx] = EMP_lexicon_obj.empathy_sentence_mean['distress']"""

    return dataset

def get_stemmed_EMP_lexicon_per_word(dataset, split, year):
    EMP_lexicon_obj = EMPlexicon()
    lexicon = {}

    for idx, row in dataset.iterrows():
        essay = row['essay']
        local_lexicon = {'empathy': [], 'distress': []}

        for word in essay.split():
            EMP_lexicon_obj.load_raw_text(word)
            if len(EMP_lexicon_obj.empathy_list) > 0:
                local_lexicon['empathy'].append(EMP_lexicon_obj.empathy_list[0])
                local_lexicon['distress'].append(EMP_lexicon_obj.distress_list[0])
            else:
                local_lexicon['empathy'].append(4)
                local_lexicon['distress'].append(0)

        lexicon[row['essay_id']] = local_lexicon
    
    path = f'./datasets/EMP{year}_lexicon_per_word_' + split + '.json'
    write_dict_to_json(lexicon, path)

def remove_punctuations(text):
    return re.sub(r'[^\w\s]', '', text)

def remove_digit(text):
    return re.sub('\d+', '', text)

def expand_contractions(text):
    return contractions.fix(text)

def correct_spelling(text):
    textblob = TextBlob(text)
    return textblob.correct()

def split_train_val(train_df):
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

    return internal_train_df, internal_val_df

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
        if (value == '1' and (word in  subjectivity_lexicon or TextBlob(word).sentiment.subjectivity >= 0.5) 
            and (positive_lexicon[word]=='1' or joy_lexicon[word]=='1')):
            hope_lexicon[word] = 1
        else:
            hope_lexicon[word] = 0

    hope_lexicon = pd.DataFrame.from_dict(hope_lexicon, orient='index', columns=['value'])
    hope_lexicon.index.name = 'word'
    hope_lexicon.sort_values(by=['value', 'word'], ascending=[False, True], inplace=True)
    hope_lexicon.to_csv(HOPE_LEXICON_PATH, sep='\t', header=False)

def add_lexicon_features(internal_train_df, internal_val_df, dev_df, test_df, year):
    hope_lexicon = read_NRC_lexicon_file(HOPE_LEXICON_PATH)
    
    # add lexicon features for emotions
    internal_train_df = get_stemmed_EMO_lexicon(internal_train_df, hope_lexicon)
    internal_val_df = get_stemmed_EMO_lexicon(internal_val_df, hope_lexicon)
    dev_df = get_stemmed_EMO_lexicon(dev_df, hope_lexicon)
    if test_df is not None:
        test_df = get_stemmed_EMO_lexicon(test_df, hope_lexicon)

    # add lexicon features for empathy and distress
    internal_train_df = get_stemmed_EMP_lexicon(internal_train_df)
    internal_val_df = get_stemmed_EMP_lexicon(internal_val_df)
    dev_df = get_stemmed_EMP_lexicon(dev_df)
    if test_df is not None:
        test_df = get_stemmed_EMP_lexicon(test_df)

    # create dictionary with emotions values per word
    train_df = pd.concat([internal_train_df, internal_val_df])
    get_stemmed_EMO_lexicon_per_word(pd.concat([train_df, dev_df]), split="", hope_lexicon=hope_lexicon, year=year)
    if test_df is not None:
        get_stemmed_EMO_lexicon_per_word(test_df, split='test', hope_lexicon=hope_lexicon, year=year)

    # create dictionary with empathy and distress values per word
    get_stemmed_EMP_lexicon_per_word(pd.concat([train_df, dev_df]), split="", year=year)
    if test_df is not None:
        get_stemmed_EMP_lexicon_per_word(test_df, split='test', year=year)
    
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

def add_prompt(dataframe):
    dataframe["prompt_bio"] = ""
    dataframe["prompt_emp"] = ""
    dataframe["prompt_emo"] = ""
    for idx, row in dataframe.iterrows():
        bio_prompt, emp_prompt, emo_prompt = generate_prompt(
            row['essay'],
            row['gender'],
            row['education'],
            row['race'],
            row['age'],
            row['income'],
            row['empathy_level'],
            row['distress_level']
            )
        dataframe.at[idx, "prompt_bio"] = bio_prompt
        dataframe.at[idx, "prompt_emp"] = emp_prompt
        dataframe.at[idx, "prompt_emo"] = emo_prompt
    return dataframe

def lower_case_labels(dataset):
    dataset['emotion'] = dataset['emotion'].str.lower()
    return dataset

def remove_space_from_essay(dataframe):
    dataframe['essay'] = dataframe['essay'].str.replace("\r\n", " ")
    return dataframe

def match_essay_id(internal_df, original_internal_train_df, original_internal_val_df):
    for idx, row in internal_df.iterrows():
        keep_cheching = True
        for idx2, row2 in original_internal_train_df.iterrows():
            if row['essay'] == row2['essay']:
                original_internal_train_df.at[idx2, 'essay_id'] = row['essay_id']
                keep_cheching = False
                break
        if keep_cheching:
            for idx2, row2 in original_internal_val_df.iterrows():
                if row['essay'] == row2['essay']:
                    original_internal_val_df.at[idx2, 'essay_id'] = row['essay_id']
                    break
    
    return original_internal_train_df, original_internal_val_df
                
def preprocess(year):

    if year == 22:
        train_path = f"datasets/WASSA22_essay_level_train.tsv"

        train_df = pd.read_csv(train_path, sep='\t')
        internal_train_df, internal_val_df = split_train_val(train_df)
    else:
        internal_train_path = f"datasets/WASSA23_essay_level_internal_train.tsv"
        internal_val_path = f"datasets/WASSA23_essay_level_internal_val.tsv"
        original_train_path = f"datasets/WASSA23_essay_level_train_original.tsv"
        test_path = f"datasets/WASSA{year}_essay_level_test.tsv"

        internal_train_df = pd.read_csv(internal_train_path, sep='\t')
        internal_val_df = pd.read_csv(internal_val_path, sep='\t')
        original_train_df = pd.read_csv(original_train_path, sep='\t')
        original_internal_train_df, original_internal_val_df = split_train_val(original_train_df)
        test_df = pd.read_csv (test_path, sep='\t')

    dev_path = f"datasets/WASSA{year}_essay_level_dev.tsv"
    dev_labels_path = f"datasets/WASSA{year}_goldstandard_dev.tsv"

    dev_df = pd.read_csv(dev_path, sep='\t')
    dev_lbl_df = pd.read_csv(dev_labels_path, sep='\t', names=DEV_COL_NAMES)
    
    build_hope_lexicon()

    # merging dev labels with data
    dev_df = dev_df.merge(dev_lbl_df, left_index=True, right_index=True, how='outer')

    # drop unknown values
    internal_train_df = drop_rows_with_unknown(internal_train_df)
    internal_val_df = drop_rows_with_unknown(internal_val_df)
    dev_df = drop_rows_with_unknown(dev_df)
    if year == 23:
        test_df = drop_rows_with_unknown(test_df)
        original_internal_train_df = drop_rows_with_unknown(original_internal_train_df)
        original_internal_val_df = drop_rows_with_unknown(original_internal_val_df)

    # add essay word count
    internal_train_df = add_word_count(internal_train_df)
    internal_val_df = add_word_count(internal_val_df)
    dev_df = add_word_count(dev_df)
    if year == 23:
        test_df = add_word_count(test_df)
        original_internal_train_df = add_word_count(original_internal_train_df)
        original_internal_val_df = add_word_count(original_internal_val_df)
    
    # convert labels to lower case
    internal_train_df = lower_case_labels(internal_train_df)
    internal_val_df = lower_case_labels(internal_val_df)
    dev_df = lower_case_labels(dev_df)
    if year == 23:
        original_internal_train_df = lower_case_labels(original_internal_train_df)
        original_internal_val_df = lower_case_labels(original_internal_val_df)
        # remouve unuselful space from essay
        test_df = remove_space_from_essay(test_df)

    # add essay_id to internal train and validation sets
    internal_train_df['essay_id'] = ""
    internal_val_df['essay_id'] = ""
    dev_df['essay_id'] = ""
    count = 0
    for idx, row in internal_train_df.iterrows():
        internal_train_df.at[idx, 'essay_id'] = count
        count += 1
    for idx, row in internal_val_df.iterrows():
        internal_val_df.at[idx, 'essay_id'] = count
        count += 1
    for idx, row in dev_df.iterrows():
        dev_df.at[idx, 'essay_id'] = count
        count += 1
    
    if year == 23:
        original_internal_train_df, original_internal_val_df = match_essay_id(
            internal_train_df, original_internal_train_df, original_internal_val_df)
        original_internal_train_df, original_internal_val_df = match_essay_id(
            internal_val_df, original_internal_train_df, original_internal_val_df)
    
    # add lexicon features
    if year == 22:
        test_df = None
    internal_train_df, internal_val_df, dev_df, test_df = add_lexicon_features(internal_train_df, 
                                                            internal_val_df, dev_df, test_df, year)
    if year == 23:
        # add lexicon features for emotions
        hope_lexicon = hope_lexicon = read_NRC_lexicon_file(HOPE_LEXICON_PATH)
        original_internal_train_df = get_stemmed_EMO_lexicon(original_internal_train_df, hope_lexicon)
        original_internal_val_df = get_stemmed_EMO_lexicon(original_internal_val_df, hope_lexicon)

        # add lexicon features for empathy and distress
        original_internal_train_df = get_stemmed_EMP_lexicon(original_internal_train_df)
        original_internal_val_df = get_stemmed_EMP_lexicon(original_internal_val_df)

    # add prompt with anagraphic data
    internal_train_df = add_prompt(internal_train_df)
    internal_val_df = add_prompt(internal_val_df)
    dev_df = add_prompt(dev_df)
    if year == 23:
        test_df = add_prompt(test_df)
        original_internal_train_df = add_prompt(original_internal_train_df)
        original_internal_val_df = add_prompt(original_internal_val_df)

    # get pre-processed train data (ordered by internal_train and internal_val)
    train_df = pd.concat([internal_train_df, internal_val_df])
    essay_level = pd.concat([train_df, dev_df])
    if year == 23:
        original_train_df = pd.concat([original_internal_train_df, original_internal_val_df])
        original_essay_level = pd.concat([original_train_df, dev_df])

    # saving pre-processed data
    train_df.to_csv(f"datasets/WASSA{year}_essay_level_train_preproc.tsv", index=False, sep='\t')
    internal_train_df.to_csv(f"datasets/WASSA{year}_essay_level_internal_train_preproc.tsv", index=False, sep='\t')
    internal_val_df.to_csv(f"datasets/WASSA{year}_essay_level_internal_val_preproc.tsv", index=False, sep='\t')
    dev_df.to_csv(f"datasets/WASSA{year}_essay_level_dev_preproc.tsv", index=False, sep='\t')
    essay_level.to_csv(f"datasets/WASSA{year}_essay_level_preproc.tsv", index=False, sep='\t')
    if year == 23:
        test_df.to_csv(f"datasets/WASSA{year}_essay_level_test_preproc.tsv", index=False, sep='\t')
        original_internal_train_df.to_csv(f"datasets/WASSA{year}_essay_level_original_internal_train_preproc.tsv", index=False, sep='\t')
        original_internal_val_df.to_csv(f"datasets/WASSA{year}_essay_level_original_internal_val_preproc.tsv", index=False, sep='\t')
        original_train_df.to_csv(f"datasets/WASSA{year}_essay_level_original_train_preproc.tsv", index=False, sep='\t')
        original_essay_level.to_csv(f"datasets/WASSA{year}_essay_level_original_preproc.tsv", index=False, sep='\t')

def main():
    # preprocess WASSA 22 dataset
    preprocess(22)

    # preprocess WASSA 23 dataset
    preprocess(23)


if __name__ == "__main__":
    main()
