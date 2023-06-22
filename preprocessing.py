import pandas as pd
import contractions
from spellchecker import SpellChecker

TRAIN_DATA_PATH = "https://raw.githubusercontent.com/HLT-Ghisolfi-Leuzzi-Testa/WASSA-2023/main/datasets/WASSA23_essay_level_with_labels_train.tsv"
DEV_DATA_PATH = "https://raw.githubusercontent.com/HLT-Ghisolfi-Leuzzi-Testa/WASSA-2023/main/datasets/WASSA23_essay_level_dev.tsv"
DEV_LABELS = "https://raw.githubusercontent.com/HLT-Ghisolfi-Leuzzi-Testa/WASSA-2023/main/datasets/goldstandard_dev.tsv"
DEV_COL_NAMES = [ # TODO: check!
    "empathy",
    "distress",
    "emotion",
    "personality_conscientiousness",
    "personality_openes",
    "personality_extraversion",
    "personality_agreeableness",
    "personality_stability",
    "iri_perspective_taking",
    "iri_personal_distress",
    "iri_fantasy",
    "iri_empathatic_concern"
]
VAL_SIZE = 0.2 # fraction of train data to be used as validation, TODO: 0.1?
RANDOM_STATE = 42

def one_hot_encode_emotions(df): # TODO: not needed?
    emotions = set()
    for perceived_emotion in df['emotion'].unique():
        for emotion in perceived_emotion.split('/'):
            emotions.add(emotion)
    
    for emotion in emotions:
        df[emotion] = df["emotion"].str.contains(emotion).astype(int)
    
    return df

def clean_text(text): # TODO: prendi parametri in input?
    '''
    This function cleans the corpus passed as parameter. 

    :param text: text string
    :return: cleaned text string
    '''

    # trim whitespaces
    text =  " ".join(text.split())

    # expand contractions
    #text = expand_contractions(text)

    # fix spelling
    #text = fix_spelling(text)

    # paraphrasing text with low frequecy emotions (?)
    # https://github.com/adityapatkar/WASSA2023_EMO/blob/main/Wassa_2023_Shared_Task_EMO_Training.ipynb

    return text

def fix_spelling(text): # TODO: many other available: jamspell, symspellpy, textblob
    '''
    This function corrects the spelling of the corpus passed as parameter.

    See also:
    [pyspellchecker](https://pypi.org/project/pyspellchecker/)

    :param text: text string
    :return: text string corrected
    '''

    spell_checker = SpellChecker()
    words = text.split()
    misspelled_words = spell_checker.unknown(words)
    corrected_words = []
    for word in words:
        if word in misspelled_words:
            corrected_word = spell_checker.correction(word)
            if corrected_word:
              corrected_words.append(corrected_word)
            else:
              corrected_words.append(word)
        else:
            corrected_words.append(word)
    return " ".join(corrected_words)

def expand_contractions(text):
    '''
    This function expands the contractions of the corpus passed as parameter.

    See also:
    [contractions](https://pypi.org/project/contractions/)

    :param text: text string
    :return: text string with expanded contractions
    '''
    return ' '.join(contractions.fix(word) for word in text.split())

train_df = pd.read_csv(TRAIN_DATA_PATH, sep='\t')
dev_df = pd.read_csv(DEV_DATA_PATH, sep='\t')
dev_lbl_df = pd.read_csv(DEV_LABELS, sep='\t', names=DEV_COL_NAMES)

# TODO: not needed?
# train_df = one_hot_encode_emotions(train_df)
# dev_lbl_df = one_hot_encode_emotions(dev_lbl_df)

# text cleaning
train_df["essay"] = train_df["essay"].apply(clean_text)
dev_df["essay"] = dev_df["essay"].apply(clean_text)

# splitting train data into train and validation
# TODO: assicurarsi che train e val abbiano stessa distribuzione di etichette
train_dataset = train_df.sample(frac=1-VAL_SIZE, random_state=RANDOM_STATE)
val_df = train_df.drop(train_dataset.index).reset_index(drop=True)
train_df = train_dataset.reset_index(drop=True)

# merging dev labels with data
dev_df = dev_df.merge(dev_lbl_df, left_index=True, right_index=True, how='outer')

# saving pre-processed data
train_df.to_csv("datasets/WASSA23_essay_level_train_preproc.tsv", index=False, sep='\t') 
val_df.to_csv("datasets/WASSA23_essay_level_val_preproc.tsv", index=False, sep='\t')
dev_df.to_csv("datasets/WASSA23_essay_level_dev_preproc.tsv", index=False, sep='\t')