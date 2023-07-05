import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
import re
import contractions
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from textblob import TextBlob

nltk.download('wordnet')
nltk.download('omw-1.4')

TRAIN_DATA_PATH = "datasets/WASSA23_essay_level_with_labels_train.tsv"
DEV_DATA_PATH = "datasets/WASSA23_essay_level_dev.tsv"
TEST_DATA_PATH = "datasets/WASSA23_essay_level_test.tsv"
DEV_LABELS = "datasets/goldstandard_dev.tsv"

TRAIN22_DATA_PATH = "datasets/WASSA22_essay_level_with_labels_train.tsv"
DEV22_DATA_PATH = "datasets/WASSA22_essay_level_dev.tsv"
TEST22_DATA_PATH = "datasets/WASSA22_essay_level_test.tsv"
DEV22_LABELS = "datasets/goldstandard_dev.tsv"

DEV_COL_NAMES = [ # TODO: check!
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
VAL_SIZE = 0.2 # fraction of train data to be used as validation, TODO: 0.1?
RANDOM_STATE = 42


def get_stemmed_EMO_lexicon(lexicon, categories):
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
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    stemmed_lexicon = {}

    lexicon['stemma'] = ['' for _ in range(len(lexicon))]
    
    for word, _ in lexicon.iterrows():
  
        lemma = lemmatizer.lemmatize(word)
        stemma = stemmer.stem(lemma)
        lexicon.loc[word, 'stemma'] = stemma

        if stemma in stemmed_lexicon:
            l = [lexicon.loc[word, 'empathy']+stemmed_lexicon[stemma][0],
                  lexicon.loc[word, 'distress']+stemmed_lexicon[stemma][1], 
                  stemmed_lexicon[stemma][-1]+1]
            stemmed_lexicon[stemma] = l
            
        else:
            l = [ lexicon.loc[word, 'empathy'], lexicon.loc[word, 'distress'], 1]
            stemmed_lexicon[stemma] = l
        

    #print(stemmed_lexicon)
    #col_names = ['word', 'empathy', 'distress', 'count']
    
    stemmed_lexicon =  pd.DataFrame(stemmed_lexicon).T
    stemmed_lexicon.rename(columns={0:'empathy', 1:'distress', 2:'count'}, inplace=True)
    #print(stemmed_lexicon)
    stemmed_lexicon['empathy'] = stemmed_lexicon['empathy'].astype(float) / stemmed_lexicon['count']
    stemmed_lexicon['distress'] = stemmed_lexicon['distress'].astype(float) / stemmed_lexicon['count']
    #stemmed_lexicon[stemmed_lexicon > 0] = 1
    #print(stemmed_lexicon)
    #stemmed_lexicon = stemmed_lexicon.loc[(stemmed_lexicon!=0).any(axis=1)]
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

def add_lexica_counts(df, lexicon, stemmed_lexicon, categories):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    
    for category in categories:
        df[f'{category}_count'] = [0 for _ in range(len(df))]

    for index, row in df.iterrows():
        # remove punctuations and digits
        essay = remove_punctuations(row['essay'])
        essay = remove_digit(essay)
        count = 0

        # count emotion flags of essay tokens in lexicon
        for word in essay.split():
            if word in lexicon.index:
                count += 1
                for category in categories:
                    df.loc[index, f'{category}_count'] += lexicon.loc[word][category]
                    
                            
        # expand contractions, correct spelling, lemmatize and stemmed essay tokens
        essay = expand_contractions(essay)
        essay = correct_spelling(essay)
        lemmas = [lemmatizer.lemmatize(token) for token in essay.split()]
        stemmas = [stemmer.stem(lemma) for lemma in lemmas]
        
        # sum the emotion flags counts of stemmed essay tokens in stemmed_lexicon
        for stemma in stemmas:
            if stemma in stemmed_lexicon.index:
                count += 0.5
                for category in categories:
                    df.loc[index, f'{category}_count'] += stemmed_lexicon.loc[stemma][category]
        
              
        # nomalize counts by essay length and multiply by 100 (longest essay has around 200 words, max count 1600)
        word_count = len(row['essay'].split())
        for category in categories:
            if (len(categories) == 2):
                df.loc[index, f'{category}_count'] /= (word_count * count/100)
            else:
                df.loc[index, f'{category}_count'] /= (word_count/100)
    
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

def hope():
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

def lexicon(train_df, dev_df, test_df):
    
    categories_EMO = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'hope']
    categories_EMP = ['empathy', 'distress']
    
    lexicon_EMO = read_lexicon_df(categories_EMO)
    lexicon_EMP = read_lexicon_df(categories_EMP)
    
    stemmed_lexicon_EMO, lexicon_EMO = get_stemmed_EMO_lexicon(lexicon_EMO, categories_EMO)
    stemmed_lexicon_EMP, lexicon_EMP = get_stemmed_EMP_lexicon(lexicon_EMP)

    #train_df = pd.read_csv(TRAIN_DATA_PATH, sep='\t')
    train_df = add_lexica_counts(train_df, lexicon_EMO, stemmed_lexicon_EMO, categories_EMO)
    train_df = add_lexica_counts(train_df, lexicon_EMP, stemmed_lexicon_EMP, categories_EMP)
    #train_df.to_csv(TRAIN_DATA_PATH_OUT, index=False, sep='\t')

    #dev_df = pd.read_csv(DEV_DATA_PATH, sep='\t')
    dev_df = add_lexica_counts(dev_df, lexicon_EMO, stemmed_lexicon_EMO, categories_EMO)
    dev_df = add_lexica_counts(dev_df, lexicon_EMP, stemmed_lexicon_EMP, categories_EMP)
    #dev_df.to_csv(DEV_DATA_PATH_OUT, index=False, sep='\t')

    #test_df = pd.read_csv(TEST_DATA_PATH, sep='\t')
    test_df = add_lexica_counts(test_df, lexicon_EMO, stemmed_lexicon_EMO, categories_EMO)
    test_df = add_lexica_counts(test_df, lexicon_EMP, stemmed_lexicon_EMP, categories_EMP)
    #test_df.to_csv(TEST_DATA_PATH_OUT, index=False, sep='\t')

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


def pre(year):

    train_path = f"datasets/WASSA{year}_essay_level_train.tsv"
    dev_path = f"datasets/WASSA{year}_essay_level_dev.tsv"
    test_path = f"datasets/WASSA{year}_essay_level_test.tsv"
    dev_labels_path = f"datasets/WASSA{year}_goldstandard_dev.tsv"

    train_df = pd.read_csv(train_path, sep='\t')
    dev_df = pd.read_csv(dev_path, sep='\t')
    test_df = pd.read_csv (test_path, sep='\t')
    dev_lbl_df = pd.read_csv(dev_labels_path, sep='\t', names=DEV_COL_NAMES)
    
    hope()

    train_df, dev_df, test_df = lexicon(train_df, dev_df, test_df)

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
    #pre(22)
    pre(23)

if __name__ == "__main__":
    main()
