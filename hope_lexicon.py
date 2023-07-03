import pandas as pd
from textblob import TextBlob

ANTICIPATION_LEXICON_PATH = "./lexicon/anticipation-NRC-Emotion-Lexicon.txt"
POSITIVE_LEXICON_PATH = "./lexicon/positive-NRC-Emotion-Lexicon.txt"
JOY_LEXICON_PATH = "./lexicon/joy-NRC-Emotion-Lexicon.txt"
SUBJECTIVITY_LEXICON_PATH = "./lexicon/subjclueslen1-HLTEMNLP05.tff"
HOPE_LEXICON_PATH = "./lexicon/hope-NRC-Emotion-Lexicon.txt"

def read_NRC_lexicon_file(file_name):
    lexicon = {}
    with open(file_name, 'r') as file:
        for line in file:
            word, value = line.strip().split()
            lexicon[word] = value
    return lexicon

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

hope_vocabulary = pd.DataFrame.from_dict(hope_lexicon, orient='index', columns=['value'])
hope_vocabulary.index.name = 'word'
hope_vocabulary.sort_values(by=['value', 'word'], ascending=[False, True], inplace=True)
hope_vocabulary.to_csv(HOPE_LEXICON_PATH, sep='\t', header=False)