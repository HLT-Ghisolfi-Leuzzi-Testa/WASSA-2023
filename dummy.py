import pandas as pd
from utils import EMPlexicon, NRC_emotions
from preprocessing import hope_frequencies, read_NRC_lexicon_file
from nrclex import NRCLex

HOPE_LEXICON_PATH = "./lexicon/hope.txt"

def get_stemmed_EMP_lexicon_per_word(row):
    EMP_lexicon_obj = EMPlexicon()
    lexicon = {}
 
    essay = row['essay']
    local_lexicon = {'empathy': [], 'distress': []}

    for word in essay.split():
        EMP_lexicon_obj.load_raw_text(word)
        empathy_values = EMP_lexicon_obj.empathy_list
        distress_values = EMP_lexicon_obj.distress_list
        for emp_value, dis_value in zip(empathy_values, distress_values):
            local_lexicon['empathy'].append(emp_value)
            local_lexicon['distress'].append(dis_value)

    lexicon[row['essay_id']] = local_lexicon

def get_stemmed_EMO_lexicon_per_word(row):
    hope_lexicon = read_NRC_lexicon_file(HOPE_LEXICON_PATH)
    lexicon = {}

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


original_train_path = "./datasets/WASSA23_essay_level_original_train_preproc.tsv"
df = pd.read_csv(original_train_path, sep='\t')


for idx, row in df.iterrows():
    if row['essay_id'] == 1791:
        print(row['essay'])
        break
get_stemmed_EMO_lexicon_per_word(row)