import pandas as pd

TRAIN_DATA_PATH = "datasets/WASSA23_essay_level_with_labels_train.tsv"
DEV_DATA_PATH = "datasets/WASSA23_essay_level_dev.tsv"
DEV_LABELS = "datasets/goldstandard_dev.tsv"

col_names = [ # TODO: check!
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

train_df = pd.read_csv(TRAIN_DATA_PATH, sep='\t')
dev_df = pd.read_csv(DEV_DATA_PATH, sep='\t')
dev_lbl_df = pd.read_csv(DEV_LABELS, sep='\t', names=col_names)

emotions = set()
for perceived_emotion in train_df['emotion'].unique():
    for emotion in perceived_emotion.split('/'):
        emotions.add(emotion)

for emotion in emotions:
    train_df[emotion] = train_df["emotion"].str.contains(emotion).astype(int)
    dev_df[emotion] = dev_lbl_df["emotion"].str.contains(emotion).astype(int)

dev_df = dev_df.merge(dev_lbl_df, left_index=True, right_index=True, how='outer')
train_df.to_csv("datasets/WASSA23_essay_level_train_preproc.tsv", index=False, sep='\t') 
dev_df.to_csv("datasets/WASSA23_essay_level_dev_preproc.tsv", index=False, sep='\t')