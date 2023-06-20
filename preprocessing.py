import pandas as pd

DATA_PATH = "datasets/WASSA23_essay_level_with_labels_train.tsv"
DATA_GOLD_DEV = "datasets/goldstandard_dev.tsv"

col_names = ["empathy", "distress", "emotion", "personality_conscientiousness", "personality_openes", "personality_extraversion", "personality_agreeableness", "personality_stability", "iri_perspective_taking", "iri_personal_distress", "iri_fantasy", "iri_empathatic_concern"]

df = pd.read_csv(DATA_PATH, sep='\t')
df_test = pd.read_csv(DATA_GOLD_DEV, sep='\t', names = col_names)

#df_test.columns(col_names)

emotions = set()
emotion_label = df["emotion"].unique()
print(emotion_label)
for label in emotion_label:
    emotions.update(label.split("/"))
print(emotions)
num_classes = len(emotions)
for emotion in emotions:
    df[emotion] = df["emotion"].str.contains(emotion).astype(int)


# df_train.merge(df_test)  
#df_merged = df_train.merge(df_test, left_index=True, right_index=True, how='outer')
df.to_csv("datasets/WASSA23_essay_level_train_preproc.tsv", index = False, sep = '\t') 