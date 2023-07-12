import pandas as pd
from utils import EmotionsLabelEncoder, compute_EMO_metrics, write_dict_to_json, write_EMO_predictions
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelEncoder

def get_predictions(clf, X_train, y_train, X_test):
    clf.fit(X_train, y_train)
    return clf.predict(X_test)

def apply_dummy_clfs(train_df, test_df, split): # TODO: per class metrics?
    ordinal_encoder = LabelEncoder()
    y = ordinal_encoder.fit_transform(train_df.emotion)
    onehot_encoder = EmotionsLabelEncoder()
    onehot_encoder.fit(train_df.emotion)
    y_bin = onehot_encoder.encode(train_df.emotion)

    mostfreq_clf = DummyClassifier(strategy="most_frequent")
    mostfreq_preds = get_predictions(clf=mostfreq_clf, X_train=train_df, y_train=y, X_test=test_df)
    mostfreq_emotions = ordinal_encoder.inverse_transform(mostfreq_preds)
    if split == "dev":
        mostfreq_metrics = compute_EMO_metrics(golds=test_df.emotion, predictions=mostfreq_emotions)
        write_dict_to_json(mostfreq_metrics, f"mostfreq_{split}_metrics.json")
    else:
        write_EMO_predictions(mostfreq_emotions, f"mostfreq_{split}_predictions_EMO.tsv")

    stratified_clf = DummyClassifier(strategy="stratified")
    stratified_preds_bin = get_predictions(clf=stratified_clf, X_train=train_df, y_train=y_bin, X_test=test_df) # TODO: repeat?
    stratified_preds = get_predictions(clf=stratified_clf, X_train=train_df, y_train=y, X_test=test_df) # TODO: repeat?
    stratified_emotions_bin = onehot_encoder.decode(stratified_preds_bin) # TODO: le labels sono i.i.d. potrebbe predirre tutti 0
    stratified_emotions = ordinal_encoder.inverse_transform(stratified_preds)
    if split == "dev":
        stratified_metrics_bin = compute_EMO_metrics(golds=test_df.emotion, predictions=stratified_emotions_bin)
        stratified_metrics = compute_EMO_metrics(golds=test_df.emotion, predictions=stratified_emotions)
        write_dict_to_json(stratified_metrics_bin, f"stratified_{split}_metrics_bin.json")
        write_dict_to_json(stratified_metrics, f"stratified_{split}_metrics.json")
    else:
        write_EMO_predictions(stratified_emotions_bin, f"stratified_{split}_bin_predictions_EMO.tsv")
        write_EMO_predictions(stratified_emotions, f"stratified_{split}_predictions_EMO.tsv")

    uniform_clf = DummyClassifier(strategy="uniform")
    uniform_preds_bin = get_predictions(clf=uniform_clf, X_train=train_df, y_train=y_bin, X_test=test_df) # TODO: repeat?
    uniform_emotions = onehot_encoder.decode(uniform_preds_bin)
    if split == "dev":
        uniform_metrics = compute_EMO_metrics(golds=test_df.emotion, predictions=uniform_emotions)
        write_dict_to_json(uniform_metrics, f"uniform_{split}_metrics.json")
    else:
        write_EMO_predictions(uniform_emotions, f"uniform_{split}_predictions_EMO.tsv")

TRAIN_DATA_PATH = "https://raw.githubusercontent.com/HLT-Ghisolfi-Leuzzi-Testa/WASSA-2023/main/datasets/WASSA23_essay_level_train_original.tsv"
DEV_DATA_PATH = "https://raw.githubusercontent.com/HLT-Ghisolfi-Leuzzi-Testa/WASSA-2023/main//datasets/WASSA23_essay_level_dev_preproc.tsv"
"""
DATA_PATH = "datasets/WASSA23_essay_level_preproc.tsv"
TEST_DATA_PATH = "datasets/WASSA23_essay_level_test_preproc.tsv"
"""

train_df = pd.read_csv(TRAIN_DATA_PATH, sep='\t')
dev_df = pd.read_csv(DEV_DATA_PATH, sep='\t')
apply_dummy_clfs(train_df, dev_df, "dev")
"""all_df = pd.read_csv(DATA_PATH, sep='\t')
test_df = pd.read_csv(TEST_DATA_PATH, sep='\t')
apply_dummy_clfs(all_df, test_df, "test")"""