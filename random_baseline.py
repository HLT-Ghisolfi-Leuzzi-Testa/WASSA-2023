import pandas as pd
from utils import EmotionsLabelEncoder, compute_metrics, write_dict_to_json, write_predictions
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

def get_predictions(clf, X_train, y_train, X_test):
    clf.fit(X_train, y_train)
    return clf.predict(X_test)

def apply_dummy_clfs(train_df, test_df, task): # TODO: per class metrics?
    
    if task == "EMO":
        ordinal_encoder = LabelEncoder()
        y = ordinal_encoder.fit_transform(train_df.emotion)
        onehot_encoder = EmotionsLabelEncoder()
        onehot_encoder.fit(train_df.emotion)
        y_bin = onehot_encoder.encode(train_df.emotion)
        golds = test_df.emotion
    else:
        y = np.array(train_df[['empathy', 'distress']])
        golds = np.array(test_df[['empathy', 'distress']])

    mostfreq_clf = DummyClassifier(strategy="most_frequent")
    mostfreq_preds = get_predictions(clf=mostfreq_clf, X_train=train_df, y_train=y, X_test=test_df)
    if task == "EMO": mostfreq_preds = ordinal_encoder.inverse_transform(mostfreq_preds)

    stratified_clf = DummyClassifier(strategy="stratified")
    stratified_preds = get_predictions(clf=stratified_clf, X_train=train_df, y_train=y, X_test=test_df)
    if task == "EMO": stratified_preds = ordinal_encoder.inverse_transform(stratified_preds)

    uniform_clf = DummyClassifier(strategy="uniform")
    if task == "EMO": 
        uniform_preds = get_predictions(clf=uniform_clf, X_train=train_df, y_train=y_bin, X_test=test_df) # TODO: repeat?
        uniform_preds = onehot_encoder.decode(uniform_preds)
    else:
        uniform_preds = get_predictions(clf=uniform_clf, X_train=train_df, y_train=y, X_test=test_df) # TODO: repeat?

    mostfreq_metrics = compute_metrics(golds=golds, predictions=mostfreq_preds, task = task)
    write_dict_to_json(mostfreq_metrics, f"{task}_mostfreq_metrics.json")
   
    stratified_metrics = compute_metrics(golds=golds, predictions=stratified_preds, task = task)
    write_dict_to_json(stratified_metrics, f"{task}_stratified_metrics_bin.json")

    uniform_metrics = compute_metrics(golds=golds, predictions=uniform_preds, task = task)
    write_dict_to_json(uniform_metrics, f"{task}_uniform_metrics.json")

if __name__ == "__main__":
    print("random baselines")
    TRAIN_DATA_PATH = "https://raw.githubusercontent.com/HLT-Ghisolfi-Leuzzi-Testa/WASSA-2023/main/datasets/WASSA23_essay_level_train_original.tsv"
    DEV_DATA_PATH = "https://raw.githubusercontent.com/HLT-Ghisolfi-Leuzzi-Testa/WASSA-2023/main//datasets/WASSA23_essay_level_dev_preproc.tsv"

    train_df = pd.read_csv(TRAIN_DATA_PATH, sep='\t')
    dev_df = pd.read_csv(DEV_DATA_PATH, sep='\t')
    apply_dummy_clfs(train_df, dev_df, "EMO")
    apply_dummy_clfs(train_df, dev_df,"EMP")