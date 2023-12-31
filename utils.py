import json
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from nrclex import NRCLex
from textblob import TextBlob
from evaluation import calculatePRF_MLabel, calculate_pearson
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, accuracy_score, jaccard_score, 
    precision_recall_fscore_support, mean_squared_error, mean_absolute_error
    )
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, LabelBinarizer
from torch.utils.data import Dataset
from transformers import EvalPrediction

EMP_LEXICON_PATH = "./lexicon/lexicon_EMP.csv"
HOPE_LEXICON_PATH = "./lexicon/hope.txt"

VAL_SIZE = 0.2

NRC_emotions = [
    'fear',
    'anger',
    'anticipation',
    'trust',
    'surprise',
    'positive',
    'negative',
    'sadness',
    'disgust',
    'joy'
]

our_emotions = [
    'fear',
    'anger',
    'surprise',
    'sadness',
    'disgust',
    'joy',
    'hope',
    'neutral'
]

def plot_metric_curve(
        values,
        epochs,
        metrics,
        title=None,
        path=None,
        ):
    '''
    This function saves the plot of the training metric curves.
    
    :param values: list of metric values
    :param epochs: list of epochs
    :param metrics: list of metric names
    :param title: title of the plot
    :param path: path where to save the plot
    '''
    
    sns.set(style='darkgrid')
    plt.rcParams["figure.figsize"] = (12,6)

    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'w', 'k']

    if len(epochs) == 1:
        for value, metric, color in zip(values, metrics, colors):
            plt.plot(epochs, value, f'{color}-o', label=metric)
    else:
        for epoch, value, metric, color in zip(epochs, values, metrics, colors):
            plt.plot(epoch, value, f'{color}-o', label=metric)
    
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    if title:
        plt.title(title)
    if path:
        plt.savefig(path)
    plt.show()

def plot_attentions(input_str, model, tokenizer, title=None, path=None):
    '''
    This function plots the attention weights for the string passed as parameter.

    :param input_str: string for which to plot the attention weights
    :param model: model
    :param tokenizer: tokenizer
    :param title: title of the plot
    :param path: path where to save the plot
    '''

    model_inputs = tokenizer(input_str, return_tensors="pt")
    with torch.no_grad():
        model_output = model(**model_inputs)
    tokens = tokenizer.convert_ids_to_tokens(model_inputs.input_ids[0])
    n_tokens = len(tokens)
    n_layers = len(model_output.attentions)
    n_heads = len(model_output.attentions[0][0])
    fig, axes = plt.subplots(n_layers, n_heads)
    fig.set_size_inches(18.5*2, 10.5*4)
    for layer in range(n_layers):
        for i in range(n_heads):
            axes[layer, i].imshow(model_output.attentions[layer][0, i])
            axes[layer][i].set_xticks(list(range(n_tokens)))
            axes[layer][i].set_xticklabels(labels=tokens, rotation="vertical")
            axes[layer][i].set_yticks(list(range(n_tokens)))
            axes[layer][i].set_yticklabels(labels=tokens)

            if layer == 5:
                axes[layer, i].set(xlabel=f"head={i}")
            if i == 0:
                axes[layer, i].set(ylabel=f"layer={layer}") 
    plt.subplots_adjust(wspace=0.3)
    plt.tight_layout()
    if title:
        plt.title(title)
    if path:
        plt.savefig(path)
    plt.show()

def plot_model_view(
        model,
        tokenizer,
        sentence_a,
        sentence_b=None,
        hide_delimiter_attn=False,
        display_mode="dark"):
    '''
    This function visualizes the attention weights produced by model on
    sentence_a. If sentence_b is provided, the two sentences are concatenated
    with separator token in between.
    '''
    
    inputs = tokenizer.encode_plus(
        sentence_a,
        sentence_b,
        return_tensors='pt',
        add_special_tokens=True)
    input_ids = inputs['input_ids']
    if sentence_b:
        token_type_ids = inputs['token_type_ids']
        attention = model(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=inputs.attention_mask).attentions # if not customed, need to set output_attentions=True
        sentence_b_start = token_type_ids[0].tolist().index(1)
    else:
        attention = model(input_ids).attentions
        sentence_b_start = None
    input_id_list = input_ids[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_id_list)  
    if hide_delimiter_attn:
        for i, t in enumerate(tokens):
            if t in ("[SEP]", "[CLS]"):
                for layer_attn in attention:
                    layer_attn[0, :, i, :] = 0
                    layer_attn[0, :, :, i] = 0
    model_view(attention, tokens, sentence_b_start, display_mode=display_mode)

def plot_confusion_matrix(golds, predictions, title=None, path=None):
    '''
    This function plots the confusion matrix given gold and predicted labels
    cosidering every label combination.
    
    :param golds: list of gold labels
    :param predictions: list of predicted labels
    :param path: path where to save the plot
    :param title: title of the plot
    '''

    labels = np.unique(golds)
    cm_df = pd.DataFrame(
        confusion_matrix(golds, predictions, labels=labels),
        index=labels,
        columns=labels)
    plt.figure(figsize = (10,7))
    sns.heatmap(cm_df, annot=True, cmap="Blues", fmt='g')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    if title:
        plt.title(title)
    if path:
        plt.savefig(path)
    plt.show()

def plot_confusion_matrix_per_emotions(
        gold_emotions,
        predicted_emotions,
        title = None,
        path = None):
    '''
    This function plots the confusion matrix given gold and predicted labels.

    :param gold_emotions: list of ground truth emotions
    :param predicted_emotions: list of predicted emotions
    :param title: title of the plot
    :param path: path where to save the plot
    '''
    
    label_names = sorted(our_emotions)

    cm = confusion_matrix(gold_emotions, predicted_emotions, labels=label_names)

    _, ax = plt.subplots()
    im = ax.imshow(cm, cmap='Blues')

    ax.set_xticks(np.arange(len(label_names)))
    ax.set_yticks(np.arange(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45)
    ax.set_yticklabels(label_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    for i in range(len(label_names)):
        for j in range(len(label_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="k")

    ax.figure.colorbar(im, ax=ax)

    plt.tight_layout()
    if title:
        plt.title(title)
    if path:
        plt.savefig(path)
    plt.show()

def plot_true_vs_predicted(golds, predictions, title=None, path=None):
    '''
    This function plots the ground true values vs the predicted values.

    :param golds: list of ground truth labels
    :param predictions: list of predicted labels
    :param title: title of the plot
    :param path: path where to save the plot
    '''

    xy = np.vstack([golds, predictions])
    kernel = gaussian_kde(xy)(xy)
    plt.scatter(golds, predictions, c=kernel, cmap="crest")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    xpoints = ypoints = plt.xlim()
    plt.plot(xpoints, ypoints)
    plt.tight_layout()
    if title:
        plt.title(title)
    if path:
        plt.savefig(path)
    plt.show()

def plot_abs_diff_emp(golds, predictions, title=None, path=None):
	'''
	This function plots the absolute difference between the true and predicted
	empathy and distress values.

	:param golds: a pandas dataframe with the true empathy (first column) and 
	distress values (second column)
	:param predictions: a pandas dataframe with the predicted empathy (first
	column) and distress values (second column)
	:param title: the title of the plot
	:param path: the path where to save the plot
	'''
	
	abs_diff = np.abs(golds - predictions)
	values = np.vstack([abs_diff[:,0], abs_diff[:,1]])
	kernel = gaussian_kde(values)(values)
	sns.scatterplot(x=abs_diff[:,0], y=abs_diff[:,1], c=kernel, cmap="crest")
	plt.xlabel("|True empathy - Predicted empathy|")
	plt.ylabel("|True distress - Predicted distress|")
	if title:
		plt.title(title)
	if path:
		plt.savefig(path)
	plt.plot()

def predict_emotions(label_encoder, results, neutral_zero):

  binarized_predictions = np.where(results >= 0.5, 1, 0)

  if not neutral_zero:
    for i, bin_pred in enumerate(binarized_predictions):
      if np.all(bin_pred==0):
        binarized_predictions[i][np.argmax(results[i])] = 1

  predicted_emotions = label_encoder.decode(binarized_predictions)
  return predicted_emotions

def flatten_logits(logits, threshold):
    '''
    This function flattens the logits passed as parameter using the specified 
    threshold.

    :param logits: logits to flatten
    :param threshold: threshold to use for flattening
    :return: flattened logits
    '''

    predictions = np.where(logits >= threshold, 1, 0)
    for i, pred in enumerate(predictions):
        if np.all(pred==0):
            predictions[i][np.argmax(logits[i])] = 1
    return predictions

def save_logits(logits, labels, path):
    '''
    This function saves the logits in the path passed as parameter. Logits are
    saved as a csv file using labels as colums.

    :param logits: logits to save
    :param labels: labels to use as columns
    :param path: path where to save the logits
    '''

    pd.DataFrame(logits, columns=labels).to_csv(path, index=False)

def ensemble_logits(logits_path_list):
    '''
    This function computes the mean of the logits in file file list
    passed as parameter.

    :param logits_path_list: list of paths to the logits files
    :return: dataframe with mean of logits
    '''

    logits = []
    labels = None
    for path in logits_path_list:
        logits_df = pd.read_csv(path)
        if labels is None:
            labels = logits_df.columns
        logits_df = logits_df[labels] # reorder columns
        logits.append(logits_df.to_numpy())
    mean_logits = np.mean(logits, axis=0)
    return pd.DataFrame(mean_logits, columns=labels)

def logits_to_predictions(logits, threshold):
    '''
    This function converts the dataframe of logits passed as parameter to a list
    of strings representing predictions. Logits are flattened using the
    specified threshold.

    :param logits: dataframe of logits
    :param threshold: threshold to use for flattening
    :return: list of predictions
    '''

    predictions_str = []
    predictions_bin = flatten_logits(logits, threshold)
    for pred in predictions_bin:
        predictions_str.append('/'.join([logits.columns[i] for i in np.where(pred==1)[0]]))
    return predictions_str

def dev_cross_val(train_set, dev_set, k, shuffle, seed):
    '''
    This function splits the dev_set datarame in k folds and returns a
    list of tuples. The first element of each tuple is a dataframe representing
    a train split, the second element is a dataframe representing a validation
    split.

    :param train_set: training set dataframe
    :param dev_set: dev set dataframe
    :param k: number of folds
    :param shuffle: whether to shuffle the data before splitting
    :param seed: seed for the random number generator
    :return: list of tuples with splitted dataframes (training dataframe in
    position 0, validation dataframe in position 1)
    '''

    splits = []
    splitter = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=seed)
    for train_idx, valid_idx in splitter.split(dev_set, dev_set['emotion']):
        train_split = pd.concat([train_set, dev_set.iloc[train_idx]], ignore_index = True)
        val_split = dev_set.iloc[valid_idx]
        splits.append((train_split, val_split))
    return splits

def split_train_val(train_df, val_size=0.2):
    '''
    This function splits the train dataframe in train and validation sets
    stratifying labels.

    :param train_df: training set dataframe
    :param val_size: size of the validation set
    :return: internal training and validation dataframes
    '''
    emotions = train_df['emotion'].unique().tolist()
    internal_train_df = pd.DataFrame()
    internal_val_df = pd.DataFrame()
    for emotion in emotions:
        emotion_df = train_df.loc[train_df['emotion']==emotion]
        if emotion_df.shape[0] < 2 : # if a class has a single sample it is added to the train set
            internal_train_df = pd.concat(
                [internal_train_df, emotion_df],
                ignore_index=True)
        else:
            t_df, v_df = train_test_split(
                emotion_df,
                test_size=val_size,
                stratify=emotion_df['emotion'],
                shuffle=True)
            internal_train_df = pd.concat(
                [internal_train_df, t_df],
                ignore_index=True)
            internal_val_df = pd.concat(
                [internal_val_df, v_df],
                ignore_index=True)

    return internal_train_df, internal_val_df

def write_dict_to_json(dict, path):
    '''
    This function saves a dictionary to a json file.
    
    :param dict: dictionary to save
    :param path: path where to save the dictionary
    '''
    
    with open(path, 'w') as fp:
        json.dump(dict, fp, indent = 1)

def write_predictions(predictions, path):
    '''
    This function saves the predictions of the EMO task to a tsv file.
    
    :param predictions: list of predictions
    :param path: path where to save the predictions
    '''
    
    df = pd.DataFrame(predictions)
    df.to_csv(path, index=False, header=False, sep='\t')

def compute_EMO_metrics(golds, predictions):
    '''
    This function computes the metrics for the EMO task.
    
    :param golds: list of gold emotions
    :param predictions: list of predicted emotions
    :return: dictionary of metrics
    '''
    
    scores_val = calculatePRF_MLabel(golds, predictions)
    scores = {
        'macro_f1': float(scores_val[5]),
        'micro_f1': float(scores_val[2]),
        'micro_jaccard': float(scores_val[6]),
        'macro_precision': float(scores_val[4]),
        'macro_recall': float(scores_val[3]),
        'micro_precision': float(scores_val[1]),
        'micro_recall': float(scores_val[0])
    }
    return scores

def compute_EMO_metrics_trainer(p: EvalPrediction):
    '''
    This function is called by Trainer to compute the metrics for the EMO task.

    :param p: EvalPrediction object
    :return: dictionary of metrics
    '''

    predictions = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    golds = p.label_ids

    # use a threshold to turn prediction into 0/1 values
    bin_predictions = np.where(predictions >= 0.5, 1, 0)
    # if no emotion is predicted, set the one with highest activation
    for i, bin_pred in enumerate(bin_predictions):
        if np.all(bin_pred==0):
            bin_predictions[i][np.argmax(predictions[i])] = 1
    predictions = bin_predictions

    metrics = {}

    prf_macro = precision_recall_fscore_support(y_true=golds, y_pred=predictions, average='macro')
    prf_micro = precision_recall_fscore_support(y_true=golds, y_pred=predictions, average='micro')

    metrics['macro_f1'] = prf_macro[2]
    metrics['micro_f1'] = prf_micro[2]
    metrics['micro_jaccard'] = jaccard_score(y_true=golds, y_pred=predictions, average='micro')
    metrics['macro_precision'] = prf_macro[0]
    metrics['macro_recall'] = prf_macro[1]
    metrics['micro_precision'] = prf_micro[0]
    metrics['micro_recall'] = prf_micro[1]
    metrics['sklearn_accuracy'] = accuracy_score(y_true=golds, y_pred=predictions)
    metrics['roc_auc_micro'] = roc_auc_score(y_true=golds, y_score=predictions, average = 'micro')
    return metrics

def compute_EMO_metrics_trainer_neutral(p: EvalPrediction):
    '''
    This function is called by Trainer to compute the metrics for the EMO task
    when encoding the emotion 'neutral' with zeros.

    :param p: EvalPrediction object
    :return: dictionary of metrics
    '''

    predictions = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    golds = p.label_ids

    # use a threshold to turn prediction into 0/1 values
    predictions = np.where(predictions >= 0.5, 1, 0)

    metrics = {}

    prf_macro = precision_recall_fscore_support(y_true=golds, y_pred=predictions, average='macro')
    prf_micro = precision_recall_fscore_support(y_true=golds, y_pred=predictions, average='micro')

    metrics['macro_f1'] = prf_macro[2]
    metrics['micro_f1'] = prf_micro[2]
    metrics['micro_jaccard'] = jaccard_score(y_true=golds, y_pred=predictions, average='micro')
    metrics['macro_precision'] = prf_macro[0]
    metrics['macro_recall'] = prf_macro[1]
    metrics['micro_precision'] = prf_micro[0]
    metrics['micro_recall'] = prf_micro[1]
    metrics['sklearn_accuracy'] = accuracy_score(y_true=golds, y_pred=predictions)
    metrics['roc_auc_micro'] = roc_auc_score(y_true=golds, y_score=predictions, average = 'micro')
    return metrics

def compute_EMP_metrics(golds, predictions):
    '''
    This function computes the metrics for the EMO task.

    :param golds: list of gold emotions
    :param predictions: list of predicted emotions
    :return: dictionary of metrics
    '''

    scores = {}
    if (predictions.shape[1] != 1):
        scores['empathy_pearson'] = float(calculate_pearson(golds[:,0], predictions[:,0]))
        scores['distress_pearson'] = float(calculate_pearson(golds[:,1], predictions[:,1]))
        scores['avg_pearson'] = float((scores['empathy_pearson']+ scores['distress_pearson']) / 2)
        scores['empathy_mse'] = float(mean_squared_error(golds[:,0], predictions[:,0]))
        scores['empathy_mae'] = float(mean_absolute_error(golds[:,0], predictions[:,0]))
        scores['distress_mse'] = float(mean_squared_error(golds[:,1], predictions[:,1]))
        scores['distress_mae'] = float(mean_absolute_error(golds[:,1], predictions[:,1]))
    else:
        scores['pearson'] = float(calculate_pearson(golds, predictions))
        scores['mse'] = float(mean_squared_error(golds, predictions))
        scores['mae'] = float(mean_absolute_error(golds, predictions))

    return scores

def compute_EMP_metrics_trainer(p: EvalPrediction):
    '''
    This function is called by Trainer to compute the metrics for the EMP task.

    :param p: EvalPrediction object
    :return: dictionary of metrics
    '''

    predictions = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    golds = p.label_ids

    metrics = {}
    if (predictions.shape[1] != 1):
        metrics['empathy_pearson'] = calculate_pearson(golds[:,0], predictions[:,0])
        metrics['distress_pearson'] = calculate_pearson(golds[:,1], predictions[:,1])  
        metrics['avg_pearson'] = (metrics['empathy_pearson']+ metrics['distress_pearson']) / 2
        metrics['empathy_mse'] = mean_squared_error(golds[:,0], predictions[:,0])
        metrics['empathy_mae'] = mean_absolute_error(golds[:,0], predictions[:,0])
        metrics['distress_mse'] = mean_squared_error(golds[:,1], predictions[:,1])
        metrics['distress_mae'] = mean_absolute_error(golds[:,1], predictions[:,1])
    else:
        metrics['pearson'] = calculate_pearson(golds, predictions)
        metrics['mse'] = mean_squared_error(golds, predictions)
        metrics['mae'] = mean_absolute_error(golds, predictions)    

    return metrics

def compute_metrics(golds, predictions, task):
    '''
    This function computes the metrics for the task passed as parameter.

    :param golds: list of gold labels
    :param predictions: list of predicted labels
    :param task: task for which to compute the metrics
    :return: dictionary of metrics
    '''

    if task == 'EMO':
        return compute_EMO_metrics(golds, predictions)
    
    if task == 'EMP':
        return compute_EMP_metrics(golds, predictions)
    
def read_NRC_lexicon_file(file_name):
    '''
    This function reads the NRC lexicon file and returns a dictionary with
    words as keys and values as values.

    :param file_name: path to the NRC lexicon file
    :return: dictionary with words as keys and values as values
    '''

    lexicon = {}
    with open(file_name, 'r') as file:
        for line in file:
            word, value = line.strip().split()
            lexicon[word] = value
    return lexicon

def hope_essay_frequency(essay, hope_lexicon):
    '''
    This function computes the hope frequency in the essay passed as parameter.

    :param essay: essay for which to compute the hope frequency
    :param hope_lexicon: hope lexicon
    :return: hope frequency
    '''

    hope_count = 0
    total_count = 0
    for word in essay.split():
        if word in hope_lexicon:
            total_count += 1
            hope_count += int(hope_lexicon[word])
    return (hope_count / total_count) if total_count>0 else 0

def generate_prompt(
        essay,
        article_id,
        gender,
        education,
        ethnicity,
        age,
        income,
        empathy,
        distress):
    '''
    This function generates the text prompt for the essay passed as parameter.

    :param essay: essay for which to generate the prompt
    :param article_id: id of the article the essay refers to
    :param gender: writer gender
    :param education: writer education level
    :param etnicity: ethnicity of the writer
    :param age: age of the writer
    :param income: income of the writer
    :param empathy: empathy level of the writer
    :param distress: distress level of the writer
    :return: biographic prompt, empathy prompt and emotion prompt
    '''

    if type(gender) == str:
        gender = float(gender)
    if type(education) == str:
        education = float(education)
    if type(ethnicity) == str:
        ethnicity = float(ethnicity)

    if gender == 1: gender_str = "male"
    else: gender_str = "female"

    if education == 1: education_str = "with less than a high-school diploma"
    elif education == 2: education_str = "with a high-school diploma"
    elif education == 3: education_str = "went to a technical/vocational school"
    elif education == 4: education_str = "went to college"
    elif education == 5: education_str = "with a two year associate degree"
    elif education == 6: education_str = "with a four year bachelor's degree"
    else: education_str = "postgradute or with a professional degree"

    if ethnicity == 1: ethnicity_str = " white"
    elif ethnicity == 2: ethnicity_str = " hispanic or latino"
    elif ethnicity == 3: ethnicity_str = " black or african american"
    elif ethnicity == 4: ethnicity_str = " native american or american indian"
    elif ethnicity == 5: ethnicity_str = " asian/pacific islander"
    else: ethnicity_str = ""

    text_prompt_bio = "An essay about the article {}, written by a {} years old{} {}, {}, with an income of {}$.".format(
        article_id,
        age, ethnicity_str,
        gender_str,
        education_str,
        income
        )
    
    text_prompt_emp = "The essay expresses {} empathy and {} distress levels.".format(
        empathy,
        distress
        )

    # compute a score for each emotion using the NRC lexicon
    obj = NRCLex(essay)
    hope_lexicon = read_NRC_lexicon_file(HOPE_LEXICON_PATH)
    our_emotions_scores = {}
    for emo in our_emotions:
        if emo != 'hope' and emo != 'neutral' and emo != 'fear':
            our_emotions_scores[emo] = obj.affect_frequencies[emo]
        if emo == 'fear':
            our_emotions_scores[emo] = obj.affect_frequencies[emo]*0.8 # assign less weight to fear lexicon
    our_emotions_scores['hope'] = hope_essay_frequency(essay, hope_lexicon)

    # get the emotion with the highest score
    max=0
    emo_string=""
    for key, value in our_emotions_scores.items():
        if value==max and max != 0:
            emo_string += ", " + key
        if value>max:
            max=value
            emo_string=key
    if max==0:
        emo_string = 'neutral'

    text_prompt_emo = " The top emotions expressed in the essay, according to the NRC lexicon, are: {}.".format(emo_string)
    
    return text_prompt_bio, text_prompt_emp, text_prompt_emo

def add_emp_dist_levels(df):
    '''
    This function adds the empathy and distress truth levels to the dataframe 
    passed as parameter.

    :param df: dataframe to which to add the empathy and distress levels
    :return: dataframe with empathy and distress levels
    '''

    targets = ['empathy', 'distress']
    df['true_empathy_level_3']=''
    df['true_distress_level_3']=''
    df['true_empathy_level_5']=''
    df['true_distress_level_5']=''

    for idx, _ in df.iterrows():
        for target in targets:
            if df.at[idx, target] < 3:
                df.at[idx, f'true_{target}_level_3'] = 'low'
            elif df.at[idx, target] < 5:
                df.at[idx, f'true_{target}_level_3'] = 'medium'
            else:
                df.at[idx, f'true_{target}_level_3'] = 'high'

            if df.at[idx, target] < 2.2:
                df.at[idx, f'true_{target}_level_5'] = 'low'
            elif df.at[idx, target] < 3.4:
                df.at[idx, f'true_{target}_level_5'] = 'low-medium'
            elif df.at[idx, target] < 4.6:
                df.at[idx, f'true_{target}_level_5'] = 'medium'
            elif df.at[idx, target] < 5.8:
                df.at[idx, f'true_{target}_level_5'] = 'medium-high'
            else:
                df.at[idx, f'true_{target}_level_5'] = 'high'
    return df

def get_emp_levels(test_df, empathy, distress):
    '''
    This function adds the empathy and distress levels passed as parameter to 
    the dataframe test_df.

    :param test_df: dataframe to which to add the empathy and distress levels
    :param empathy: empathy values
    :param distress: distress values
    :return: dataframe with empathy and distress levels
    '''

    test_df['empathy_level'] = ""
    test_df['distress_level'] = ""

    for idx, _ in test_df.iterrows():
        emp_value = empathy[idx]
        distress_value = distress[idx]

        if emp_value < 3:
            emp_level = 'low'
        elif emp_value < 5:
            emp_level = 'medium'
        else:
            emp_level = 'high'

        if distress_value < 3:
            dist_level = 'low'
        elif distress_value < 5:
            dist_level = 'medium'
        else:
            dist_level = 'high'
        
        test_df.at[idx, "empathy_level"] = emp_level
        test_df.at[idx, "distress_level"] = dist_level
    
    return test_df

def add_prompt_truth(df, TASK, EMP_levels=None):
    '''
    This function adds the task text prompt to the dataframe passed as parameter.

    :param df: dataframe to which to add the text prompt
    :param TASK: task for which to generate the text prompt
    :param EMP_levels: empathy levels to use for the EMP task ("3" or "5")
    :return: dataframe with text prompts
    '''

    if TASK == 'EMP':
        for idx, _ in df.iterrows():
            if EMP_levels == "3":
                emp_level = df['true_empathy_level_3'].loc[idx]
                dist_level = df['true_distress_level_3'].loc[idx]
                df.at[idx, 'prompt_emp'] = f'This essay expresses {emp_level} level of empathy and a {dist_level} level of distress'
            else:
                emp_level = df['true_empathy_level_5'].loc[idx]
                dist_level = df['true_distress_level_5'].loc[idx]
                df.at[idx, 'prompt_emp'] = f'This essay expresses {emp_level} level of empathy and a {dist_level} level of distress'
    else:
        for idx, row in df.iterrows():
            emotions_str = "neutral"
            if 'emotion' in row:        
                emotions_str = row['emotion'].replace("/", ", ")         
            df.at[idx, 'prompt_emo'] = f'This essay expresses {emotions_str}'
    return df

class EMPlexicon():
    '''
    This class is used to compute the empathy and distress values for each word
    in a sentence according to the EMP lexicon.
    '''

    def __init__(self):
        self.lexicon = pd.read_csv("./lexicon/lexicon_EMP.csv", sep=',')
        self.empathy_lexicon_dict = self.lexicon.set_index('word')['empathy'].to_dict()
        self.distress_lexicon_dict = self.lexicon.set_index('word')['distress'].to_dict()

    def load_token_list(self, token_list):
        '''
        This function loads a list of tokens and computes the empathy and distress
        values for each word in the list according to the EMP lexicon.

        :param token_list: list of tokens
        '''
        self.text = ""
        self.words = token_list
        self.sentences = []
        self.build_empathy_counter()

    def load_raw_text(self, text):
        '''
        This function loads a raw text and computes the empathy and distress
        values for each word in the text according to the EMP lexicon.

        :param text: raw text
        '''

        self.text = text
        blob = TextBlob(self.text)
        self.words = [w.lemmatize() for w in blob.words]
        self.sentences = list(blob.sentences)
        self.build_empathy_counter()

    def build_empathy_counter(self):
        '''
        This function computes the empathy and distress values for each word in
        the sentence according to the EMP lexicon.
        '''

        self.empathy_list = [] # list of empathy values for each word in sentence
        self.empathy_dict = dict() # dict of empathy values for each word in sentence

        self.distress_list = [] # list of distress values for each word in sentence
        self.distress_dict = dict() # dict of distress values for each word in sentence

        value_counts = np.zeros(2)
        self.lexicon_keys = self.lexicon['word'].tolist()

        low_emp_count = 0
        medium_emp_count = 0
        high_emp_count = 0
        low_dist_count = 0
        medium_dist_count = 0
        high_dist_count = 0

        for word in self.words:
            if word in self.lexicon_keys:
                self.empathy_list.append(self.empathy_lexicon_dict[word])
                self.empathy_dict.update({word: self.empathy_lexicon_dict[word]})

                self.distress_list.append(self.distress_lexicon_dict[word])
                self.distress_dict.update({word: self.distress_lexicon_dict[word]})

                if self.empathy_lexicon_dict[word] < 3:
                    low_emp_count += 1
                elif self.empathy_lexicon_dict[word] < 5:
                    medium_emp_count += 1
                else:
                    high_emp_count += 1

                if self.distress_lexicon_dict[word] < 3:
                    low_dist_count += 1
                elif self.distress_lexicon_dict[word] < 5:
                    medium_dist_count += 1
                else:
                    high_dist_count += 1

                value_counts[0] += self.empathy_dict[word]
                value_counts[1] += self.distress_dict[word]
            else:
                self.empathy_list.append(4)
                self.distress_list.append(0)

        # weigh counts to get better estimates
        high_emp_count *= 4
        high_dist_count *= 4
        low_emp_count *= 1.2
        low_dist_count *= 1.2

        if low_emp_count > medium_emp_count and low_emp_count > high_emp_count:
            emp_level = 'low'
        elif medium_emp_count > low_emp_count and medium_emp_count > high_emp_count:
            emp_level = 'medium'
        elif high_emp_count > low_emp_count and high_emp_count > medium_emp_count:
            emp_level = 'high'
        else:
            emp_level = 'medium'
        if low_dist_count > medium_dist_count and low_dist_count > high_dist_count:
            dist_level = 'low'
        elif medium_dist_count > low_dist_count and medium_dist_count > high_dist_count:
            dist_level = 'medium'
        elif high_dist_count > low_dist_count and high_dist_count > medium_dist_count:
            dist_level = 'high'
        else:
            dist_level = 'medium'
        
        # dict with mean values for empathy and distress over the sentence
        word_in_lexicon = (np.count_nonzero(self.empathy_list) if np.count_nonzero(self.empathy_list) > 0 else 1)
        self.empathy_sentence_mean = {
            'empathy': (value_counts[0] / word_in_lexicon),
            'empathy_level': emp_level,
            'distress': (value_counts[1] / word_in_lexicon),
            'distress_level': dist_level
        }

class FeaturesEncoder():
    '''
    This class is used to encode the additional features.
    '''

    def __init__(self):
        self.gender_encoder = LabelBinarizer() # disentagled
        self.race_encoder = LabelBinarizer() # disentagled
        self.education_encoder = LabelEncoder() # ordinal values

    def fit(self, dataframe):
        '''
        This function fits the encoder to the dataframe passed as parameter.

        :param dataframe: dataset dataframe
        '''
        if 'gender' in dataframe.columns:
            self.gender_encoder.fit(dataframe.gender)
            self.gender = True
        else:
            self.gender = False
        if 'race' in dataframe.columns:
            self.race_encoder.fit(dataframe.race)
            self.race = True
        else:
            self.race = False
        if 'education' in dataframe.columns:
            self.education_encoder.fit(dataframe.education)
            self.education = True
        else:
            self.education = False
        if 'age' in dataframe.columns:
            self.age = True
        else:
            self.age = False
        if 'income' in dataframe.columns:
            self.income = True
        else:
            self.income = False

    def encode(self, dataframe):
        '''
        This method encodes the dataframe passed as parameter.

        :param dataframe: dataset dataframe
        :return: numpy array with encoded features
        '''
        concat_features = None
        if self.gender:
            genders = self.gender_encoder.transform(dataframe.gender)
            concat_features = genders
        if self.education:
            educations = self.education_encoder.transform(dataframe.education).reshape(-1,1)
            if concat_features is None:
                concat_features = educations
            else:
                concat_features = np.concatenate((concat_features, educations), axis=1)
        if self.race:
            races = self.race_encoder.transform(dataframe.race)
            if concat_features is None:
                concat_features = races
            else:
                concat_features = np.concatenate((concat_features, races), axis=1)
        if self.age:
            ages = dataframe.age.to_numpy().reshape(-1,1)
            if concat_features is None:
                concat_features = ages
            else:
                concat_features = np.concatenate((concat_features, ages), axis=1)
        if self.income:
            incomes = dataframe.income.to_numpy().reshape(-1,1)
            if concat_features is None:
                concat_features = incomes
            else:
                concat_features = np.concatenate((concat_features, incomes), axis=1)
        return concat_features

class EmotionsLabelEncoder():
    '''
    This class is used to one-hot encode and decode the emotions labels.
    '''

    def __init__(self):
        self.mlb = MultiLabelBinarizer()

    def fit(self, emotions):
        '''
        This function fits the encoder to the emotions passed as parameters.

        :param emotions: list of emotions
        '''

        emotions = [emotion.lower().split('/') for emotion in emotions]
        self.mlb.fit(emotions)
        self.classes = self.mlb.classes_

    def encode(self, emotions):
        '''
        This method one-hot encodes the emotions passed as parameters.

        :param emotions: list of emotions
        :return: numpy array with encoded emotions
        '''

        emotions = [emotion.lower().split('/') for emotion in emotions]
        encoded_emotions = self.mlb.transform(emotions)
        return encoded_emotions

    def decode(self, encoded_emotions):
        '''
        This method decodes the one-hot encoded emotions passed as parameters.

        :param encoded_emotions: list of one-hot encoded emotions
        :return: strings list of decoded emotions
        '''

        labels = self.mlb.inverse_transform(np.array(encoded_emotions))
        emotions = ["/".join(emotion) for emotion in labels]
        return emotions
    
class EmotionsLabelEncoderNeutral():
    '''
    This class is used to one-hot encode and decode the emotions labels when
    neutral is not encoded as an emotion.
    '''

    def __init__(self):
        self.mlb = MultiLabelBinarizer()

    def fit(self, emotions):
        '''
        This function fits the encoder to the emotions passed as parameters.

        :param emotions: list of emotions
        '''

        emotions = [emotion.lower().split('/') for emotion in emotions]
        new_emotions = []
        for i in range(len(emotions)):
          if 'neutral' not in emotions[i]:
            new_emotions.append(emotions[i])
          elif 'neutral' in emotions[i] and len(emotions[i])>1:
            emotions[i].remove('neutral') # if an essay expresses neutral and other emotions, neutral is removed
            new_emotions.append(emotions[i])
        self.mlb.fit(new_emotions)
        self.classes = self.mlb.classes_

    def encode(self, emotions):
        '''
        This method one-hot encodes the emotions passed as parameters.

        :param emotions: list of emotions
        :return: numpy array with encoded emotions
        '''

        emotions = [emotion.lower().split('/') for emotion in emotions]
        encoded_emotions = self.mlb.transform(emotions)
        return encoded_emotions

    def decode(self, encoded_emotions):
        '''
        This method decodes the one-hot encoded emotions passed as parameters.

        :param encoded_emotions: list of one-hot encoded emotions
        :return: strings list of decoded emotions
        '''

        labels = self.mlb.inverse_transform(np.array(encoded_emotions))
        emotions = ["/".join(emotion) for emotion in labels]
        emotions = ["neutral" if emo == '' else emo for emo in emotions]
        return emotions
