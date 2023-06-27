import json
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from evaluation import calculatePRF_MLabel
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, accuracy_score, jaccard_score, 
    precision_recall_fscore_support
    )
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, LabelBinarizer
from torch.utils.data import Dataset
from transformers import EvalPrediction
from torchsummary import summary
from torchview import draw_graph
from bertviz import model_view

def print_model_summary(model, path):
    '''
    This function saves a textual summary of the pytorch model passed as input.

    :param model: pytorch model
    :param path: path where to save the summary
    '''

    model_summary = str(summary(model, dtypes=['torch.IntTensor']))
    file = open(path, "w")
    file.write(model_summary)
    file.close()

def plot_model_graph(model, input_data, path):
    '''
    This function saves the graph of the pytorch model passed as input.

    :param model: pytorch model
    :param input_data: input data to the model
    :param path: path where to save the graph (no need to specify the extension)
    '''
    model_graph = draw_graph(model, input_data=input_data)
    model_graph.visual_graph.render(filename=path)
    model_graph.visual_graph.view()

def plot_loss_curve(training_loss, validatin_loss, path, title):
    '''
    This function saves the plot of the training and validation loss curves.
    
    :param training_loss: list of training loss values
    :param validatin_loss: list of validation loss values
    :param path: path where to save the plot
    :param title: title of the plot
    '''
    
    sns.set(style='darkgrid')
    plt.rcParams["figure.figsize"] = (12,6)
    plt.plot(training_loss, 'b-o', label="Training")
    plt.plot(validatin_loss, 'r-o', label="Validation")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(title)
    plt.xticks(np.arange(1, len(training_loss) + 1, 1))
    plt.tight_layout()
    plt.savefig(path)
    plt.show()

def plot_attentions(input_str, model, tokenizer, title, path):
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
    plt.title(title)
    plt.tight_layout()
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
        token_type_ids = inputs['token_type_ids'] # 0 for first sentence, 1 for second
        attention = model(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=inputs.attention_mask).attentions # if not customed, need to set output_attentions=True
        sentence_b_start = token_type_ids[0].tolist().index(1)
    else:
        attention = model(input_ids).attentions
        sentence_b_start = None
    input_id_list = input_ids[0].tolist() # Batch index 0
    tokens = tokenizer.convert_ids_to_tokens(input_id_list)  
    if hide_delimiter_attn:
        for i, t in enumerate(tokens):
            if t in ("[SEP]", "[CLS]"):
                for layer_attn in attention:
                    layer_attn[0, :, i, :] = 0
                    layer_attn[0, :, :, i] = 0
    model_view(attention, tokens, sentence_b_start, display_mode=display_mode)

def plot_confusion_matrix(golds, predictions, path, title):
    '''
    This function plots the confusion matrix given gold and predicted labels.
    
    :param golds: list of gold labels
    :param predictions: list of predicted labels
    :param path: path where to save the plot
    :param title: title of the plot
    '''

    labels = np.unique(golds)
    cm_df = pd.DataFrame(confusion_matrix(golds, predictions, labels=labels),index=labels, columns=labels)
    plt.figure(figsize = (10,7))
    sns.heatmap(cm_df, annot=True, cmap="Blues", fmt='g')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.show()

def write_dict_to_json(dict, path):
    '''
    This function saves a dictionary to a json file.
    
    :param dict: dictionary to save
    :param path: path where to save the dictionary
    '''
    
    with open(path, 'w') as fp:
        json.dump(dict, fp)

def write_EMO_predictions(predictions, path):
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
    '''
    
    scores_val = calculatePRF_MLabel(golds, predictions)
    scores = {
    'micro_recall': scores_val[0],
    'micro_precision': scores_val[1],
    'micro_f': scores_val[2],
    'macro_recall': scores_val[3],
    'macro_precision': scores_val[4],
    'macro_F': scores_val[5],
    'accuracy': scores_val[6]
    }
    return scores

def compute_EMO_metrics_trainer(p: EvalPrediction):
    '''
    This function is called by Trainer to compute the metrics for the EMO task.

    :param p: EvalPrediction object
    :return: dictionary of metrics
    '''

    predictions = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions # TODO: ?
    golds = p.label_ids

    # NOTE: not needed if using multilabel
    # https://szuyuchu.medium.com/multi-label-text-classification-with-bert-52fa78eddb9
    # apply sigmoid on predictions
    # sigmoid = torch.nn.Sigmoid()
    # probs = sigmoid(torch.Tensor(predictions))

    # use a threshold to turn prediction into 0/1 values
    bin_predictions = np.where(predictions >= 0.5, 1, 0)
    # TODO: if no emotion is predicted, set the one with highest activation
    # for i, bin_pred in enumerate(bin_predictions):
    #     if np.all(bin_pred==0):
    #         bin_predictions[i][np.argmax(predictions[i])] = 1
    predictions = bin_predictions

    # compute metrics
    metrics = {}
    metrics['sklearn_accuracy'] = accuracy_score(y_true=golds, y_pred=predictions)
    metrics['roc_auc_micro'] = roc_auc_score(y_true=golds, y_score=predictions, average = 'micro')
    metrics['accuracy'] = jaccard_score(y_true=golds, y_pred=predictions, average='micro')
    prf_micro = precision_recall_fscore_support(y_true=golds, y_pred=predictions, average='micro')
    metrics['micro_precision'] = prf_micro[0]
    metrics['micro_recall'] = prf_micro[1]
    metrics['micro_f'] = prf_micro[2]
    prf_macro = precision_recall_fscore_support(y_true=golds, y_pred=predictions, average='macro')
    metrics['macro_precision'] = prf_macro[0]
    metrics['macro_recall'] = prf_macro[1]
    metrics['macro_f'] = prf_macro[2]
    return metrics

class EMODataset(Dataset):
    '''
    This class is used to create a pytorch dataset for the EMO task.
    '''

    def __init__(
        self,
        tokenizer,
        essay,
        targets,
        features=None, # additional numerical features (n_features for each sample)
        max_len=None
        ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.essay = essay
        self.targets = targets
        self.features = features

    def __len__(self):
        return len(self.essay)

    def __getitem__(self, index):
        essay = str(self.essay[index])
        inputs = self.tokenizer.encode_plus(
            text=essay,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
            return_token_type_ids=True
        )

        item = {
          'input_ids': inputs['input_ids'].flatten(),
          'attention_mask': inputs['attention_mask'].flatten(),
          'token_type_ids': inputs["token_type_ids"].flatten(),
          'labels': torch.FloatTensor(self.targets[index])
        }
        if self.features is not None:
          item['features'] = torch.FloatTensor(self.features[index])
        return item

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

        emotions = [emotion.split('/') for emotion in emotions]
        self.mlb.fit(emotions)

    def encode(self, emotions):
        '''
        This method one-hot encodes the emotions passed as parameters.

        :param emotions: list of emotions
        :return: numpy array with encoded emotions
        '''

        emotions = [emotion.split('/') for emotion in emotions]
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

    # decode non needed?

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