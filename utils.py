import json
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from evaluation import calculatePRF_MLabel
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
from transformers import BertForSequenceClassification, EvalPrediction

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

def write_dict_to_json(dict, path):
    '''
    This function saves a dictionary to a json file.
    
    :param dict: dictionary to save
    :param path: path where to save the dictionary
    '''
    
    with open(path, 'w') as fp:
        json.dump(dict, fp)

def write_EMO_prediction(predictions, path):
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
    predictions = np.where(predictions >= 0.5, 1, 0) # TODO: works with tensors?

    # compute metrics
    metrics = compute_EMO_metrics(golds=golds, predictions=predictions) # TODO: testa...
    metrics['accuracy'] = accuracy_score(y_true=golds, y_pred=predictions)
    metrics['roc_auc_micro'] = roc_auc_score(y_true=golds, y_pred=predictions, average = 'micro')
    
    return metrics

# TODO: sistemare restore del modello
"""def save_best_weights(model, MODEL_NAME):
  torch.save(model.state_dict(), 'checkpoints/model_weights_'+MODEL_NAME+'.pt')

def restore_model(MODEL_NAME, device, num_labels=8): #TODO: MODELLO auto
  model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
  model.load_state_dict(torch.load('checkpoints/model_weights_'+MODEL_NAME+'.pt'))
  model.to(device)
  return model"""

class EMODataset(Dataset):
    '''
    This class is used to create a pytorch dataset for the EMO task.
    '''

    def __init__(
        self,
        tokenizer,
        essay,
        targets,
        max_len=None
        ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.essay = essay
        self.targets = targets

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
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'token_type_ids': inputs["token_type_ids"].flatten(),
            'label': torch.FloatTensor(self.targets[index])
        }

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