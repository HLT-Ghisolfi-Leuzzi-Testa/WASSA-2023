import json
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from evaluation import calculatePRF_MLabel, calculate_pearson
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, accuracy_score, jaccard_score, 
    precision_recall_fscore_support
    )
from scipy.stats import gaussian_kde
from sklearn.model_selection import KFold
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, LabelBinarizer
from torch.utils.data import Dataset
from transformers import EvalPrediction
from torchsummary import summary
from torchview import draw_graph
from bertviz import model_view
from textblob import TextBlob

EMP_LEXICON_PATH = "./lexicon/lexicon_EMP.csv"


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

def plot_loss_curve(train_loss, eval_loss, train_epochs, eval_epochs, title=None, path=None):
    '''
    This function saves the plot of the training and validation loss curves.
    
    :param training_loss: list of training loss values
    :param validatin_loss: list of validation loss values
    :param path: path where to save the plot
    :param title: title of the plot
    '''
    
    sns.set(style='darkgrid')
    plt.rcParams["figure.figsize"] = (12,6)
    plt.plot(train_epochs, train_loss, 'b-o', label="Training")
    plt.plot(eval_epochs, eval_loss,'r-o', label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks(np.arange(1, len(training_loss) + 1, 1))
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

def plot_confusion_matrix(golds, predictions, title=None, path=None):
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
    plt.scatter(golds, predictions, c=kernel)
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
    '''

    splits = []
    splitter = KFold(n_splits=k, shuffle=shuffle, random_state=seed)
    for train_idx, valid_idx in splitter.split(dev_set):
        train_split = pd.concat([train_set, dev_set.iloc[train_idx]])
        val_split = dev_set.iloc[valid_idx]
        splits.append((train_split, val_split))
    return splits

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

def compute_EMP_metrics(golds, predictions):
    '''
    This function computes the metrics for the EMO task.

    :param golds: list of gold emotions
    :param predictions: list of predicted emotions
    '''

    scores = {}
    if (len(predictions.shape) == 2):
      scores['empathy_pearson'] = calculate_pearson(golds[:,0], predictions[:,0])
      scores['distress_pearson'] = calculate_pearson(golds[:,1], predictions[:,1])
      scores['avg_pearson'] = (scores['empathy_pearson']+ scores['distress_pearson']) / 2
    else:
      scores['pearson'] = calculate_pearson(golds, predictions)

    return scores

def compute_EMP_metrics_trainer(p: EvalPrediction):

    predictions = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions # TODO: ?
    golds = p.label_ids

    # NOTE: not needed if using multilabel
    # https://szuyuchu.medium.com/multi-label-text-classification-with-bert-52fa78eddb9
    # apply sigmoid on predictions
    # sigmoid = torch.nn.Sigmoid()
    # probs = sigmoid(torch.Tensor(predictions))

    # use a threshold to turn prediction into 0/1 values
    #bin_predictions = np.where(predictions >= 0.5, 1, 0)
    # TODO: if no emotion is predicted, set the one with highest activation
    # for i, bin_pred in enumerate(bin_predictions):
    #     if np.all(bin_pred==0):
    #         bin_predictions[i][np.argmax(predictions[i])] = 1
    metrics = {}
    if (len(predictions.shape) == 2):
      metrics['empathy_pearson'] = calculate_pearson(golds[:,0], predictions[:,0])
      metrics['distress_pearson'] = calculate_pearson(golds[:,1], predictions[:,1])
      metrics['avg_pearson'] = (metrics['empathy_pearson']+ metrics['distress_pearson']) / 2
    else:
      metrics['pearson'] = calculate_pearson(golds, predictions)

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
          'token_type_ids': inputs["token_type_ids"].flatten()
        }
        if self.features is not None:
          item['features'] = torch.FloatTensor(self.features[index])
        if self.targets is not None:
            item['labels'] = torch.FloatTensor(self.targets[index])
        return item
    
def add_prompt_to_test_from_EMP_predictions(test_df, emp_predictions_path): #TODO: verificare che funzioni
    emp_predictions = pd.read_csv(emp_predictions_path, header=None)

    for idx, row in test_df.iterrows():
        if row['gender'] == 1: gender = "male"
        else: gender = "female"

        if row['education'] == 1: education = "less than a high school diploma"
        elif row['education'] == 2: education = "high School diploma"
        elif row['education'] == 3: education = "technical/vocational School"
        elif row['education'] == 4: education = "some college but no degree"
        elif row['education'] == 5: education = "two year associate degree"
        elif row['education'] == 6: education = "four year bachelor's degree"
        else: education = "postgradute or professional degree"

        if row['race'] == 1: ethnicity = "white"
        elif row['race'] == 2: ethnicity = "hispanic or latino"
        elif row['race'] == 3: ethnicity = "black or african american"
        elif row['race'] == 4: ethnicity = "native american or american indian"
        elif row['race'] == 5: ethnicity = "asian/pacific islander"
        else: ethnicity = ""

        text_prompt_bio = "An essay written by a {} years old {} {}, with {}, with an income of {}$.".format(
                                        row["age"], ethnicity, gender, education, row["income"]) 
        
        for line in emp_predictions[idx]:
            empathy = line.split('/')[0]
            distress = line.split('/')[1]

            if empathy < 3: emp = "The essay expresses low empathy"
            elif empathy < 5: emp = "The essay not expresses empathy"
            else: empathy = "The essay expresses high empathy"
            if distress < 3: dis = "low"
            elif distress < 5: dis = "medium"
            else: dis = "high"
            text_prompt_emp = " {} and {} distress level.".format(emp,  dis)

        emotions = NRCLex(row["essay"]).top_emotions
        if (sum(np.array([emo[1] for emo in emotions])))== 0:
           emotions = {'neutral': 0}
        n_emo = len(emotions)
        string = ""
        for i, emo in enumerate(emotions):
            string += emo[0]
            if i < n_emo-1:
                string += ", "
        text_prompt_emo = " Top emotions expressed by the writer are: {}.".format(string)

        text_prompt = row["essay"] + '"' + text_prompt_bio + text_prompt_emp + text_prompt_emo + '"'
        test_df["prompt"][idx] = text_prompt
        
    return test_df

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
        self.classes = self.mlb.classes_

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
    

class EMPlexicon():

    def __init__(self, lexicon_path=EMP_LEXICON_PATH):
        self.lexicon = pd.read_csv("./lexicon/lexicon_EMP.csv", sep=',')
        self.empathy_lexicon_dict = self.lexicon.set_index('word')['empathy'].to_dict()
        self.distress_lexicon_dict = self.lexicon.set_index('word')['distress'].to_dict()

    def load_token_list(self, token_list):
        self.text = ""
        self.words = token_list
        self.sentences = []
        self.build_empathy_counter()

    def load_raw_text(self, text):
        self.text = text
        blob = TextBlob(self.text)
        self.words = [w.lemmatize() for w in blob.words]
        self.sentences = list(blob.sentences)
        self.build_empathy_counter()

    def build_empathy_counter(self):
        self.empathy_list = [] # list of empathy values for each word in sentence
        self.empathy_dict = dict() # dict of empathy values for each word in sentence

        self.distress_list = [] # list of distress values for each word in sentence
        self.distress_dict = dict() # dict of distress values for each word in sentence

        value_counts = np.zeros(2)
        self.lexicon_keys = self.lexicon['word'].tolist()

        for word in self.words:
            if word in self.lexicon_keys:
                self.empathy_list.append(self.empathy_lexicon_dict[word])
                self.empathy_dict.update({word: self.empathy_lexicon_dict[word]})

                self.distress_list.append(self.distress_lexicon_dict[word])
                self.distress_dict.update({word: self.distress_lexicon_dict[word]})

                value_counts[0] += self.empathy_dict[word]
                value_counts[1] += self.distress_dict[word]
            else:
                self.empathy_list.append(0)
                self.distress_list.append(0)

        # dict with mean values for empathy and distress over the sentence   
        word_in_lexicon = (np.count_nonzero(self.empathy_list) if np.count_nonzero(self.empathy_list) > 0 else 1)
        self.empathy_sentence_mean = {'empathy': (value_counts[0] / word_in_lexicon), 
                                 'distress': (value_counts[1] / word_in_lexicon)}