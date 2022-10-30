################################################################################################################################################
######################################################## Import required modules ###############################################################
################################################################################################################################################

import functools
import multiprocessing
import logging
from datetime import datetime
import subprocess
import sys
subprocess.check_call([sys.executable, '-m', 'conda', 'install', '-c', 'pytorch', 'pytorch==1.6.0', '-y'])
import torch
from torch import nn
subprocess.check_call([sys.executable, '-m', 'conda', 'install', '-c', 'conda-forge', 'transformers==3.5.1', '-y'])
from transformers import RobertaModel, RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'matplotlib==3.2.1'])
import pandas as pd
import os
import re
import collections
import argparse
import json
import os
import numpy as np
import csv
import glob
from pathlib import Path
import tarfile
import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


################################################################################################################################################
########################################################### Tools and variables ################################################################
################################################################################################################################################

MODEL_NAME = 'pytorch_model.bin'

PRE_TRAINED_MODEL_NAME = 'roberta-base'

classes = [-1, 0, 1]

# Load Hugging Face Tokenizer
tokenizer = RobertaTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)  

def list_arg(raw_value):
    """argparse type for a list of strings"""
    return str(raw_value).split(',')

################################################################################################################################################
###################################################### SageMaker load model function ###########################################################
################################################################################################################################################  

# You need to put in config.json from saved fine-tuned Hugging Face model in code/ 
# Reference it in the inference container at /opt/ml/model/code
def model_fn(model_dir):
    model_path = '{}/{}'.format(model_dir, MODEL_NAME) 
    model_config_path = '{}/{}'.format(model_dir, 'config.json')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = RobertaConfig.from_json_file(model_config_path)
    model = RobertaForSequenceClassification.from_pretrained(model_path, config=config)
    model.to(device)
    return model

################################################################################################################################################
####################################################### Encoding of the reviews ################################################################
################################################################################################################################################

def encode_review(review_body, max_seq_length):
    encode_plus_token = tokenizer.encode_plus(
            review_body,
            max_length=max_seq_length,
            add_special_tokens=True,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True)
    
    input_ids = encode_plus_token['input_ids']
    attention_mask = encode_plus_token['attention_mask']

    return input_ids, attention_mask

################################################################################################################################################
######################################################## SageMaker predict function ############################################################
################################################################################################################################################

def predict_fn(input_data, model, max_seq_length):
    model.eval()
    
    data_str = input_data.decode('utf-8')
    
    jsonlines = data_str.split("\n")

    predicted_classes = []

    for jsonline in jsonlines:
        # features[0]:  review_body
        # features[1..n]:  is anything else (we can define the order ourselves)
        # Example:  
        #    {"features": ["This is good."]}        
        #
        review_body = json.loads(jsonline)["features"][0]

        input_ids, attention_mask = encode_review(review_body, max_seq_length)

        output = model(input_ids, attention_mask)

        # output is a tuple: 
        # output: (tensor([[-1.9840, -0.9870,  2.8947]], grad_fn=<AddmmBackward>),
        # for torch.max() you need to pass in the tensor, output[0]   
        _, prediction = torch.max(output[0], dim=1)

        predicted_class_idx = prediction.item()
        
        predicted_class = classes[predicted_class_idx]

        prediction_dict = {}
        prediction_dict['predicted_label'] = predicted_class

        jsonline = json.dumps(prediction_dict)

        predicted_classes.append(jsonline)

    predicted_classes_jsonlines = '\n'.join(predicted_classes)

    return predicted_classes_jsonlines

################################################################################################################################################
###################################################### Parse input arguments ###################################################################
################################################################################################################################################

def parse_args():
    # Unlike SageMaker training jobs (which have `SM_HOSTS` and `SM_CURRENT_HOST` env vars), processing jobs to need to parse the resource config file directly
    resconfig = {}
    try:
        with open('/opt/ml/config/resourceconfig.json', 'r') as cfgfile:
            resconfig = json.load(cfgfile)
    except FileNotFoundError:
        print('/opt/ml/config/resourceconfig.json not found.  current_host is unknown.')
        pass # Ignore

    # Local testing with CLI args
    parser = argparse.ArgumentParser(description='Process')

    parser.add_argument('--hosts', type=list_arg,
        default=resconfig.get('hosts', ['unknown']),
        help='Comma-separated list of host names running the job'
    )
    parser.add_argument('--current-host', type=str,
        default=resconfig.get('current_host', 'unknown'),
        help='Name of this host running the job'
    )
    parser.add_argument('--input-data', type=str,
        default='/opt/ml/processing/input/data',
    )
    parser.add_argument('--input-model', type=str,
        default='/opt/ml/processing/input/model',
    )
    parser.add_argument('--output-data', type=str,
        default='/opt/ml/processing/output',
    )
    parser.add_argument('--max-seq-length', type=int,
        default=64,
    )  
    
    return parser.parse_args()

################################################################################################################################################
####################################################### Processing function ####################################################################
################################################################################################################################################
        
def process(args):
    print('Current host: {}'.format(args.current_host))
    
    print('input_data: {}'.format(args.input_data))
    print('input_model: {}'.format(args.input_model))          
          
    print('Extracting model.tar.gz')
    model_tar_path = '{}/model.tar.gz'.format(args.input_model)                
    model_tar = tarfile.open(model_tar_path)
    model_tar.extractall(args.input_model)
    model_tar.close()                                
    model_path = '{}/transformer/'.format(args.input_model)
    model = model_fn(model_path)

    def predict(text):
        input_data = json.dumps({'features': [text]}).encode('utf-8')
        response = predict_fn(input_data, model, args.max_seq_length)
        response_json = json.loads(response)
        return int(response_json['predicted_label'])
    
    print(predict("""I loved it!  I will recommend this to everyone."""))
    print(predict("""It's OK."""))
    print(predict("""Really bad.  I hope they don't make this anymore."""))

    print('Listing contents of input data dir: {}'.format(args.input_data))
    input_files = glob.glob('{}/*.tsv'.format(args.input_data))
    print('input_files: {}'.format(input_files))
    
    df_test_reviews = pd.DataFrame(columns=['sentiment', 'review_body'])
    df_test_reviews.head()
    
    for file in input_files:
        file_path = os.path.join(args.input_data, file)
        print(file_path)
        
        df_temp = pd.read_csv(file_path,
                              sep='\t', 
                              usecols=['sentiment', 'review_body'])
        df_test_reviews = df_test_reviews.append(df_temp)
        df_test_reviews.head()

    df_test_reviews = df_test_reviews #.sample(n=2000)
    df_test_reviews.shape
    df_test_reviews.head()

    y_test = df_test_reviews['review_body'].map(predict)
    y_actual = df_test_reviews['sentiment'].astype('int64')

    print(classification_report(y_true=y_test, y_pred=y_actual))

    accuracy = accuracy_score(y_true=y_test, y_pred=y_actual)        
    print('Test accuracy: ', accuracy)

    def plot_conf_mat(cm, classes, title, cmap):
        print(cm)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
            horizontalalignment="center",
            color="black" if cm[i, j] > thresh else "black")

            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')

    cm = confusion_matrix(y_true=y_test, y_pred=y_actual)

    plt.figure()
    fig, ax = plt.subplots(figsize=(10,5))
    plot_conf_mat(cm, 
                  classes=classes, 
                  title='Confusion Matrix',
                  cmap=plt.cm.Greens)

    # Save the confusion matrix        
    plt.show()

    # Model Output         
    metrics_path = os.path.join(args.output_data, 'metrics/')
    os.makedirs(metrics_path, exist_ok=True)
    plt.savefig('{}/confusion_matrix.png'.format(metrics_path))

    report_dict = {
        "metrics": {
            "accuracy": {
                "value": accuracy,
            },
        },
    }

    evaluation_path = "{}/evaluation.json".format(metrics_path)
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
        
    print('Listing contents of output dir: {}'.format(args.output_data))
    output_files = os.listdir(args.output_data)
    for file in output_files:
        print(file)

    print('Listing contents of output/metrics dir: {}'.format(metrics_path))
    output_files = os.listdir('{}'.format(metrics_path))
    for file in output_files:
        print(file)

    print('Complete')
    
################################################################################################################################################
#################################################################### Main ######################################################################
################################################################################################################################################
    
if __name__ == "__main__":
    args = parse_args()
    print('Loaded arguments:')
    print(args)
    
    print('Environment variables:')
    print(os.environ)

    process(args)    

