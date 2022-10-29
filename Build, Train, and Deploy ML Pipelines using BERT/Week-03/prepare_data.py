################################################################################################################################################
######################################################## Import required modules ###############################################################
################################################################################################################################################
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import functools
import multiprocessing

from datetime import datetime
from time import gmtime, strftime, sleep

import pandas as pd
import argparse
import subprocess
import sys
import os
import re
import collections
import json
import csv
import glob
from pathlib import Path
import time
import boto3

subprocess.check_call([sys.executable, "-m", "conda", "install", "-c", "pytorch", "pytorch==1.6.0", "-y"])

subprocess.check_call([sys.executable, "-m", "conda", "install", "-c", "conda-forge", "transformers==3.5.1", "-y"])
from transformers import RobertaTokenizer

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'sagemaker==2.35.0'])
import sagemaker

from sagemaker.session import Session
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.feature_store.feature_definition import (
    FeatureDefinition,
    FeatureTypeEnum,
)

################################################################################################################################################
###################################################### Setup environmental variables ###########################################################
################################################################################################################################################

region = os.environ['AWS_DEFAULT_REGION']
sts = boto3.Session(region_name=region).client(service_name='sts', region_name=region)
iam = boto3.Session(region_name=region).client(service_name='iam', region_name=region)
featurestore_runtime = boto3.Session(region_name=region).client(service_name='sagemaker-featurestore-runtime', region_name=region)
sm = boto3.Session(region_name=region).client(service_name='sagemaker', region_name=region)

caller_identity = sts.get_caller_identity()
assumed_role_arn = caller_identity['Arn']
assumed_role_name = assumed_role_arn.split('/')[-2]
get_role_response = iam.get_role(RoleName=assumed_role_name) 
role = get_role_response['Role']['Arn']
bucket = sagemaker.Session().default_bucket()

sagemaker_session = sagemaker.Session(boto_session=boto3.Session(region_name=region), 
                            sagemaker_client=sm,
                            sagemaker_featurestore_runtime_client=featurestore_runtime)

# list of sentiment classes: -1 - negative; 0 - neutral; 1 - positive
classes = [-1, 0, 1]

# label IDs of the target class (sentiment) setup as a dictionary
classes_map = {
    -1: 0, 
    0: 1,
    1: 2
}

# tokenization model
PRE_TRAINED_MODEL_NAME = 'roberta-base'

# create the tokenizer to use based on pre trained model
tokenizer = RobertaTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

################################################################################################################################################
################################################################# Tools ########################################################################
################################################################################################################################################
# functions which can be considered as tools

def cast_object_to_string(data_frame):
    for label in data_frame.columns:
        if data_frame.dtypes[label] == 'object':
            data_frame[label] = data_frame[label].astype("str").astype("string")
    return data_frame
            

def wait_for_feature_group_creation_complete(feature_group):
    try:
        status = feature_group.describe().get("FeatureGroupStatus")
        print('Feature Group status: {}'.format(status))
        while status == "Creating":
            print("Waiting for Feature Group Creation")
            time.sleep(5)
            status = feature_group.describe().get("FeatureGroupStatus")
            print('Feature Group status: {}'.format(status))
        if status != "Created":
            print('Feature Group status: {}'.format(status))
            raise RuntimeError(f"Failed to create feature group {feature_group.name}")
        print(f"FeatureGroup {feature_group.name} successfully created.")
    except:
        print('No feature group created yet.')
    
def list_arg(raw_value):
    """argparse type for a list of strings"""
    return str(raw_value).split(',')

def to_sentiment(star_rating):
    if star_rating in {1, 2}: # negative
        return -1 
    if star_rating == 3:      # neutral
        return 0
    if star_rating in {4, 5}: # positive
        return 1

################################################################################################################################################
################################################### Create or load Feature Group ###############################################################
################################################################################################################################################

def create_or_load_feature_group(prefix, feature_group_name):

    # Feature Definitions for the records
    feature_definitions= [
        FeatureDefinition(feature_name='review_id', feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name='date', feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name='sentiment', feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name='label_id', feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name='input_ids', feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name='review_body', feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name='split_type', feature_type=FeatureTypeEnum.STRING)
    ]
    
    # setup the Feature Group
    feature_group = FeatureGroup(
        name=feature_group_name,
        feature_definitions=feature_definitions,
        sagemaker_session=sagemaker_session
    )
    
    print('Feature Group: {}'.format(feature_group))
    
    try:                
        print('Waiting for existing Feature Group to become available if it is being created by another instance in our cluster...')
        wait_for_feature_group_creation_complete(feature_group)
    except Exception as e:
        print('Before CREATE FG wait exeption: {}'.format(e))
        
    try:
        record_identifier_feature_name = "review_id"
        event_time_feature_name = "date"
        
        print('Creating Feature Group with role {}...'.format(role))
        
        # create Feature Group
        feature_group.create(
            s3_uri=f"s3://{bucket}/{prefix}",
            record_identifier_name=record_identifier_feature_name,
            event_time_feature_name=event_time_feature_name,
            role_arn=role,
            enable_online_store=False
        )
        print('Creating Feature Group. Completed.')
        
        print('Waiting for new Feature Group to become available...')
        wait_for_feature_group_creation_complete(feature_group)
        print('Feature Group available.')  
        
        # the information about the Feature Group
        feature_group.describe()
    
    except Exception as e:
        print('Exception: {}'.format(e))
        
    return feature_group

################################################################################################################################################
################################################### Tokenization of the reviews ################################################################
################################################################################################################################################

# Convert the review into the BERT input ids using 
def convert_to_bert_input_ids(review, max_seq_length):
    encode_plus = tokenizer.encode_plus(
          review,
          add_special_tokens=True,
          
          max_length=max_seq_length,
          
          return_token_type_ids=False,
          padding='max_length',
          return_attention_mask=True,
          return_tensors='pt',
          truncation=True
    )

    return encode_plus['input_ids'].flatten().tolist()

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
        print('/opt/ml/config/resourceconfig.json not found. current_host is unknown.')
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
    parser.add_argument('--output-data', type=str,
        default='/opt/ml/processing/output',
    )
    parser.add_argument('--train-split-percentage', type=float,
        default=0.90,
    )
    parser.add_argument('--validation-split-percentage', type=float,
        default=0.05,
    )    
    parser.add_argument('--test-split-percentage', type=float,
        default=0.05,
    )
    parser.add_argument('--balance-dataset', type=eval,
        default=True
    )    
    parser.add_argument('--max-seq-length', type=int, 
        default=128
    )
    parser.add_argument('--feature-store-offline-prefix', type=str,
        default=None,
    ) 
    parser.add_argument('--feature-group-name', type=str,
        default=None,
    ) 
    
    return parser.parse_args()

################################################################################################################################################
####################################################### Processing functions ###################################################################
################################################################################################################################################
    
def _preprocess_file(file, 
                     balance_dataset, 
                     max_seq_length,
                     prefix,                
                     feature_group_name):
    
    print('file {}'.format(file))
    print('balance_dataset {}'.format(balance_dataset))
    print('max_seq_length {}'.format(max_seq_length))
    print('prefix {}'.format(prefix))     
    print('feature_group_name {}'.format(feature_group_name))    

    # Create a feature group
    # the Feature Group that was set in the main notebook cannot be passed here - it will be used later in the notebook for other purposes
    # you need to create a Feature Group with the same Feature Definitions within the processing job
    feature_group = create_or_load_feature_group(prefix, feature_group_name)
    
    filename_without_extension = Path(Path(file).stem).stem

    # read file
    df = pd.read_csv(file, 
                     index_col=0)

    df.isna().values.any()
    df = df.dropna()
    df = df.reset_index(drop=True)
    print('Shape of dataframe {}'.format(df.shape))
    
    # convert star rating into sentiment
    df['sentiment'] = df['Rating'].apply(lambda star_rating: to_sentiment(star_rating=star_rating))
    print('Shape of dataframe with sentiment {}'.format(df.shape))

    # convert sentiment (-1, 0, 1) into label_id (0, 1, 2)
    df['label_id'] = df['sentiment'].apply(lambda sentiment: classes_map[sentiment])
    print('df[label_id] after using classes_map: {}'.format(df['label_id']))
    
    df['input_ids'] = df['Review Text'].apply(lambda review: convert_to_bert_input_ids(review, max_seq_length))
    print('df[input_ids] after calling convert_to_bert_input_ids: {}'.format(df['input_ids']))   
    
    # convert the index into a review_id
    df.reset_index(inplace=True)
    df = df.rename(columns = {'index': 'review_id', 
                              'Review Text': 'review_body'})
    
    # drop all columns except the following:
    df = df[['review_id', 'sentiment', 'label_id', 'input_ids', 'review_body']]
    df = df.reset_index(drop=True)

    print(df.head())
    
    print('Shape of dataframe after dropping columns {}'.format(df.shape))
    
    # balance the dataset by sentiment down to the minority class
    if balance_dataset:  
        
        df_unbalanced_grouped_by = df.groupby('sentiment')
        df_balanced = df_unbalanced_grouped_by.apply(lambda x: x.sample(df_unbalanced_grouped_by.size().min()).reset_index(drop=True))
        print('Shape of balanced df: {}'.format(df_balanced.shape))        
        
        print(df_balanced['sentiment'].head())

        df = df_balanced
        
    # adding timestamp as date column
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    print(timestamp)
        
    df['date'] = timestamp
    print('Shape of df with date: {}'.format(df.shape))    
    
    # split dataset
    print('Shape of dataframe before splitting {}'.format(df.shape))
    
    print('train split percentage {}'.format(args.train_split_percentage))
    print('validation split percentage {}'.format(args.validation_split_percentage))
    print('test split percentage {}'.format(args.test_split_percentage))    
    
    holdout_percentage = 1.00 - args.train_split_percentage
    print('holdout percentage {}'.format(holdout_percentage))
    df_train, df_holdout = train_test_split(df, 
                                            test_size=holdout_percentage, 
                                            stratify=df['sentiment'])

    test_holdout_percentage = args.test_split_percentage / holdout_percentage
    print('test holdout percentage {}'.format(test_holdout_percentage))
    df_validation, df_test = train_test_split(df_holdout, 
                                              test_size=test_holdout_percentage,
                                              stratify=df_holdout['sentiment'])
    
    df_train = df_train.reset_index(drop=True)
    df_validation = df_validation.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    print('Shape of train dataframe {}'.format(df_train.shape))
    print('Shape of validation dataframe {}'.format(df_validation.shape))
    print('Shape of test dataframe {}'.format(df_test.shape))

    train_data = '{}/sentiment/train'.format(args.output_data)
    validation_data = '{}/sentiment/validation'.format(args.output_data)
    test_data = '{}/sentiment/test'.format(args.output_data)
    
    ## write TSV Files
    df_train.to_csv('{}/part-{}-{}.tsv'.format(train_data, args.current_host, filename_without_extension), sep='\t', index=False)
    df_validation.to_csv('{}/part-{}-{}.tsv'.format(validation_data, args.current_host, filename_without_extension), sep='\t', index=False)
    df_test.to_csv('{}/part-{}-{}.tsv'.format(test_data, args.current_host, filename_without_extension), sep='\t', index=False)

    # dataframe
    df_train.head()   
    df_validation.head()   
    df_test.head()   

    column_names = ['review_id', 'sentiment', 'date', 'label_id', 'input_ids', 'review_body']
    
    df_train_records = df_train[column_names]
    df_train_records['split_type'] = 'train'
    df_train_records.head()   
    
    df_validation_records = df_validation[column_names]
    df_validation_records['split_type'] = 'validation'
    df_validation_records.head()   

    df_test_records = df_test[column_names]
    df_test_records['split_type'] = 'test'
    df_test_records.head()   

    # add record to feature store    
    df_fs_train_records = cast_object_to_string(df_train_records)
    df_fs_validation_records = cast_object_to_string(df_validation_records)
    df_fs_test_records = cast_object_to_string(df_test_records)

    print('Ingesting features...')
    feature_group.ingest(
        data_frame=df_fs_train_records, max_workers=3, wait=True
    )        
    feature_group.ingest(
        data_frame=df_fs_validation_records, max_workers=3, wait=True
    )        
    feature_group.ingest(
        data_frame=df_fs_test_records, max_workers=3, wait=True
    )
    
    offline_store_status = None
    while offline_store_status != 'Active':
        try:
            offline_store_status = feature_group.describe()['OfflineStoreStatus']['Status']
        except:
            pass
        print('Offline store status: {}'.format(offline_store_status))    
        sleep(15)
    print('...features ingested!')


def process(args):
    print('Current host: {}'.format(args.current_host))
    
    feature_group = create_or_load_feature_group(prefix=args.feature_store_offline_prefix,
                                                 feature_group_name=args.feature_group_name)

    feature_group.describe()
    
    preprocessed_data = '{}/sentiment'.format(args.output_data)
    train_data = '{}/sentiment/train'.format(args.output_data)
    validation_data = '{}/sentiment/validation'.format(args.output_data)
    test_data = '{}/sentiment/test'.format(args.output_data)
    
    # partial functions allow to derive a function with some parameters to a function with fewer parameters 
    # and fixed values set for the more limited function
    # here 'preprocess_file' will be more limited function than '_preprocess_file' with fixed values for some parameters
    preprocess_file = functools.partial(_preprocess_file,                 
                                        balance_dataset=args.balance_dataset,
                                        max_seq_length=args.max_seq_length,
                                        prefix=args.feature_store_offline_prefix,
                                        feature_group_name=args.feature_group_name)
    
    input_files = glob.glob('{}/*.csv'.format(args.input_data))

    num_cpus = multiprocessing.cpu_count()
    print('num_cpus {}'.format(num_cpus))

    p = multiprocessing.Pool(num_cpus)
    p.map(preprocess_file, input_files)

    print('Listing contents of {}'.format(preprocessed_data))
    dirs_output = os.listdir(preprocessed_data)
    for file in dirs_output:
        print(file)

    print('Listing contents of {}'.format(train_data))
    dirs_output = os.listdir(train_data)
    for file in dirs_output:
        print(file)

    print('Listing contents of {}'.format(validation_data))
    dirs_output = os.listdir(validation_data)
    for file in dirs_output:
        print(file)

    print('Listing contents of {}'.format(test_data))
    dirs_output = os.listdir(test_data)
    for file in dirs_output:
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