import pandas as pd
from pandas import DataFrame
import numpy as np 
import traceback
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import scipy.spatial

#Global vairable for cache purpose
model = None

#EDA and Basic Preprocessing

def map_labels(df: DataFrame):
    '''
        This function takes in the dataframe and maps the labels to their meanings. 
        This is to allow for comprehensive EDA.

        Args:
            - @param: df: the dataframe whose label column is to be updated
        
        Returns:
            - df with column labels_text to show the meaning of the labels. 
    '''

    label_map = {
        0: 'False',
        1: 'Mixture',
        2: 'True', 
        3: 'Unproven', 
        -1: np.nan

    }

    df['labels_text'] = df['label'].apply(lambda x: label_map[x]) 

    return df


def replace_double_quotatons(df: DataFrame, columns: list):
    '''
        Remove double quotation marks which occur in the text to make it look like proper grammar. 
        This only removes quotation marks which occur in succession without anything in between

     Args:
            - @param: df: the dataframe which needs to be cleaned
            - @param: columns: list, the columns which need to have the double quotation marks removed
        
        Returns:
            - df with double quotation marks removed
        
    
    '''

    try:
        for column in columns: 
            df[column] = df[column].apply(lambda x: x.replace('""', ''))
    
        return df
    except:
        print("Column not found", traceback.format_exc())




def create_dataframe(data: dict): 
    '''

    This functions takes in a datalist from the datasets library, and converts them to dataframes for EDA  and further processing. 

    Arguments:
        - @param data: dict containing all the data from datasets module 
    
    Returns:
        - df: DataFrames, one for train, test, validation
    
    '''

    dataframes = [] #List to store all the dataframes. Order followed: train, test, validation
    for key in data.keys():
        dataframes.append(DataFrame(data[key]))

    return dataframes


def eda_preprocess_dataframe(df: DataFrame):
    '''
        This is a parent function that calls other functions to preprocess the dataframe. 

        Args:
            - @param: df: the dataframe which is to be processed
        
        Returns:
            - df with basic preprocessing done
    
    '''

    if 'label' in df.columns:
        df = map_labels(df)
    
    df = replace_double_quotatons(df, ['main_text', 'explanation', 'claim'])
    df = split_date_into_columns(df, 'date_published')
  
    return df


def split_date_into_columns(df: DataFrame, column: str):
    '''
        This is a function which takes in a column and converts it to datetime, then creates columns for Month and Year
        Args:
            - @param: df: the dataframe which has the column
            - @param: column: str, the column name which is to be split into months and year while being converted to datetime objects

        Returns:
            - df: DataFrame with updated columns

    '''
    try:
        df[column] = df[column].apply(lambda x: pd.to_datetime(x, errors='coerce'))
        #Get month and year from datetime object
        df[f'{column}_month'] = df[column].apply(lambda x: x.month if x else np.nan)
        df[f'{column}_year'] = df[column].apply(lambda x: x.year if x else np.nan)

        return df
    
    except: 
        print(traceback.format_exc())



# Preparing Data for Bert

def split_into_sentences(sentences: str):
    '''
        This function takes in the paragraph and splits it into sentences. 

        - Args:
            - @param: sentences, str, paragraph which needs to be split
        
        - Returns:
            - sentences_list: A list of sentences
    '''

    sentences_list = sent_tokenize(sentences)

    return sentences_list


def get_top_k_similar_sentences(match_sentence: str, sentence_list: list, k: int): 
    '''
        This function is used to get the top k similar sentences for a sentence and a list of sentences

        - Args:
            - @param: match_sentence, str, The sentence with which similarity will be measured
            - @param: sentence_list, list, the list of sentences from the main_text
            - @param: k, int, the number of top matched sentences to return based on cosine similarity

        - return
            - k_sentences, a list with sentences which are the most similiar to the match sentence
    
    '''

    model = load_model()

    sentence_list_embeddings = model.encode(sentence_list)
    match_sentence_embedding = model.encode(match_sentence)
    
    distances = scipy.spatial.distance.cdist([match_sentence_embedding], sentence_list_embeddings, "cosine")[0]

    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])

    k_sentences = []
    for index, distance in results [:k]: 
        k_sentences.append(sentence_list[index])

    return k_sentences










def load_model(reset=False):
    #This function has been created to cache the model so as to not reload them in various functions
    global model

    if reset==True or model==None:
        model = SentenceTransformer('paraphrase-mpnet-base-v2')

    return model 