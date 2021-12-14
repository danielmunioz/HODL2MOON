import torch
import numpy as np
import pandas as pd


def tokenizer_map_dataset(element):
  return tokenizer(element['sentence'], max_length=tokenizer.model_max_length, truncation=True)


def tokenizer_map_df_simple(element):
  tokenized_elemen = tokenizer(element['sentence'], max_length=tokenizer.model_max_length, truncation=True)
  return element.append(pd.Series([tokenized_elemen['input_ids'], tokenized_elemen['attention_mask']], index=['input_ids', 'attention_mask']))


def str2arrray(elemen):
  #reads strings as arrays
  return np.fromstring(elemen[1:-1], sep=',', dtype=np.int64)


#### -----current version
class tokenizer_map_df():
  def __init__(self, tokenizer, column_name=None):
    '''
    Column name: the name of the column containing the text to process
    '''
    self.tokenizer = tokenizer
    self.column_name = column_name
  
  def __call__(self, element):
    if not self.column_name:
      tokenized_elemen = self.tokenizer(element['sentence'], max_length=self.tokenizer.model_max_length, truncation=True)
    else:
      tokenized_elemen = self.tokenizer(element[self.column_name], max_length=self.tokenizer.model_max_length, truncation=True)

    return element.append(pd.Series([tokenized_elemen['input_ids'], tokenized_elemen['attention_mask']], index=['input_ids', 'attention_mask']))


### -----current version
class Data_collator():
  """
  Data collator designed to work with a 'df_Dataset'
  Assumes a list of dictionaries as input
  """
  def __init__(self, tokenizer): 
    self.tokenizer = tokenizer

  def __call__(self, batch):
    training_dict = {'input_ids': [element['input_ids'] for element in batch], 
                     'attention_mask': [element['attention_mask'] for element in batch]}
    training_samples = self.tokenizer.pad(training_dict, return_tensors = 'pt')
    label = torch.tensor([element['label'] for element in batch], dtype=torch.float32)

    return {'input_ids': training_samples['input_ids'],
            'attention_mask': training_samples['attention_mask'],
            'label': label}