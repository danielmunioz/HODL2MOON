from torch.utils.data import Dataset


class sst_binary_dataset(Dataset):
  def __init__(self, dataframe):
    self.df = dataframe

  def __len__(self):
    return len(self.df)
  
  def __getitem__(self, idx):
    return {'input_ids': self.df.iloc[idx]['input_ids'], 
            'attention_mask': self.df.iloc[idx]['attention_mask'],
            'label': self.df.iloc[idx]['label']}


class fp_dataset(Dataset):
  def __init__(self, dataframe):
    self.df = dataframe

  def __len__(self):
    return len(self.df)
    
  def __getitem__(self, idx): 
    return {'input_ids': self.df['input_ids'].iloc[idx],
            'attention_mask': self.df['attention_mask'].iloc[idx],
            'label': self.df['label'].iloc[idx]}


class Sentiment140(Dataset):
  def __init__(self, dataframe):
    self.df = dataframe

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    return {'input_ids': self.df['input_ids'].iloc[idx],
            'attention_mask': self.df['attention_mask'].iloc[idx],
            'label': self.df['label'].iloc[idx]}


class inference_dataset(Dataset):
  def __init__(self, dataframe):
    self.df = dataframe
  
  def __len__(self):
    return len(self.df)
  
  def __getitem__(self, idx):
    return {'input_ids': self.df['input_ids'].iloc[idx],
            'attention_mask': self.df['attention_mask'].iloc[idx]}


class df_dataset(Dataset):
  def __init__(self, dataframe, label_name=None):
    """
    Initializes a torch.dataset from a pandas dataframe containing: 'input_ids', 'attention_mask', 'label'

    Args:
      dataframe: A pandas dataframe
      label_name: str, name of the column containing the labels, defaults to 'label'
    """
    self.df = dataframe
    self.label_name = label_name
  
  def __len__(self):
    return len(self.df)
  
  def __getitem__(self, idx):
    return {'input_ids': self.df['input_ids'].iloc[idx],
            'attention_mask': self.df['attention_mask'].iloc[idx],
            'label': self.df[self.label_name].iloc[idx] if self.label_name else self.df['label'].iloc[idx]}