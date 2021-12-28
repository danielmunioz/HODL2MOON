import os
import torch
import argparse
import pandas as pd
from collections import OrderedDict

from torch import nn
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler, DistilBertTokenizerFast

from .utils import Data_collator, tokenizer_map_df, split_dataframe
from .models import DistilBertClassifier
from .testing import inner_testing
from .datasets import df_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--backbone_weights_dir', metavar='PATH', type=str, required=True, help='path to the backbone weights')
parser.add_argument('--dataset_dir', metavar='PATH', type=str, required=True, help='path to the dataset file')
parser.add_argument('--model_name', metavar='NAME', type=str, default='FP_Distilbert', help='name of the model')
parser.add_argument('--save_dir', metavar='PATH', type=str, default='./', help='path to save the model')
#args=parser.parse_args() may wanna try using this while importing a something from here


def main():
  args=parser.parse_args()
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

  checkpoint_backbone = torch.load(args.backbone_weights_dir, map_location=device)    #loading weights
  #may wanna add the option to load weights with different names
  checkpoint_backbone['model'] = OrderedDict({k[11:]: v for k, v in checkpoint_backbone['model'].items() if 'distilbert' in k})

  financial_phrasebank_train(checkpoint_backbone['model'], data_frame_dir=args.dataset_dir, 
                             saving_dir = args.save_dir, model_name=args.model_name)


def financial_phrasebank_train(backbone_weights, train_dloader=None, test_dloader=None, dataframe_dir=None, 
                               saving_dir='./', model_name='FP_Model', return_loss_acc=False):
  
  
  if not train_dloader and not dataframe_dir:
    raise RuntimeError('Must provide either train_dloader or dataframe_dir to proceed')

  if dataframe_dir:
    fp_dataframe = pd.read_csv(dataframe_dir, sep='@', names=['sentence', 'label'], engine='python')
    label_mapping = {'negative':0, 'neutral':1, 'positive':2}
    fp_dataframe['label'] = fp_dataframe['label'].map(label_mapping)
    

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased') #using ver. 4.10.0.dev0
    collate_fn = Data_collator(tokenizer)
    tokenizer_map = tokenizer_map_df(tokenizer, column_name='sentence')
    tokenized_df = fp_dataframe.apply(tokenizer_map, axis=1)

    train, test, _= split_dataframe(tokenized_df)                                #uses the default split data values (for now)


    train_dset = df_dataset(train, label_name='label')
    test_dset = df_dataset(test, label_name='label')  
    train_dloader = DataLoader(train_dset, shuffle=True, batch_size=16, collate_fn=collate_fn)
    test_dloader = DataLoader(test_dset, shuffle=True, batch_size=16, collate_fn=collate_fn)


  model = DistilBertClassifier(backbone_weights=backbone_weights, out_dim=3).to(device)

  train_loss, train_acc = training_distilbert(model, train_dloader, device, test_dloader=test_dloader, save_every_iter=30, 
                                              saving_dir=saving_dir, model_name=model_name)

  #not using eval yet, may be a better to create an evaluation function
  #val_dset = df_dataset(val, label_name='label') 
  #val_dloader = DataLoader(val_dset, shuffle=True, batch_size=16, collate_fn=collate_fn)
  if return_loss_acc:
    return train_loss, train_acc



def training_distilbert(model, data_loader, device, test_dloader=None, epochs=3, model_optimizer='AdamW', learning_rate=1e-5, loss_function=nn.CrossEntropyLoss(), 
                        scheduler='linear', warmup_steps=600, save_every_iter=None,
                        saving_dir='./', model_name='distilbert_model'):
  """
  Trains a DistilBert-based model, for classification only (for now)

  Args:
    model: A pytorch model
    data_loader: A pytorch Dataloader, used for training
    device: A torch.device object indicating the device where the training will be taking place
    epochs: Int, number of training Epochs
    testd_loader: A pytorch Dataloader, used for testing, if None the testing phase is skipped

    model_optimizer: Optional, a pytorch optimizer, 'AdamW' by default
    learning_rate: Optional, float number, specifies the learning rate of the model
    loss_function: Optional, defaults to 'CrossEntropyLoss()'
    scheduler: Optional, defaults to 'Linear', 'get_scheduler' input parameter (from the transformers library)
    warmup_steps: Optional, defaults to 600

    save_every_iter: Optional, num of iterations after wich a checkpoint of the model will be saved
    saving dir: str, directory to save the model into, defaults to current dir
    model_name: str, name used to save the model

  Returns:
    train_loss: A float number describing the total loss of the model
    train_accuracy: A float number describing the total accuracy of the model over the training set
  """

  model.train()
  loss_fn = loss_function
  optim = AdamW(model.parameters(), lr=learning_rate) if model_optimizer == 'AdamW' else model_optimizer(model.parameters(), lr=learning_rate)

  num_epochs = epochs
  num_training_steps = num_epochs * len(data_loader)
  lr_scheduler = get_scheduler(scheduler, optimizer=optim, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)
  
  save_failsafe = True if save_every_iter else False
  save_every_iter = save_every_iter if save_every_iter else 0
    
  for epoch in range(num_epochs):
    epoch_loss = 0.0
    running_loss = 0.0    
    for i, batch in enumerate(data_loader):
      #'label' comes from data_collator
      input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['label'].type(torch.LongTensor).to(device)
      optim.zero_grad()

      out = model(input_ids, attention_mask, return_hidden_embeddings=False)
      loss = loss_fn(out, labels)

      loss.backward()
      optim.step()
      lr_scheduler.step()

      epoch_loss+=loss.item()
      running_loss+=loss.item()
      #should probably add running accuracy
      if i!=0 and i%(save_every_iter-1) == 0 and save_failsafe:
        running_loss/=save_every_iter
        print('epoch: {}, iter: {}, running_loss: {:.4f}   --------saving model'.format(epoch, i, running_loss))

        torch.save({'epoch': epoch,
                    'model_state_dict':model.state_dict(),
                    'optimizer_state_dict':optim.state_dict(),
                    'lr_scheduler':lr_scheduler.state_dict(),
                    'running_loss':running_loss},
                    os.path.join(saving_dir, model_name)+'_epoch_{}_iter_{}.pt'.format(epoch, i))
        running_loss = 0.0
      
      #this is supposed to display how the training is going after a certain number of iterations
      #if i!=0 and i%1000==0:
      #  print('epoch: {} iter: {} last_batch_loss: {}'.format(epoch, i, loss.item()))

    epoch_loss = epoch_loss/len(data_loader)
    print('storing "end of the epoch" weights, epoch_loss: {:.4f}'.format(epoch_loss))
    torch.save({'epoch': epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optim.state_dict(),
                'lr_scheduler':lr_scheduler.state_dict(),
                'epoch_loss':epoch_loss},
                os.path.join(saving_dir, model_name)+'_epoch_{}.pt'.format(epoch))
     
    if test_dloader:
      testing_loss, testing_accuracy = inner_testing(model, test_dloader, device, keep_training=True)
      print('----Test Set:  loss: {:.4f}, accuracy: {:.4f}'.format(testing_loss, testing_accuracy))

  train_loss, train_accuracy = inner_testing(model, data_loader, device)
  print('----Train set Metrics:  loss: {:.4f}, accuracy: {:.4f}'.format(train_loss, train_accuracy))
  return train_loss, train_accuracy


if __name__ == '__main__': 
  main()