import os
import torch
from torch import nn
from transformers import AdamW, get_scheduler

from .testing import inner_testing


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
      #the name 'label' comes from the use of data_collator
      input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['label'].type(torch.LongTensor).to(device)
      optim.zero_grad()

      out = model(input_ids, attention_mask, return_hidden_embeddings=False)
      loss = loss_fn(out, labels)

      loss.backward()
      optim.step()
      lr_scheduler.step()

      epoch_loss+=loss.item()
      running_loss+=loss.item()
      
      if i!=0 and i%(save_every_iter-1) == 0 and save_failsafe:
        running_loss/=save_every_iter
        print('epoch: {}, iter: {}, running_loss: {}   --------saving model'.format(epoch, i, running_loss))

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
    print('storing "end of the epoch" weights epoch_loss: {}'.format(epoch_loss))
    torch.save({'epoch': epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optim.state_dict(),
                'lr_scheduler':lr_scheduler.state_dict(),
                'epoch_loss':epoch_loss},
                os.path.join(saving_dir, model_name)+'_epoch_{}.pt'.format(epoch))
     
    if test_dloader:
      testing_loss, testing_accuracy = inner_testing(model, test_dloader, device, keep_training=True)
      print('----Testing:  loss: {}, accuracy: {}'.format(testing_loss, testing_accuracy))

  train_loss, train_accuracy = inner_testing(model, data_loader, device)
  print('----After training metrics:  loss: {}, accuracy: {}'.format(train_loss, train_accuracy))
  return train_loss, train_accuracy