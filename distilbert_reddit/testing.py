import torch
import torch.nn.functional as F

def inner_testing(model, data_loader, device, loss_fn=F.cross_entropy, keep_training=True):
  """
  Designed to test a model inside a training loop.
  Recomended to use a loss from the torch.nn.funcional Module

  Returns loss and accuracy as float values.
  """
  loss = 0.0
  accuracy = 0.0
  model.eval()
  with torch.no_grad():
    for _, batch in enumerate(data_loader):
      input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].type(torch.LongTensor).to(device)
      out_model = model(input_ids, attention_mask)
      batch_loss = loss_fn(out_model, labels)

      probs = F.softmax(out_model, dim=1) #may wanna modify this to account regressors
      vals, idx = probs.topk(1, dim=1)
      acc = sum(idx.squeeze() == labels) / len(idx.squeeze())

      loss+=batch_loss.item()
      accuracy+=acc.item()

  loss/=len(data_loader)
  accuracy/=len(data_loader)
  if keep_training:
    model.train()
    
  return loss, accuracy