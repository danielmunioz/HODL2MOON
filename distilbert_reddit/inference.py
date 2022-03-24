import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast

from .datasets import inference_dataset
from .utils import Data_collator, tokenizer_map_df


def inference(model, dframe, column, device, return_probs=True, batch_size=32):
  tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
  collate_fn = Data_collator(tokenizer, using_labels=False)
  mapper = tokenizer_map_df(tokenizer, column_name=column)
  dframe = dframe.apply(mapper, axis=1)

  dset = inference_dataset(dframe) #using inf_dset
  dloader = DataLoader(dset, batch_size=batch_size, collate_fn=collate_fn)


  soft_tensor = torch.Tensor([]).to(device)
  preds_tensor = torch.Tensor([]).to(device)
  model.eval()
  with torch.no_grad():
    for i, elemen in enumerate(dloader):
      inputs, att_mask = elemen['input_ids'].to(device), elemen['attention_mask'].to(device)

      out = model(inputs, att_mask)
      soft_out = F.softmax(out, dim=1)
      probs, preds = soft_out.topk(1, dim=1)

      soft_tensor = torch.concat((soft_tensor, soft_out), dim=0)
      preds_tensor = torch.concat((preds_tensor, preds), dim=0)


  dframe = dframe.drop(columns=['input_ids', 'attention_mask'])
  dframe['predictions'] = preds_tensor.cpu().numpy().astype(int)
  if return_probs:
    dframe = pd.concat([dframe.reset_index(), pd.DataFrame(soft_tensor.cpu().numpy())], axis=1)

  return dframe