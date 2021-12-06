from torch import nn
from transformers import DistilBertModel, DistilBertConfig


class DistilBertRegressor(nn.Module):
  def __init__(self, backbone_weights=None, freeze_backbone=False):
    super().__init__()  
    self.backbone = DistilBertModel(config=DistilBertConfig())
    
    if backbone_weights:
      self.backbone.load_state_dict(backbone_weights)
      print('backbone weights loaded succesfully')
    if freeze_backbone:
      for param in self.backbone.parameters():
        param.requires_grad = False

    self.linear =  nn.Linear(self.backbone.config.dim, self.backbone.config.dim)  #'backbone.config.dim' stores the output dimension of the backbone
    self.dropout = nn.Dropout(0.2)
    self.regressor = nn.Linear(self.backbone.config.dim, 1)
  
  def forward(self, input_ids, attention_mask, return_hidden_embeddings=False):
    hidden_embeddings = self.backbone(input_ids=input_ids, attention_mask=attention_mask)[0]
    out = self.linear(hidden_embeddings[:, 0, :])
    out = F.relu(out)
    out = self.dropout(out)
    out = self.regressor(out)

    if not return_hidden_embeddings:
      return out
    else:
      return hidden_embeddings, out


class DistilBertClassifier(nn.Module):
  def __init__(self, out_dim=2, backbone_weights=None, freeze_backbone=False):
    super().__init__()  
    self.backbone = DistilBertModel(config=DistilBertConfig())
    
    if backbone_weights:
      self.backbone.load_state_dict(backbone_weights)
      print('backbone weights loaded succesfully')
    if freeze_backbone:
      for param in self.backbone.parameters():
        param.requires_grad = False

    self.linear =  nn.Linear(self.backbone.config.dim, self.backbone.config.dim)  #'backbone.config.dim' stores the output dimension of the backbone
    self.dropout = nn.Dropout(0.2)
    self.classifier = nn.Linear(self.backbone.config.dim, out_dim)
  
  def forward(self, input_ids, attention_mask, return_hidden_embeddings=False):
    hidden_embeddings = self.backbone(input_ids=input_ids, attention_mask=attention_mask)[0]
    out = self.linear(hidden_embeddings[:, 0, :])
    out = F.relu(out)
    out = self.dropout(out)
    out = self.classifier(out)

    if not return_hidden_embeddings:
      return out
    else:
      return hidden_embeddings, out