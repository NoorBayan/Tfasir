from transformers import BertConfig, BertModel, BertTokenizer, AutoModel

def load_bert():
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  #model = AutoModel.from_pretrained("google/bert_uncased_L-4_H-512_A-8")
  model = AutoModel.from_pretrained("bert-base-uncased")
  #print(model)

  for param in model.parameters():
    param.requires_grad = False

  for param in model.encoder.layer[11].parameters():
    param.requires_grad = True

  for param in model.encoder.layer[10].parameters():
    param.requires_grad = True

  for param in model.pooler.parameters():
    param.requires_grad = True
  
  return tokenizer, model
