import torch
from torch import nn
from model.AttentivePooling import AttentivePooling
from model.PositionalEncoding import PositionalEncoding
import numpy as np
  
class HiExtSumm(nn.Module):
    def __init__(self, model, tokenizer):
        super(HiExtSumm, self).__init__()
        self.bertEnc = model
        self.bertTokenizer = tokenizer
        #self.attn1 = nn.MultiheadAttention(embed_dim = 512, num_heads = 8)
        self.hidden_size = 768
        self.pos_encoder = PositionalEncoding(self.hidden_size)
        self.documentEncoderLayer = nn.TransformerEncoderLayer(d_model = self.hidden_size, nhead = 8, dim_feedforward = 2048, dropout=0.1)
        self.documentEncoder = nn.TransformerEncoder(self.documentEncoderLayer, num_layers = 2)
        self.linear1 = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)
        self.linear2 = nn.Linear(self.hidden_size * 2, 1)
        self.attnPooling = AttentivePooling(self.hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # max_len = []
        # for i in range(len(x[0])):
        #   is_max_length = True
        #   for j in range(len(x)):
        #     if (x[j][i] == ""):
        #       max_len.append(j)
        #       is_max_length = False
        #       break

        #   if (is_max_length == True):
        #     max_len.append(len(x))
          
        #print(max_len)
        x2 = self.encode(x)
        #print(tokenizer.decode(x[23]["input_ids"][1]))
        
        y = torch.ones(len(x2), len(x2[0]["input_ids"]), self.hidden_size)
        #memory = 0
        #for loop has to go
        for i in range(len(x2)):
          y[i] = (self.bertEnc(input_ids = x2[i]["input_ids"], attention_mask = x2[i]["attention_mask"], output_attentions = False, output_hidden_states = False).last_hidden_state)[:,0,:]
          #memory += y[i].element_size() * y[i].nelement()/1000000

        y = y.to("cuda")
        # attn_mask = torch.ones(y.size()[1], y.size()[0]).to("cuda")
        # for i in range(y.size()[1]):
        #   for j in range(max_len[i], y.size()[0]):
        #     attn_mask[i][j] = 0

        #y = self.pos_encoder(y)
        y = self.pos_encoder(torch.transpose(y,0,1))
        y = torch.transpose(y,0,1)
        #maybe add batch norm here. 
  
        #y = self.documentEncoder(y, src_key_padding_mask=attn_mask)
        y = self.documentEncoder(y) # y_shape is (max_no_of_sentences, batch_size, hidden_size)
        # print("y shape: ",y.shape)

        globalDocVec = self.attnPooling(y) #globalDocVec shape is (batch_size, hidden_size)
        #globalDocVec = torch.zeros((y.shape[1], y.shape[2])).to("cuda")
        # print("globalDocVec shape: ", globalDocVec.shape)

        temp = torch.zeros(y.shape).to("cuda")
        for k in range(temp.shape[0]):
          temp[k] = globalDocVec
        z = torch.cat((y, temp), 2)
        # print("Y shape: ", y.shape)
        # print("Z shape: ", z.shape)
        #z = y + globalDocVec 

        z = self.linear1(z)
        z = self.dropout(z)
        z = self.linear2(z)

        z = np.squeeze(z,2)
        z = self.sigmoid(z)
        z = torch.transpose(z,0,1)
   
        return z,torch.transpose(y,0,1) #z_shape is (batch_size, max_no_of_sentences)


    def encode(self, x):
      for i in range(len(x)):
        x[i] = self.bertTokenizer(x[i], padding=True, truncation=True, return_tensors="pt")
        x[i]["input_ids"] = x[i]["input_ids"].to("cuda")
        x[i]["attention_mask"] = x[i]["attention_mask"].to("cuda")
      
      return x
