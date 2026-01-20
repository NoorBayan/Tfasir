import torch
from torch import nn

class AttentivePooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size) #linear takes (batch_size, in_features) and outputs (batch_size, out_features)
        self.linear2 = nn.Linear(250, 1)
        self.tanh = nn.Tanh()
        #self.queryVec = nn.Parameter(torch.normal(0,1,size=(hidden_size,))) #shape = (in_features, out_features)
        self.softmax = nn.Softmax(1)
        self.dropout = nn.Dropout(0.1)

    def forward(self,x):
        # x_shape is (max_no_of_sentences, batch_size, hidden_size)
        x = self.linear(x)
        x = self.dropout(x)
        x = self.tanh(x)
        x = torch.transpose(x,0,1) # (batch_size, max_no_of_sentences, hidden_size)
        tem = torch.transpose(x,1,2)
        queryVec = self.linear2(tem).squeeze(2)
        # print("X shape: ", x.shape)
        # print("Queryvec shape: ", queryVec.shape)

        temp = torch.zeros((x.shape[0], x.shape[1])).to("cuda")
        for i in range(x.shape[0]):
          temp[i] = torch.matmul(x[i], queryVec[i])

        #temp = torch.matmul(x, queryVec) # (batch_size, max_no_of_sentences)
        attn_weights = self.softmax(temp)
        #print("Attn_weights sum: ", torch.sum(attn_weights, 1))

        combinedRep = torch.zeros(x.shape[0], x.shape[2]).to("cuda")
        #print(combinedRep.shape)
        #print(x[:,0,:].shape)
        for i in range(x.shape[1]):
          formatted_attn_weights = torch.transpose(torch.unsqueeze(attn_weights[:,i],0),0,1)
          combinedRep += formatted_attn_weights * x[:,i,:]
        
        #print("Query Vec in Attentive Pooling: ", self.queryVec)
        return combinedRep
