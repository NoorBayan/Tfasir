import os
import torch
from torch.utils.data import Dataset

class ExtSumDataset(Dataset):
  def __init__(self, data, gold_summaries):
    self.data = data
    self.gold_summaries = gold_summaries

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    x = self.data[idx]["article"]
    y = torch.tensor(self.data[idx]["abstract"]).float()
    z = self.gold_summaries[idx]["abstract"]
    return x,y,z