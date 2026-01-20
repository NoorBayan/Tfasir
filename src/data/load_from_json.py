from datasets import load_dataset
from datasets import concatenate_datasets

def load_data_from_json(path, splits=1):
  if (splits == 1):
    data = load_dataset("json",data_files=path, split="train")
  else:
    for i in range(splits):
      temp_path =  path[0:len(path) - 5] + str(i + 1) + path[len(path) - 5:len(path)]
      temp_data = load_dataset("json",data_files=temp_path, split="train")
      if (i != 0):
        data = concatenate_datasets([data, temp_data])
      else:
        data = temp_data
  
  return data

def load_all_ext_sum_data(train_path, val_path, test_path, train_splits, val_splits, test_splits):
  train_data = load_data_from_json(train_path, train_splits)
  val_data = load_data_from_json(val_path, val_splits)
  test_data = load_data_from_json(test_path, test_splits)
  return train_data, val_data, test_data

def load_gold_sum(train_path, val_path, test_path):
  train_data = load_dataset("json", data_files=train_path, split="train")
  val_data = load_dataset("json", data_files=val_path, split="train")
  test_data = load_dataset("json", data_files=test_path, split="train")
  return train_data, val_data, test_data
