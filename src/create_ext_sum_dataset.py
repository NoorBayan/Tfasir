from data.preprocess_dataset import get_raw_data, split_raw_data_into_sentences
from utils.rouge import _get_ngrams, cal_rouge
import argparse
import json
import re
import gzip
import shutil
from datasets import load_dataset
import copy


def get_extractive_summary(split_gold_summary, split_article, max_no_of_sentences):
  max_sentences_in_sum = 10
  extractive_sum = [0] * max_no_of_sentences
  rouge_1_best = 0
  rouge_2_best = 0
  count = 0
  a = True
  split_gold_summary = [(re.sub(r'[^\w\s]', '', i)).lower() for i in split_gold_summary]
  split_article = [(re.sub(r'[^\w\s]', '', i)).lower() for i in split_article]

  gold_sum_1_gram = [_get_ngrams(1, i.split()) for i in split_gold_summary]
  gold_sum_1_gram = set.union(*map(set, gold_sum_1_gram))
  gold_sum_2_gram = [_get_ngrams(2, i.split()) for i in split_gold_summary]
  gold_sum_2_gram = set.union(*map(set, gold_sum_2_gram))

  article_sum_1_gram = [_get_ngrams(1, i.split()) for i in split_article]
  article_sum_2_gram = [_get_ngrams(2, i.split()) for i in split_article]
  selected = []

  while (a == True and count < max_sentences_in_sum):
    count += 1
    a = False
    best = -1
    for i in range(max_no_of_sentences):
      if (extractive_sum[i] == 0 and split_article[i] != ""):
        candidate_sum = selected + [i]
        candidate_1_gram = [article_sum_1_gram[j] for j in candidate_sum]
        candidate_1_gram = set.union(*map(set, candidate_1_gram))
        rouge_1_f = cal_rouge(gold_sum_1_gram, candidate_1_gram)["f"]

        candidate_2_gram = [article_sum_2_gram[j] for j in candidate_sum]
        candidate_2_gram = set.union(*map(set, candidate_2_gram))
        rouge_2_f = cal_rouge(gold_sum_2_gram, candidate_2_gram)["f"]

        if (rouge_1_f + rouge_2_f > rouge_1_best + rouge_2_best):
          best = i
          rouge_1_best = rouge_1_f
          rouge_2_best = rouge_2_f
          a = True
        

    if (best != -1):
      selected += [best]
  
  for i in selected:
    extractive_sum[i] = 1

  return extractive_sum, rouge_1_best, rouge_2_best, count

def ext_sum_data_builder(data, splits, path, max_no_of_sentences):
  split_data = copy.copy(data)
  data_size = len(split_data)
  avg_rouge1_score = 0
  avg_rouge2_score = 0
  avg_count = 0
  split_count = 0
  data_saved_till = 0

  for i in range(data_size):
    split_data[i]["abstract"], rouge1_score, rouge2_score, count = get_extractive_summary(split_data[i]["abstract"], split_data[i]["article"], max_no_of_sentences)
    avg_rouge1_score += rouge1_score
    avg_rouge2_score += rouge2_score
    avg_count += count
    if (i % (int(data_size/splits)) == 0 and split_count != splits - 1 and i != 0):
      split_count += 1
      temp_path = path[0:len(path) - 5] + str(split_count) + path[len(path) - 5:len(path)]
      create_json_dataset(split_data[data_saved_till:i],temp_path)
      data_saved_till = i

  if splits == 1:
    create_json_dataset(split_data[data_saved_till:data_size],path)
  else:
    split_count += 1
    temp_path = path[0:len(path) - 5] + str(split_count) + path[len(path) - 5:len(path)]
    create_json_dataset(split_data[data_saved_till:data_size],temp_path)

  avg_rouge1_score = avg_rouge1_score/data_size
  avg_rouge2_score = avg_rouge2_score/data_size
  avg_count = avg_count/data_size
  print(avg_rouge1_score)
  print(avg_rouge2_score)
  print(avg_count) 


def create_json_dataset(ext_sum_data, path):
  temp = ext_sum_data
  with open(path, "w") as outfile:
      json.dump(temp, outfile)

def compress_json_file(json_path, json_compressed_path):
  with open(json_path, 'rb') as f_in, gzip.open(json_compressed_path, 'wb') as f_out:
      shutil.copyfileobj(f_in, f_out)

def create_list_of_gold_summaries(split_data):
  list_of_gold_summaries = []
  for i in range(len(split_data)):
    temp = {"index":i, "abstract": split_data[i]["abstract"]}
    list_of_gold_summaries.append(temp)
  
  return list_of_gold_summaries

def create_gold_summaries_dataset(split_train_data, split_val_data, split_test_data, train_path, val_path, test_path):
  list_of_gold_summaries_train = create_list_of_gold_summaries(split_train_data)
  list_of_gold_summaries_val = create_list_of_gold_summaries(split_val_data) 
  list_of_gold_summaries_test = create_list_of_gold_summaries(split_test_data) 

  create_json_dataset(list_of_gold_summaries_train, train_path)
  create_json_dataset(list_of_gold_summaries_val, val_path)
  create_json_dataset(list_of_gold_summaries_test, test_path)

def create_ext_summ_dataset(max_no_of_sentences, split_train_data, split_val_data, split_test_data, train_path, val_path, test_path, train_splits, val_splits, test_splits):
  ext_sum_data_builder(split_train_data, train_splits, train_path, max_no_of_sentences)
  ext_sum_data_builder(split_val_data, val_splits, val_path, max_no_of_sentences)
  ext_sum_data_builder(split_test_data, test_splits, test_path, max_no_of_sentences)

def arg_parser():
  parser = argparse.ArgumentParser(description = 'Script to create extractive summarization dataset')
  parser.add_argument('--dataset_path', type=str, default="ccdv/arxiv-summarization", help="Path of dataset repo in Hugginface or path to dataset in local system")
  parser.add_argument("--file_extension", type=str, default=None, help="File format of data files. To be specified only for dataset in local system")
  parser.add_argument('--max_no_of_sentences', type=int, default=250, help="The maximum number of sentences selected from each article for the model input")
  parser.add_argument('--extractive_summ_train_path', type=str, default="../data/extractive_summ/train.json", help="Path where training data of articles and its extractive summaries will be stored")
  parser.add_argument('--extractive_summ_val_path', type=str, default="../data/extractive_summ/validation.json", help="Path where validation data of articles and its extractive summaries will be stored")
  parser.add_argument('--extractive_summ_test_path', type=str, default="../data/extractive_summ/test.json", help="Path where test data of articles and its extractive summaries will be stored")
  parser.add_argument('--gold_summ_train_path', type=str, default="../data/gold_summ/gold_summaries_train.json", help="Path where training data of gold summaries will be stored")
  parser.add_argument('--gold_summ_val_path', type=str, default="../data/gold_summ/gold_summaries_validation.json", help="Path where training data of gold summaries will be stored")
  parser.add_argument('--gold_summ_test_path', type=str, default="../data/gold_summ/gold_summaries_test.json", help="Path where training data of gold summaries will be stored")
  parser.add_argument('--train_splits', type=int, default=1, help="number of splits to store train data, i.e if splits=2 training data will be stored in 2 files")
  parser.add_argument('--val_splits', type=int, default=1, help="number of splits to store validation data, i.e if splits=2 validation data will be stored in 2 files")
  parser.add_argument('--test_splits', type=int, default=1, help="number of splits to store test data, i.e if splits=2 test data will be stored in 2 files")
  return parser
  

def main():
  parser = arg_parser()
  args = parser.parse_args()
  dataset_path = args.dataset_path
  extension = args.file_extension
  max_no_of_sentences = args.max_no_of_sentences
  train_ext_path = args.extractive_summ_train_path
  val_ext_path = args.extractive_summ_val_path
  test_ext_path = args.extractive_summ_test_path
  train_gold_sum_path = args.gold_summ_train_path
  val_gold_sum_path = args.gold_summ_val_path
  test_gold_sum_path = args.gold_summ_test_path
  train_ext_splits = args.train_splits
  val_ext_splits = args.val_splits
  test_ext_splits = args.test_splits

  raw_data = get_raw_data(dataset_path, extension)
  raw_train_data = raw_data["train"]
  raw_val_data = raw_data["validation"]
  raw_test_data = raw_data["test"]

  raw_split_train_data = split_raw_data_into_sentences(raw_train_data, max_no_of_sentences)
  raw_split_val_data = split_raw_data_into_sentences(raw_val_data, max_no_of_sentences)
  raw_split_test_data = split_raw_data_into_sentences(raw_test_data, max_no_of_sentences)

  create_gold_summaries_dataset(raw_split_train_data, raw_split_val_data, raw_split_test_data, train_gold_sum_path, val_gold_sum_path, test_gold_sum_path)
  create_ext_summ_dataset(max_no_of_sentences, raw_split_train_data, raw_split_val_data, raw_split_test_data, train_ext_path, val_ext_path, test_ext_path, train_ext_splits , val_ext_splits, test_ext_splits)
main()