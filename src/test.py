import torch
import copy
import argparse
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
from utils.rouge import get_rouge_score_for_sum
from utils.loss_rd import get_loss_rd
from utils.mmr_summary import convert_ext_sum_list_to_text
from utils.mmr_summary import create_mmr_summary
from utils.load_bert import load_bert
from model.HiExtSumm import HiExtSumm
from data.ExtSumDataset import ExtSumDataset
from data.load_from_json import load_data_from_json

def transpose(x):
  y = []
  for i in range(len(x[0])):
    temp = []
    for j in range(len(x)):
      temp.append(x[j][i])
    y.append(temp)

  return y

def arg_parser():
  parser = argparse.ArgumentParser(description = 'Script to test model on dataset')
  parser.add_argument('model_path', type=str, help="Path where the model to be tested is stored")
  parser.add_argument('--extractive_summ_test_path', type=str, default="../data/extractive_summ/test.json", help="Path where test data of articles and its extractive summaries is stored")
  parser.add_argument('--gold_summ_test_path', type=str, default="../data/gold_summ/gold_summaries_test.json", help="Path where test data of gold summaries is stored")
  parser.add_argument('--test_splits', type=int, default=1, help="number of splits in which extractive_summ_data is stored, i.e if splits=2 data is stored in 2 files")
  parser.add_argument('--batch_size', type=int, default=4, help="Testing batch size")
  parser.add_argument('--alpha', type=float, default = 0.9, help="hyperparameter for MMR-based selection during evalution")
  parser.add_argument('--gamma', type=float, default = 0.98, help="hyperparameter for the weight of the cross entropy loss term")
  parser.add_argument('--test_data_ratio', type=int, default=100, help="Ratio of data in percentage used for testing")
  parser.add_argument('--rouge_output_freq', type=int, default = 5, help="Frequency of printing mean rouge score during testing")

  return parser

def test_loop():
    parser = arg_parser()
    args = parser.parse_args()
    model_path = args.model_path
    test_ext_data_path = args.extractive_summ_test_path
    test_gold_sum_data_path = args.gold_summ_test_path
    test_splits = args.test_splits
    batch_size = args.batch_size
    alpha = args.alpha
    gamma = args.gamma
    test_data_ratio = args.test_data_ratio
    rouge_output_freq = args.rouge_output_freq

    test_ext_data = load_data_from_json(test_ext_data_path, test_splits)
    test_gold_sum_data = load_data_from_json(test_gold_sum_data_path)
    indices = [i for i in range(int(len(test_ext_data) * test_data_ratio/100))]
    #indices = np.arange(int(len(test_ext_data) * test_data_ratio/100))
    test_ext_data = torch.utils.data.Subset(test_ext_data, indices)
    test_gold_sum_data = torch.utils.data.Subset(test_gold_sum_data, indices)
    print(len(test_ext_data))
    print(len(test_gold_sum_data))

    test_dataset = ExtSumDataset(test_ext_data, test_gold_sum_data)
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    loss_fn = nn.BCELoss()
    tokenizer, model = load_bert()
    hiExtSummModel = HiExtSumm(model, tokenizer).cuda()
    hiExtSummModel.load_state_dict(torch.load(model_path))
    hiExtSummModel.eval()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    rouge1_score = 0
    rouge2_score = 0
    count = 0

    with torch.no_grad():
      for X, y, z in dataloader:
          count += 1
          articles = copy.copy(X)
          pred, sentence_embeds = hiExtSummModel(X)
          
          mmr_summ = create_mmr_summary(pred, sentence_embeds, alpha)
          mmr_summ_text = convert_ext_sum_list_to_text(mmr_summ, transpose(articles))

          rouge1_scores_list = torch.zeros(len(mmr_summ_text)).to("cuda")
          rouge2_scores_list = torch.zeros(len(mmr_summ_text)).to("cuda")
          for i in range(len(mmr_summ_text)):
            rouge1_scores_list[i], rouge2_scores_list[i] = get_rouge_score_for_sum(z[i], mmr_summ_text[i])

          rouge1_score += torch.mean(rouge1_scores_list)
          rouge2_score += torch.mean(rouge2_scores_list)

          y = y.to("cuda")
          #loss_rl = get_loss_rl(pred, sentence_embeds, transpose(articles), z, alpha)
          loss_rd = get_loss_rd(pred, sentence_embeds)
          temp_test_loss = gamma * loss_fn(pred, y) + (1 - gamma) * loss_rd
          test_loss += temp_test_loss.item()

          if (count % rouge_output_freq == 0):
            current = count * batch_size
            print(f"Rouge1: {rouge1_score/count:>7f} [{current:>5d}/{size:>5d}]")
            print(f"Rouge2: {rouge2_score/count:>7f} [{current:>5d}/{size:>5d}]")
            print(f"Loss:   {test_loss/count:>7f} [{current:>5d}/{size:>5d}]")
            print("\n")
    
    test_loss /= num_batches
    rouge1_score /= num_batches
    rouge2_score /= num_batches
    print(f"Test Error: \n Rouge1 Score: {rouge1_score:>8f}, Rouge2 Score: {rouge2_score:>8f}, Avg loss: {test_loss:>8f} \n")

    return rouge1_score, rouge2_score, test_loss

def main():
  test_loop()
main()
