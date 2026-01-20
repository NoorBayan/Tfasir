import torch
import copy
import argparse
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import gc
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
  parser = argparse.ArgumentParser(description = 'Script to train model on dataset')
  parser.add_argument('--checkpoint_path', type=str, default=None, help="Path where checkpoint (includes model, optimizer, current step, and base_lr) is stored")
  parser.add_argument('--extractive_summ_train_path', type=str, default="../data/extractive_summ/train.json", help="Path where training data of articles and its extractive summaries is stored")
  parser.add_argument('--gold_summ_train_path', type=str, default="../data/gold_summ/gold_summaries_train.json", help="Path where training data of gold summaries is stored")
  parser.add_argument('--extractive_summ_val_path', type=str, default="../data/extractive_summ/validation.json", help="Path where validation data of articles and its extractive summaries is stored")
  parser.add_argument('--gold_summ_val_path', type=str, default="../data/gold_summ/gold_summaries_validation.json", help="Path where validation data of gold summaries is stored")
  parser.add_argument('--train_splits', type=int, default=1, help="number of splits in which extractive_summ_train_data is stored, i.e if splits=2 data is stored in 2 files")
  parser.add_argument('--val_splits', type=int, default=1, help="number of splits in which extractive_summ_val_data is stored, i.e if splits=2 data is stored in 2 files")
  parser.add_argument('--train_data_ratio', type=int, default=100, help="Ratio of data in percentage used for training")
  parser.add_argument('--val_data_ratio', type=int, default=100, help="Ratio of data in percentage used for validation")
  parser.add_argument('--batch_size', type=int, default=4, help="Training batch size")
  parser.add_argument('--alpha', type=float, default = 0.9, help="hyperparameter for MMR-based selection during evalution")
  parser.add_argument('--gamma', type=float, default = 0.98, help="hyperparameter for the weight of the cross entropy loss term")
  parser.add_argument('--base_lr', type=float, default = 1e-3, help="hyperparameter used to calculate current learning rate: lr = base_lr * min(step**(-0.5), step * warmup**(-1.5))")
  parser.add_argument('--warmup_steps', type=int, default = 2500, help="Number of warmup steps")
  parser.add_argument('--beta1', type=float, default = 0.9, help="coefficient used for computing running average of gradients in Adam")
  parser.add_argument('--beta2', type=float, default = 0.999, help="coefficient used for computing running average of the square of gradients in Adam")
  parser.add_argument('--no_of_epochs', type=int, default=1, help="Number of epochs for training")
  parser.add_argument('--model_freq', type=int, default=100, help="Frequency of saving checkpoints")
  parser.add_argument('--val_freq', type=int, default=100, help="Frequency (i.e number of batches) of performing validation")
  parser.add_argument('--train_loss_output_freq', type=int, default=5, help="Frequency (i.e number of batches) of pritining training loss")

  return parser


def validation_loop(dataloader, model, loss_fn, rouge_output_freq = 5, alpha = 0.9, gamma = 0.98):
    print("\nPerforming Validation \n-------------------------")
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    batch_size = dataloader.batch_size
    test_loss = 0
    rouge1_score = 0
    rouge2_score = 0
    count = 0

    with torch.no_grad():
      for X, y, z in dataloader:
          count += 1
          articles = copy.copy(X)
          pred, sentence_embeds = model(X)
          
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
            print(f"Rouge1: {rouge1_score/count:>7f} [{current:>5d}/{size:>2d}]")
            print(f"Rouge2: {rouge2_score/count:>7f} [{current:>5d}/{size:>2d}]")
            print(f"Loss:   {test_loss/count:>7f} [{current:>5d}/{size:>2d}]")
    
    test_loss /= num_batches
    rouge1_score /= num_batches
    rouge2_score /= num_batches
    print(f"\nValidation Error: \nRouge1 Score: {rouge1_score:>8f}, Rouge2 Score: {rouge2_score:>8f}, Avg loss: {test_loss:>8f} \n")

    return rouge1_score, rouge2_score, test_loss

def train_loop():
  parser = arg_parser()
  args = parser.parse_args()
  checkpoint_path = args.checkpoint_path
  extractive_summ_train_path = args.extractive_summ_train_path
  gold_summ_train_path = args.gold_summ_train_path
  extractive_summ_val_path = args.extractive_summ_val_path
  gold_summ_val_path = args.gold_summ_val_path
  train_splits = args.train_splits
  val_splits = args.val_splits
  train_data_ratio = args.train_data_ratio
  val_data_ratio = args.val_data_ratio
  batch_size = args.batch_size
  alpha = args.alpha
  gamma = args.gamma
  base_lr = args.base_lr
  warmup_steps = args.warmup_steps
  beta1 = args.beta1
  beta2 = args.beta2
  no_of_epochs = args.no_of_epochs
  model_freq = args.model_freq
  val_freq = args.val_freq
  train_loss_output_freq = args.train_loss_output_freq
  step_num = 1

  train_ext_data = load_data_from_json(extractive_summ_train_path, train_splits)
  train_gold_sum_data = load_data_from_json(gold_summ_train_path)
  val_ext_data = load_data_from_json(extractive_summ_val_path, val_splits)
  val_gold_sum_data = load_data_from_json(gold_summ_val_path)
  train_indices = [i for i in range(int(len(train_ext_data) * train_data_ratio/100))]
  val_indices = [i for i in range(int(len(val_ext_data) * val_data_ratio/100))]
  train_ext_data = torch.utils.data.Subset(train_ext_data, train_indices)
  train_gold_sum_data = torch.utils.data.Subset(train_gold_sum_data, train_indices)
  val_ext_data = torch.utils.data.Subset(val_ext_data, val_indices)
  val_gold_sum_data = torch.utils.data.Subset(val_gold_sum_data, val_indices)

  train_dataset = ExtSumDataset(train_ext_data, train_gold_sum_data)
  val_dataset = ExtSumDataset(val_ext_data, val_gold_sum_data)
  train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

  size = len(train_dataloader.dataset)
  loss_fn = nn.BCELoss()
  bert_tokenizer, bert_model = load_bert()
  hiExtSummModel = HiExtSumm(bert_model, bert_tokenizer).cuda()
  optimizer = torch.optim.Adam(hiExtSummModel.parameters(), lr=base_lr, betas=(beta1, beta2), eps=1e-08, weight_decay=0)

  if (checkpoint_path != None):
    checkpoint = torch.load(checkpoint_path)
    hiExtSummModel.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.Adam(hiExtSummModel.parameters(), lr=base_lr, betas=(beta1, beta2), eps=1e-08, weight_decay=0)
    optimizer.load_state_dict(checkpoint["optimizer"])
    step_num = checkpoint["step_num"]
    base_lr = checkpoint["base_lr"]
  else:
    checkpoint_path = "../models/checkpoint.pth"

  avg_loss = 0
  loss_list_output_freq = np.array([])
  loss_list = np.array([])
  avg_loss_list = np.array([])
  batch_list = np.array([])
  val_loss = np.array([])
  val_rouge1 = np.array([])
  val_rouge2 = np.array([])

  for t in range(no_of_epochs):
    print(f"Epoch {t+1}\n-------------------------------")

    for batch, (X, y, z) in enumerate(train_dataloader):
      articles = copy.copy(X)
      pred, sentence_embeds = hiExtSummModel(X)
      d_model = sentence_embeds.shape[2]
      y = y.to("cuda")
      loss_func = loss_fn(pred, y)
      loss_rd = get_loss_rd(pred, sentence_embeds)
      loss = gamma * loss_func + (1 - gamma) * loss_rd
      avg_loss += loss.item()

      for g in optimizer.param_groups:
        temp = [step_num ** (-0.5), step_num * (warmup_steps ** (-1.5))]
        g['lr'] = base_lr * torch.min(torch.tensor(temp))

      # Backpropagation
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if (batch+1) % train_loss_output_freq == 0:
          avg_loss = avg_loss/train_loss_output_freq
          loss_list = np.append(loss_list, avg_loss)
          loss_list_output_freq = np.append(loss_list_output_freq, avg_loss)
          batch_list = np.append(batch_list, batch + 1)
          loss, current = avg_loss, (batch + 1) * batch_size
          print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

          torch.cuda.empty_cache()
          gc.collect()
          avg_loss = 0

      if ((batch + 1) % model_freq == 0):
        avg_loss_list = np.append(avg_loss_list, np.mean(loss_list))
        print("Last ", model_freq, " Batches Loss Mean: ", np.mean(loss_list))
        loss_list = np.array([])

        checkpoint = { 
          'model': hiExtSummModel.state_dict(),
          'optimizer': optimizer.state_dict(),
          'step_num': step_num,
          'base_lr': base_lr,
        }
        torch.save(checkpoint, checkpoint_path)
        plt.title("Epoch " + str(t))
        plt.scatter(batch_list, loss_list_output_freq)
        plt.show()
        #add an option to save this graph. 

      if (batch + 1) % val_freq == 0:
        temp_rouge1, temp_rouge2, temp_loss = validation_loop(val_dataloader, hiExtSummModel, loss_fn, train_loss_output_freq, alpha, gamma)
        temp_rouge1 = temp_rouge1.to("cpu").numpy()
        temp_rouge2 = temp_rouge2.to("cpu").numpy()
        #temp_loss = temp_loss.to("cpu").numpy()
        val_rouge1 = np.append(val_rouge1, temp_rouge1)
        val_rouge2 = np.append(val_rouge2, temp_rouge2)
        val_loss = np.append(val_loss, temp_loss)
        # print("Validation Rouge1 Score: ", val_rouge1)
        # print("Validation Rouge2 Score: ", val_rouge2)
        # print("Validation Loss: ", val_loss)

        tempp = np.arange(len(val_loss))
        plt.scatter(tempp, val_rouge1)
        plt.title("Rouge 1")
        plt.show()
        plt.scatter(tempp, val_rouge2)
        plt.title("Rouge 2")
        plt.show()
        plt.scatter(tempp, val_loss)
        plt.title("Loss")
        plt.show()
        print("-------------------------\n")


      del articles
      step_num += 1
    
    loss_list_output_freq = np.array([])

def main():
  train_loop()
main()

