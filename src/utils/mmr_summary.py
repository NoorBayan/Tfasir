import torch

def get_cosine_similarity(a,b):
    #print(torch.dot(a,b)/(torch.linalg.norm(a) * torch.linalg.norm(b)))
    return torch.dot(a,b)/(torch.linalg.norm(a) * torch.linalg.norm(b))


def get_mmr_score(sen_i_score, sen_i_embed, list_of_summary_sent_embed, alpha):
    #list_of_summary_sent_embed :  a list of the sentence embeddings in the summary
    temp = []
    for i in range(len(list_of_summary_sent_embed)):
      sim_score = get_cosine_similarity(sen_i_embed, list_of_summary_sent_embed[i])
      temp.append(sim_score)
    
    if temp:
      max_sim_score = max(temp)
    else:
      max_sim_score = 0

    mmr_score = alpha * sen_i_score - (1 - alpha) * max_sim_score
    return mmr_score

def create_mmr_summary(sent_scores, sent_embeds, alpha):
    #sent_scores shape is (batch_size, max_no_of_sentences)
    #sent_embeds shape is (batch_size, max_no_of_sentences, hidden_size)

    summary_completed = torch.zeros(sent_scores.shape[0]) #0 indicates that summary is not yet completed, 1 indicates completed
    sum_done = False
    count = 0
    max_no_of_sentences = 250

    current_sum = torch.zeros(sent_scores.shape)
    list_of_summary_sent_embed = []
    for i in range(sent_scores.shape[0]):
      list_of_summary_sent_embed.append([])
    
    #list_of_summary_sent_embed is a list of lists. E.g the list_of_summary_sent_embed[0] is a list
    #of all sentences embeddings in summary of the first sentences in given batch

    mmr_scores = torch.zeros(sent_scores.shape)
    while (sum_done == False and count < max_no_of_sentences/25):
      count += 1
      for i in range(sent_scores.shape[0]):
        if (summary_completed[i] == 0):
          for j in range(sent_scores.shape[1]):
            #print(list_of_summary_sent_embed[i])
            mmr_scores[i][j] = get_mmr_score(sent_scores[i][j], sent_embeds[i][j], list_of_summary_sent_embed[i], alpha)
            if (mmr_scores[i][j] > 0.5*alpha and current_sum[i][j] != 1):
              #print(mmr_scores[i][j])
              list_of_summary_sent_embed[i].append(sent_embeds[i][j])
              current_sum[i][j] = 1
              break
            
            if (j == sent_scores.shape[1] - 1):
              summary_completed[i] = 1
      
      for k in range(len(summary_completed)):
        if summary_completed[k] == 0:
          break
        if (k == len(summary_completed) - 1):
          sum_done = True

    return current_sum  


def convert_ext_sum_list_to_text(current_sum, article):
    current_sum_text = []

    for i in range(current_sum.shape[0]):
      temp_sum = []
      for j in range(current_sum.shape[1]):
        if (current_sum[i][j] == 1):
          temp_sum.append(article[i][j])

      current_sum_text.append(temp_sum)
      temp_sum = []
    
    return current_sum_text

