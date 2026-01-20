import torch

def get_cosine_similarity_of_list_of_vec(a,b): 
  # a.size (batch_size, max_no_of_sentences, hidden_size) (2,3,4)
  # b.size (batch_size, max_no_of_sentences, hidden_size)
  a = a.to("cuda")
  b = b.to("cuda")
  dot_prod = torch.matmul(a, torch.transpose(b,1,2)) # (batch_size, max_no_of_sentences, max_no_of_sentences)

  #dot_prod_sqr = torch.pow(dot_prod, 2)
  #a_norm = torch.sqrt(torch.matmul(a, torch.transpose(a,1,2))) 
  #b_norm = torch.sqrt(torch.matmul(b, torch.transpose(b,1,2)))
  a_norm = torch.norm(a, dim=2) #(batch_size, max_no_of_sentences)
  b_norm = torch.norm(b, dim=2) #(batch_size, max_no_of_sentences)
  temp = torch.ones(dot_prod.shape).to("cuda")
  eps = torch.tensor(10**(-8)).to("cuda")
  
  for i in range(temp.shape[0]):
    temp[i] = torch.outer(a_norm[i], b_norm[i])

  return torch.div(dot_prod, torch.maximum(temp, eps)) 

def get_loss_rd(sent_scores, sent_embeds):
  #sent_scores shape is (batch_size, max_no_of_sentences)
  #sent_embeds shape is (batch_size, max_no_of_sentences, hidden_size)

  loss_rd = torch.zeros(sent_embeds.shape[0]).to("cuda")
  cosine_sims = get_cosine_similarity_of_list_of_vec(sent_embeds, sent_embeds).to("cuda")
      
  for i in range(sent_embeds.shape[0]):
    out_prod = torch.outer(sent_scores[i], sent_scores[i])
    #print("out_prod: ", out_prod)
    tempp = torch.mul(out_prod, cosine_sims)
    #print("tempp: ", tempp)
    loss_rd[i] = torch.mean(tempp + 10**(-8))

  #print("Loss_rd: ", loss_rd)
  # for i in range(sent_embeds.shape[0]):
  #   for j in range(sent_embeds.shape[1]):
  #     for k in range(sent_embeds.shape[1]):
  #       #temp_cosin_sim = get_cosine_similarity(sent_embeds[i][j], sent_embeds[i][k]).to("cuda")
  #       loss_rd[i] += sent_scores[i][j] * sent_scores[i][k] * cosine_sims[i][j][k]

    #loss_rd[i] = loss_rd[i]/(sent_embeds.shape[1] * sent_embeds.shape[1])

  mean_loss_rl = torch.mean(loss_rd).to("cuda")
  return mean_loss_rl
