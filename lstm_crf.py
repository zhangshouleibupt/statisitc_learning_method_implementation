import torch
import torch.nn as nn

class LSTMCrf(nn.Module):
    def __init__(self,voc_size,embed_dim,hidden_dim,tag_size,lstm_layer_num):
        self.voc_size = voc_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.tag_size = tag_size
        self.lstm_layer_num = lstm_layer_num
        self.embed = nn.Embedding(voc_size,embed_dim)
        self.re_map = nn.Linear(embed_dim,hidden_dim//2)
        self.lstm = nn.LSTM(hidden_dim//2,hidden_dim//2,num_layers = 2,
                            batch_first = True,bidirectional=True)
        self.emission = nn.Linear(hidden_dim,tag_size)
        self.transmission = nn.Prameter(torch.randn(tag_size,tag_size))
        self.rule = nn.Relu()
    def forward(self,input_seq):
        #input: torch.tensor size = (1,time_length)
        #return:torch.tensor size = (time_length,tag_size)
        out = self.relu(self.re_map(self.embed(input_seq)))
        out = out.unsqueeze(0)
        out,hidden = self.lstm(out)
        out = out.squeeze()
        emission_value = self.emission(out)
        return emission_value
    def cal_path_value(self,emission_value,real_labels):
        l = len(emission_value)
        ans = 0
        for i in range(l):
            ans += emission[i][real_labels[i]]
        for i in range(l-1):
            tmp_tag = real_labels[i]
            next_tag = real_labels[i+1]
            ans += self.transmission.data[tmp_tag][next_tag]
        return torch.exp(ans)
    def forward_alpha(self,emission_value):
        l = len(emission_value)
        alpha = torch.ones((1,self.tag_size))
        for i in range(l):
            alpha = torch.dot(alpha,torch.exp(self.transmmision.data)) * torch.exp(emission_value[i])
        return torch.sum(alpha)
    def loss(self,input_seq,labels):
        emission_value = self.forward(input_seq)
        path_score = self.cal_path_value(emission_value,labels)
        alpha_value = self.forward(emission_value)
        return -torch.log(path_score) + torch.log(alpha_value)

