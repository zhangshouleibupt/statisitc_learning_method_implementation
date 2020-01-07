import torch 
import torch.nn as nn

class LSTMCRF(nn.Module):
    def __init__(sefl,
                 tag_nums,
                 lstm_layers,
                 voc_size,
                 hidden_dim,
                 ):
        self. = args
        self.tag_nums = tag_nums
        self.lstm_layers = lstm_layers
        self.voc_size = voc_size
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(voc_size,hidden_dim)
        self.lstm = nn.LSTM(hidden_dim,hidden_dim,layer_num=lstm_layers
                            bidirectional=True)
        self.emission = nn.Linear(hidden_dim*2,tag_nums)
        self.transmision = nn.Linear(tag_nums*2,tag_nums+2)
    def neg_log_likehood(self,feats,tags):
        golden_socre = self._get_tag_score(feats,tags)
        add_score = self._get_add_score(feats)
        #the orignial score is golden_score - add_score
        #cause we need to minimize the objecte fuction
        #so we should add the negative label in front of the objection
        return add_score - golden_score
    def _get_add_score(self,feats):
        pass
    def _vitebi_decode(self,feats):
        pass
    def _get_tag_score(self,feats,tags):
        pass
    def _get_lstm_feats(self):
        pass

