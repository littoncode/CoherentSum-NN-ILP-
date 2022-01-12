import math
import torch
import torch.nn as nn
from pytorch_transformers import BertModel, BertConfig
from neural import MultiHeadedAttention, PositionwiseFeedForward
import config
import pickle
import sys
import torch.nn.functional as F
from beam_search import Beam_Search_Decoder

class Decoder(nn.Module):

    def __init__(self, hidden_size, output_size, word_embedding_size,pre_train,dropout_p=0.1):

        super(Decoder, self).__init__()

        self.hidden_size = hidden_size

        self.output_size = output_size

        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(self.output_size, word_embedding_size).cpu()

        #self.embedding.weight.requires_grad=False

        self.dropout = nn.Dropout(self.dropout_p)

        self.attention = GlobalAttention(hidden_size, coverage=False, attn_type="general",
                 attn_func="softmax")

        self.gru =nn.GRU(word_embedding_size, hidden_size)

        self.out = nn.Linear(2*self.hidden_size, self.output_size)

        self.test = False
        self.pre_train = pre_train

    def set_pretrain(self,pre_train):

        self.pre_train = pre_train       
 
    def packForSelfAttention(self,output):

        #print(output.shape)

        sys.exit(0)

    def forward(self, input, hidden,encoder_ouputs=None,is_SoftMax=False):

        #print(input)
        #sys.exit(0)

        embedded = self.embedding(input)

        embedded = self.dropout(embedded)

        
        
        if not self.test:

          hidden = hidden.view(1,1,-1)
          

        elif self.test:

          hidden = hidden.view(1,len(hidden),-1)
          embedded = embedded.view(1,len(embedded),-1)
         

        output, hidden = self.gru(embedded,hidden)

        context_vector = None 

        if self.pre_train:

          
          context_vector = self.initHidden()
          
          if self.test:
             context_vector = context_vector.repeat(1,output.shape[1],1)
          else:
             context_vector = context_vector.repeat(output.shape[0],1,1)
             
          #sys.exit(0)
         
        if not self.pre_train:
          
          output_c = None

          if self.test:
          
            output_c = output.view(output.shape[1],1,-1)
 
            encoder_ouputs = encoder_ouputs.repeat(output.shape[1],1,1)
                     
          else:

            output_c = output.squeeze(1).unsqueeze(0)

            encoder_ouputs = encoder_ouputs.unsqueeze(0)

          
          context_vector,attentions = self.attention(output_c,encoder_ouputs)

          
        out_prob_dist = torch.cat([output,context_vector],2)

        
        out_prob_dist = self.out(out_prob_dist)
        
        #print(output.shape)
        #print(context_vector.shape)

        if is_SoftMax:
         
           out_prob_dist =  F.softmax(out_prob_dist,dim=2)

        else:
         
           out_prob_dist =  F.log_softmax(out_prob_dist,dim=2)
      
        return out_prob_dist,output,context_vector

    def initHidden(self,is_reduced=False):

        if is_reduced:
           return torch.zeros(1, self.hidden_size, device=config.device)

        return torch.zeros(1, 1, self.hidden_size, device=config.device)


class GlobalAttention(nn.Module):
    
    def __init__(self, dim, coverage=False, attn_type="dot",
                 attn_func="softmax"):
        self.count = 0
        super(GlobalAttention, self).__init__()

        self.dim = dim
        assert attn_type in ["dot", "general", "mlp"], (
            "Please select a valid attention type (got {:s}).".format(
                attn_type))
        self.attn_type = attn_type
        assert attn_func in ["softmax", "sparsemax"], (
            "Please select a valid attention function.")
        self.attn_func = attn_func

        if self.attn_type == "general":
            self.linear_in = nn.Linear(dim, dim, bias=False)
        elif self.attn_type == "mlp":
            self.linear_context = nn.Linear(dim, dim, bias=False)
            self.linear_query = nn.Linear(dim, dim, bias=True)
            self.v = nn.Linear(dim, 1, bias=False)
        # mlp wants it with bias
        out_bias = self.attn_type == "mlp"
        self.linear_out = nn.Linear(dim * 2, dim, bias=out_bias)

        if coverage:
            self.linear_cover = nn.Linear(1, dim, bias=False)

    def score(self, h_t, h_s):

        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        #aeq(src_batch, tgt_batch)
        #aeq(src_dim, tgt_dim)
        #aeq(self.dim, src_dim)

        if self.attn_type in ["general", "dot"]:

            if self.attn_type == "general":
                h_t_ = h_t.view(tgt_batch * tgt_len, tgt_dim)
                h_t_ = self.linear_in(h_t_)
                h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
            h_s_ = h_s.transpose(1, 2)
            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)

            return torch.bmm(h_t, h_s_)
        else:
            dim = self.dim
            wq = self.linear_query(h_t.view(-1, dim))
            wq = wq.view(tgt_batch, tgt_len, 1, dim)
            wq = wq.expand(tgt_batch, tgt_len, src_len, dim)

            uh = self.linear_context(h_s.contiguous().view(-1, dim))
            uh = uh.view(src_batch, 1, src_len, dim)
            uh = uh.expand(src_batch, tgt_len, src_len, dim)

            # (batch, t_len, s_len, d)
            wquh = torch.tanh(wq + uh)

            return self.v(wquh.view(-1, dim)).view(tgt_batch, tgt_len, src_len)

    def forward(self, source, memory_bank, memory_lengths=None, coverage=None):
        

        # one step input
        if source.dim() == 2:
            one_step = True
            source = source.unsqueeze(1)
        else:
            one_step = False
       
        #sys.exit(0)
        batch, source_l, dim = memory_bank.size()
        
        batch_, target_l, dim_ = source.size()

        if coverage is not None:
            batch_, source_l_ = coverage.size()
            
        if coverage is not None:
            cover = coverage.view(-1).unsqueeze(1)
            memory_bank += self.linear_cover(cover).view_as(memory_bank)
            memory_bank = torch.tanh(memory_bank)

        # compute attention scores, as in Luong et al.
        align = self.score(source, memory_bank)

        if memory_lengths is not None:

            mask = sequence_mask(memory_lengths, max_len=align.size(-1))
            mask = mask.unsqueeze(1)  # Make it broadcastable.
            align.masked_fill_(1 - mask, -float('inf'))

        # Softmax or sparsemax to normalize attention weights
        if self.attn_func == "softmax":
            align_vectors = F.softmax(align.view(batch*target_l, source_l), -1)
        else:
            align_vectors = sparsemax(align.view(batch*target_l, source_l), -1)
        align_vectors = align_vectors.view(batch, target_l, source_l)

        # each context vector c_t is the weighted average
        # over all the source hidden states
        c = torch.bmm(align_vectors, memory_bank)

        # concatenate
        concat_c = torch.cat([c, source], 2).view(batch*target_l, dim*2)
        attn_h = self.linear_out(concat_c).view(batch, target_l, dim)
        if self.attn_type in ["general", "dot"]:
            attn_h = torch.tanh(attn_h)

        if one_step:

            attn_h = attn_h.squeeze(1)
            align_vectors = align_vectors.squeeze(1)
            batch_, dim_ = attn_h.size()
            batch_, source_l_ = align_vectors.size()

        else:

            attn_h = attn_h.transpose(0, 1).contiguous()
            align_vectors = align_vectors.transpose(0, 1).contiguous()
            target_l_, batch_, dim_ = attn_h.size()
            target_l_, batch_, source_l_ = align_vectors.size()

        
        return attn_h, align_vectors


class Bert(nn.Module):

    def __init__(self, bert_model_key, finetune=False):
        super(Bert, self).__init__()

       
        self.model = BertModel.from_pretrained(bert_model_key)
        
        self.finetune = finetune

    def forward(self, x, segs):

        if(self.finetune):
            hidden_reps, cls_head = self.model(x, segs)
        else:
            self.eval()

            with torch.no_grad():
                hidden_reps, cls_head = self.model(x, segs)

        return cls_head


class Classifier(nn.Module):

    def __init__(self, hidden_size):

        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask_cls):

        h = self.linear1(x).squeeze(-1)
        sent_scores = self.sigmoid(h) * mask_cls.float()
        return sent_scores


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 90):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        print(pe.shape)
        #print(pe)
        #sys.exit(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
        print(self.pe[:x.size(0)].shape)
        #sys.exit(0)

        x = x + self.pe[:x.size(0)]
        return self.dropout(x)



class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, heads, d_ff, dropout):

        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, query, inputs):
        if (iter != 0):
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs

        
        context = self.self_attn(input_norm, input_norm, input_norm)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class ExtTransformerEncoder(nn.Module):

    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers=0):

        super(ExtTransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers
        self.pos_emb = PositionalEncoding(d_model,dropout,90)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_inter_layers)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.wo = nn.Linear(d_model, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, top_vecs):
        """ See :obj:`EncoderBase.forward()`"""

        print(top_vecs.shape)
        

        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
 
        print(batch_size)
        print(n_sents)
        x = self.pos_emb(top_vecs)
       

        for i in range(self.num_inter_layers):

            x = self.transformer_inter[i](i, x, x)  # all_sents * max_tokens * dim

        print(x.shape)

       

        x = self.layer_norm(x)
        #sent_scores = self.sigmoid(self.wo(x))
        #print(sent_scores)
        #sent_scores = sent_scores.squeeze(-1) * mask.float()

        #sys.exit(0)

        return x


class SelectionNetwork(nn.Module):
    
     def __init__(self,input_size_layer1,i_dim):
        
        super(SelectionNetwork,self).__init__()

        input_size_layer2 = int(input_size_layer1*1.8)
    
        self.layer1 = nn.Linear(input_size_layer1, input_size_layer2,bias=True)

        self.layer2 = nn.Linear(input_size_layer2,i_dim,bias=True) 

     def forward(self,_input1):
        
        _input2 = F.relu(self.layer1(_input1))

        selection_level = torch.sigmoid(self.layer2(_input2))

        return selection_level



class summary_content_selector(nn.Module):

    def __init__(self, d_ff, heads, dropout, num_inter_layers,bert_model_key,output_vocab_size,word_embedding_size):

        super(summary_content_selector, self).__init__()

       
        self.bert_model = Bert(bert_model_key, finetune=False) 

        self.d_model = self.bert_model.model.config.hidden_size 

        self.ext_transformer = ExtTransformerEncoder(self.d_model, d_ff, heads, dropout, num_inter_layers=0)
 
        self.subset_ordering_gru = nn.GRU(input_size = self.d_model,hidden_size = self.d_model)

        self.info_gate = SelectionNetwork(2*self.d_model,self.d_model)
        
  
        self.coh_gate = SelectionNetwork(2*self.d_model,self.d_model)

        self.control_gate = SelectionNetwork(self.d_model,1)

        self.sentence_decoder = Decoder(self.d_model, output_vocab_size, word_embedding_size,True,dropout_p=0.1)

        self.sentence_decoder.set_pretrain(True)

        self.beam_search_decoder = Beam_Search_Decoder(self,self.sentence_decoder)
        

    def forward(self, doc_sents, segs,output_seqs):

        #encoding summary sentences ............................................................................

        x = []

        for i in range(0,len(doc_sents)):

            token_ids = torch.tensor(doc_sents[i]).unsqueeze(0)
            seg_ids   = torch.tensor(segs[i]).unsqueeze(0)

            x.append(self.bert_model(token_ids,seg_ids))

        x = torch.stack(x)  #summary sentence representations
        x = self.ext_transformer(x) # context specific summary sentence representations
        l1, l2, n = x.shape 
     

        #init gru hidden states................................................................................
        subset_order_hidden_state = torch.zeros(1, self.d_model, device=config.device)

        print(subset_order_hidden_state.shape)

        for i in range(0,len(output_seqs)):

            n_subset_order_hidden_state = subset_order_hidden_state.view(1,1,-1).repeat(l1,l2,1)
            selection_levels = self.info_gate(torch.cat([n_subset_order_hidden_state,x],2))            #if i>0:

            print(selection_levels.shape)
            
            coherence_levels = self.coh_gate(torch.cat([n_subset_order_hidden_state,x],2))
            print(coherence_levels.shape)

            current_iout = torch.mul(selection_levels,x)
            coherence_out = torch.mul(coherence_levels,x)

            control = self.control_gate(n_subset_order_hidden_state)
            current_iout = control*current_iout+(1-control)*coherence_out

            print(current_iout.shape)


        print(x.shape)
       


class trainer:


      def __init__(self,training_data_info):

          

          print('cat')
          dbfile = open(training_data_info,'rb')

          self.entity_dict,self.role_dict,self.output_vocab,self.doc_sum_pair = pickle.load(dbfile)   
          config.output_vocab_size = len(self.output_vocab.index_dict) 
          dbfile.close()

          config.SOS_Index = self.output_vocab.dict['<SOS>']
          config.EOS_Index = self.output_vocab.dict['<EOS>']

          self.model = summary_content_selector(config.ext_ff_size, config.ext_heads, config.ext_dropout, config.ext_layers,config.bert_model_key,config.output_vocab_size,config.word_embedding_size)
                 
      def train(self):

          doc_sents = self.doc_sum_pair['doc_sents']

          segs = self.doc_sum_pair['doc_seg_ids']

          sum_tokens = self.doc_sum_pair['sum_tokens']

          print(sum_tokens)

          sys.exit(0)

          print(doc_sents)
          print(segs)

          print(len(self.output_vocab.index_dict))
          
          self.model(doc_sents,segs,sum_tokens)

          
trainer = trainer(config.COHERENCE_TRAINSET)
trainer.train()



    


     

          
          






