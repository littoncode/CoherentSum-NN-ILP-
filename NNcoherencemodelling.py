import torch
from torch import nn
import config
import sys 
import torch.nn.functional as F
import math


class Net(nn.Module):

  def __init__(self,input_shape):
    
      super(Net,self).__init__()

      input_shape1 = math.floor(input_shape*0.70) 

      input_shape2 = math.floor(input_shape*0.30) 

      self.fc1 = nn.Linear(input_shape,input_shape1)
    
      self.fc2 = nn.Linear(input_shape1,input_shape2)

      self.fc3 = nn.Linear(input_shape2,1)

  def forward(self,x):

      x = torch.relu(self.fc1(x))

      x = torch.relu(self.fc2(x))

      x = torch.sigmoid(self.fc3(x))

      return x


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


class coherencemodelling_nn(nn.Module):

      def __init__(self,entity_list_size,entity_dim,entityrole_list_size,entity_role_dim):

          super(coherencemodelling_nn, self).__init__()

          self.entity_list_size = entity_list_size
          self.entity_dim = entity_dim
          self.entityrole_list_size = entityrole_list_size
          self.entity_role_dim = entity_role_dim

          
          self.emb_entities = nn.Embedding(self.entity_list_size,self.entity_dim)
          self.emb_entitytypes = nn.Embedding(self.entityrole_list_size,self.entity_role_dim)
          self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.entity_dim+self.entity_role_dim, nhead=5)
          self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
          self.cross_attention_layer = GlobalAttention(self.entity_dim+self.entity_role_dim)
          self.fcn = Net(2*(self.entity_dim+self.entity_role_dim))

          self.loss = nn.MSELoss()

      def forward(self,records):

          record = records[0]

          ent = self.emb_entities(torch.tensor(record[0]).to(config.device))

          ent_role = self.emb_entitytypes(torch.tensor(record[1]).to(config.device))

          pre_record = torch.cat((ent,ent_role),1).unsqueeze(0)

          pre_record = self.transformer_encoder(pre_record)
         
          coh_scores = []
          
          for i in range(1,len(records)):
            
              record = records[i] 

              ent = self.emb_entities(torch.tensor(record[0]).to(config.device))

              ent_role = self.emb_entitytypes(torch.tensor(record[1]).to(config.device))

              record = torch.cat((ent,ent_role),1).unsqueeze(0)
              
              record = self.transformer_encoder(record)

              attn_h1, align_vectors = self.cross_attention_layer(record,pre_record)
              attn_h2, align_vectors = self.cross_attention_layer(pre_record,record)
              attn_h1 = torch.mean(attn_h1,0)
              attn_h2 = torch.mean(attn_h2,0)
              sent_pair = torch.cat([attn_h1,attn_h2],1)
              print(sent_pair.shape)
              out = self.fcn(sent_pair)
              coh_scores.append(out.squeeze(0).squeeze(0))
              print(out.shape)
              #sys.exit(0)

              print('self.cross_attention_layer(record,pre_record)'+str(attn_h1.shape))
              print('self.cross_attention_layer(record,pre_record)'+str(attn_h2.shape))

              print(record.shape)
              print(pre_record.shape)

              pre_record = record

       
          coh_scores = torch.stack(coh_scores)
          target = torch.full(coh_scores.shape,1)
          print(self.loss(coh_scores,target))

          #self.loss



      
             
             
              
