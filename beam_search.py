import config
import torch
import sys
import util
class SimilarityComputer:
   __instance = None
   @staticmethod 
   def getInstance():
      """ Static access method. """
      if SimilarityComputer.__instance == None:
         SimilarityComputer()
      return SimilarityComputer.__instance
   def __init__(self):
     
      if SimilarityComputer.__instance != None:
         raise Exception("This class is a singleton!")
      else:
         self.seq = []
         SimilarityComputer.__instance = self

   def SEQ(self,sequences):
   
      self.seq = sequences

   def set_entity_tensors(self,entity_list_tensor, subset_order_input):

      self.entity_list_tensor = entity_list_tensor

      self.subset_order_input = subset_order_input

   def entity_tensors(self):

      return self.entity_list_tensor, self.subset_order_input
 
   def similarity(self,seq):

      if len(self.seq)==0:
         return 0
  
      return max([util.jaccard_similarity(c,seq) for c in self.seq])

   #def set

class beam_search_decoder:
  
    SOS_Index = -1
    EOS_Index = -1

    def __init__(self, decoder):

        self.decoder = decoder

    def initialize(self, beam_hidden):

        input = torch.tensor([beam_search_decoder.SOS_Index,beam_search_decoder.SOS_Index,beam_search_decoder.SOS_Index],dtype=torch.long, device=config.device)
        print(input.shape)
        beam_hidden = torch.stack([beam_hidden,beam_hidden,beam_hidden])
        #sys.exit(0)
        self.decoder(input,beam_hidden)
    

    def decode(self, beam_hidden, encoder_outputs): 

        print('ccc')
        self.initialize(beam_hidden)


class Beam_Node:
  
   def __init__(self,parent,decoder_input,decoder_hidden,hidden_lm,total_value,number_of_nodes,seq):
     
     self.number_of_nodes = number_of_nodes
     self.decoder_input = decoder_input

     #print(self.decoder_input)
     #sys.exit(0)
     self.decoder_hidden = decoder_hidden
     self.hidden_lm = hidden_lm
     self.parent = parent
     self.children = []
     self.total_value = total_value
     self.total_cost = self.eval()
     self.seq = seq + [self.decoder_input]

     self.entity_index = None
     

     #print(self.seq)
     if parent == None:
      self.seq_nodes = [self]

     else:
      self.seq_nodes = parent.seq_nodes + [self] 

     if self.parent != None:
      self.parent.add_child(self)


   def eval(self, alpha=1.0):
      reward = 0
      # Add here a function for shaping a reward

      return (self.total_value / float(self.number_of_nodes - 1 + 1e-6) + alpha * reward)

   def remove_child(self,child_node):

     self.children.remove(child_node)

   def add_child(self,child_node):

     self.children.append(child_node)

   def add_children(self,child_nodes):

     self.children = self.children + child_nodes

   def decode_entity(self,model):

     index = self.decoder_input.item()

     if config.ENTITY_INDEX == index:

        entity_list_tensor, subset_order_input = SimilarityComputer.getInstance().entity_tensors()

        self.entity_index = model.decode_context_entity(entity_list_tensor, subset_order_input, self.decoder_hidden)

        

class Beam_Search_Decoder:

   def __init__(self,model,decoder):

     self.tree_level_node_sets = {}
     self.level_id = 1
     self.model = model
     self.decoder = decoder
     self.beam_size = config.beam_size
     self.EOS_Index = config.EOS_Index
     self.SOS_Index = config.SOS_Index
     self.MAX_LEN = config.MAX_LEN

   def set_entity_indices(self,nodes):

     for node in nodes:
       
       node.decode_entity(model)

   def decode_step(self,parent_nodes,encoder_outputs,entity_list_tensor,output_index_to_entity_index):

     decoder_inputs = torch.stack([parent_node.decoder_input for parent_node in parent_nodes])
     decoder_hiddens = torch.stack([parent_node.decoder_hidden for parent_node in parent_nodes])
     decoder_outputs, decoder_hiddens,context_vector = self.decoder(decoder_inputs, decoder_hiddens,encoder_outputs,is_SoftMax = True)
      
     if config.is_copy_mechanism:

        decoder_outputs = self.model.beam_copy_mechanism(decoder_outputs,decoder_hiddens,entity_list_tensor,output_index_to_entity_index,context_vector)

     else:

        decoder_outputs = decoder_outputs.squeeze(0) 
     
     decoder_hiddens = decoder_hiddens.squeeze(0) 
     
     child_nodes = []
     #print(decoder_outputs)

     for index, parent_node in enumerate(parent_nodes):
    
      prob, topi = decoder_outputs[index].topk(self.beam_size)
      prob = prob.squeeze()
      decoder_inputs = topi.squeeze().detach()
      #print(topi)
      #sys.exit(0)
      child_nodes  = child_nodes+ [Beam_Node(parent_node,decoder_inputs[i],decoder_hiddens[index],None,parent_node.total_value+(prob[i]),parent_node.number_of_nodes+1, parent_node.seq) for i in range(0,self.beam_size)]

     #sys.exit(0)

     return child_nodes

   def BEAM_NODES(self,level_nodes):

     #print(level_nodes)
     #sys.exit(0)
    
     #level_nodes.sort(key=lambda x:x.total_cost, reverse=True)
     

     if len(level_nodes)>self.beam_size:

       level_nodes = level_nodes[:self.beam_size]

     
     return level_nodes

   

   def decode(self,decoder_hidden,encoder_outputs,entity_list_tensor,output_index_to_entity_index):

     self.level_id = 0
     #print(self.SOS_Index)
     #print(torch.tensor(self.SOS_Index).view(1).squeeze(0))
     ROOT = Beam_Node(None,torch.tensor(self.SOS_Index).to(config.device).view(1).squeeze(0),decoder_hidden,None,0,1,[])

     self.tree_level_node_sets[self.level_id] = self.decode_step([ROOT],encoder_outputs,entity_list_tensor,output_index_to_entity_index)

     while self.level_id <= self.MAX_LEN: 
     
       self.level_id = self.level_id + 1
       
       self.tree_level_node_sets[self.level_id] = []
       parent_nodes = [parent_node for parent_node in self.tree_level_node_sets[self.level_id-1] if parent_node.decoder_input != self.EOS_Index]
       #print(len(self.tree_level_node_sets))
       #print(self.level_id)
       #sys.exit(0)
       parent_nodes = [parent_node for parent_node in self.tree_level_node_sets[self.level_id-1] if parent_node.decoder_input != self.EOS_Index]
       #sys.exit(0)

       if len(parent_nodes) == 0:
          break

       self.tree_level_node_sets[self.level_id] = self.decode_step(parent_nodes,encoder_outputs,entity_list_tensor,output_index_to_entity_index)
       self.tree_level_node_sets[self.level_id]=self.BEAM_NODES(self.tree_level_node_sets[self.level_id])

     seq,seq_nodes = self.generate_seq()

     if config.ENTITY_RANKING_LOSS:

        self.set_entity_indices(seq_nodes)
    
     return seq
   
   
   def generate_seq(self):

      
      
      c_nodes = [node for key,tree_level_nodes in self.tree_level_node_sets.items() for node in tree_level_nodes  if len(node.children)==0]
     
      c_nodes.sort(key=lambda x: x.total_cost-config.beam_cost_const*(SimilarityComputer.getInstance().similarity(x.seq)), reverse=True)

      #c_nodes.sort(key=lambda x: x.total_value)
      

      return [x.item() for x in c_nodes[0].seq], c_nodes[0].seq_nodes
 
        
        
        
