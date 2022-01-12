from preprocess import *
import nltk
import config
import pickle
from transformers import BertTokenizer
import torch
from pytorch_transformers import BertModel, BertConfig
from nltk.tokenize import word_tokenize

class data_set_creator:

      def __init__(self): 

          self.entity_dict = word_dict()
          self.role_dict = word_dict()
          self.output_vocab = word_dict()
          self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

      #add entity and roles to dictionaries
      def add_to_dictionaries(self,sent_entity_roles):

          sent_records = []

          for sent_entity_role in sent_entity_roles:
       
              sent_ent_record = []
              sent_ent_role_record = []

              for entity_role in sent_entity_role:

                  entity_roles = entity_role.split(':') 
                  sent_ent_record.append(self.entity_dict.addWord(entity_roles[0]))
                  sent_ent_role_record.append(self.role_dict.addWord(entity_roles[1]))

              sent_records.append((sent_ent_record,sent_ent_role_record))

          return sent_records

      #entity role sequence in sentences for coherence training 
      def data_for_coherence_text(self,sum_sents):

          print([sentence for sentence in sum_sents])
          sent_entity_roles = [preprocessor.extract_entity_roles(sentence) for sentence in sum_sents]

          print(sent_entity_roles)
          pre_sent_entity_roles = self.add_to_dictionaries(sent_entity_roles)
          print(pre_sent_entity_roles)
         
          return pre_sent_entity_roles


      #converting sentence to BERT indices
      def sents_to_BERT_indices(self,doc_sents):

          doc_sents = [['[CLS]'] + self.bert_tokenizer.tokenize(sentence)+['[SEP]'] for sentence in doc_sents]
          print('doc_sents : '+str(doc_sents))
          doc_sents = [self.bert_tokenizer.convert_tokens_to_ids(sent_tokens) for sent_tokens in doc_sents]
          print('doc_sents : '+str(doc_sents))
          doc_seg_ids = [[0 for _ in range(len(sent_tokens))] for sent_tokens in doc_sents]

          return doc_sents, doc_seg_ids 

      #summary tokens to the dictionary

      def add_summary_tokens_to_dictionary(self,sum_sents):
          
          sum_sents = [word_tokenize(sent) for sent in sum_sents]

          sum_sent_indices = [self.output_vocab.addWords(['<SOS>']+_sent+['<EOS>']) for _sent in sum_sents]

          return sum_sent_indices

      #preprocess document summary pairs 

      def pre_process_doc_sum_pairs(self,doc,summary):

          doc_sum_pair = {}

          summary = preprocessor.coref_resolved_text(summary)  
          doc = preprocessor.coref_resolved_text(doc) 
          preprocessed_texts  = preprocessor.replace_entities_in_texts([doc,summary])

          print(preprocessed_texts)
          #sys.exit(0)
          print("str(text) : "+str(summary))

          summary = preprocessed_texts[1]

          sum_sents = nltk.sent_tokenize(summary)

          #summary text 

          doc_sum_pair['sum_tokens'] = self.add_summary_tokens_to_dictionary(sum_sents)

          #info coherent text training 
          
          doc_sum_pair['coh_text'] = self.data_for_coherence_text(sum_sents)
          print(str(sum_sents))
          #sys.exit(0)

          doc = preprocessed_texts[0]

          doc_sents = nltk.sent_tokenize(doc) 

          doc_sents,doc_seg_ids = self.sents_to_BERT_indices(doc_sents)

          #info for document summary pair 
      
          doc_sum_pair['doc_sents'] = doc_sents

          doc_sum_pair['doc_seg_ids'] = doc_seg_ids

          return doc_sum_pair


'''entity_dict = word_dict()
role_dict = word_dict()

def add_to_dictionaries(sent_entity_roles):

    sent_records = []

    for sent_entity_role in sent_entity_roles:
       
        sent_ent_record = []
        sent_ent_role_record = []

        for entity_role in sent_entity_role:

            entity_roles = entity_role.split(':') 
            sent_ent_record.append(entity_dict.addWord(entity_roles[0]))
            sent_ent_role_record.append(role_dict.addWord(entity_roles[1]))

        sent_records.append((sent_ent_record,sent_ent_role_record))

    return sent_records'''


doc = "whether a sign of a good read ; or a comment on the ' pulp ' nature of some genres of fiction , the oxfam second-hand book charts have remained in the da vinci code author ' s favour for the past four years . dan brown has topped oxfam ' s ' most donated ' list again , his fourth consecutive year . having sold more than 80 million copies of the da vinci code and had all four of his novels on the new york times bestseller list in the same week , it ' s hardly surprising that brown ' s hefty tomes are being donated to charity by readers keen to make some room on their shelves . another cult crime writer responsible to heavy-weight hardbacks , stieg larsson , is oxfam ' s ' story_separator_special_tag a woman reads a copy of the newly released book ' ' the lost symbol ' ' by dan brown , at a speed reading book launch event in sydney , september 15 , 2009. reuters/tim wimborne san francisco the latest novel from \" da vinci code \" author dan brown , \" the lost symbol , \" broke one-day sales records , its publisher and booksellers said . readers snapped up over one million hardcover copies across the united states , canada and the united kingdom after it was released on tuesday , said publisher knopf doubleday , a division of random house inc. \" we are seeing historic , record-breaking sales across all types of our accounts in north america for ' the lost symbol , \" said sonny mehta , editor in chief of knopf doubleday story_separator_special_tag bestselling author is also the most frequently given away to charity shops dan brown might be one of the world ' s bestselling authors but it turns out that readers aren ' t too keen on keeping his special blend of religious conspiracy and scholarly derring-do on their shelves once they ' ve bought it . brown , who has sold more than 81m copies of the da vinci code worldwide , has been revealed as the most donated author to oxfam ' s 700 high street shops . with just four books to his name – although his long-awaited fifth the lost symbol is published next month – brown did well to see off competition from john grisham , author of more than 20 and the second-most likely writer to be ditched in a charity shop by readers story_separator_special_tag a charity shop is urging people to stop donating the da vinci code after becoming overwhelmed with copies . the oxfam shop in swansea has been receiving an average of one copy of the dan brown novel a week for months , leaving them with little room for any other books . staff who are struggling to sell copies of the book have put a note up in the store saying they would rather donors hand in their vinyl instead ."


s = 'During briefing in Geneva, WHO chief Tedros Adhanom Ghebreyesus said that while Omicron variant of COVID-19 does appear to be less severe compared to Delta, it does not mean it should be categorised as mild. "Just like previous variants, Omicron is hospitalising people and it is killing people," he added.'

'''text = preprocessor.coref_resolved_text(s)  
text = preprocessor.replace_entities(text)

print("str(text) : "+str(text))
sent_text = nltk.sent_tokenize(text)

print(text)

print([sentence for sentence in sent_text])
sent_entity_roles = [preprocessor.extract_entity_roles(sentence) for sentence in sent_text]
pre_sent_entity_roles = add_to_dictionaries(sent_entity_roles)'''

data_set_creator = data_set_creator()
#data_set_creator.data_for_coherence_text(s)

doc_sum_pair  = data_set_creator.pre_process_doc_sum_pairs(doc,s)

dbfile = open(config.COHERENCE_TRAINSET,'wb')

pickle.dump((data_set_creator.entity_dict,data_set_creator.role_dict,data_set_creator.output_vocab,doc_sum_pair), dbfile)   
                  
dbfile.close()
   
