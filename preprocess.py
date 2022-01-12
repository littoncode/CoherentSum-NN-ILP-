import spacy
from spacy import displacy
import neuralcoref
import sys
import nltk
from nltk.corpus import stopwords

class word_dict:

    def __init__(self):

        self.dict = {}
        self.word_frequency = {}
        self.index = -1
        self.index_dict = {}

    def addWord(self,n_word):
       
        if n_word not in self.dict:

           self.index = self.index + 1
           self.dict[n_word] = self.index 
           self.index_dict[self.index] = n_word
           self.word_frequency[n_word] = 0

        self.word_frequency[n_word] = self.word_frequency[n_word] + 1

        return self.dict[n_word] 
           
 
    def addWords(self,n_words):

        return [self.addWord(n_word) for n_word in n_words]

    def n_words(self,n_indices):

       return [self.index_dict[n_index] for n_index in n_indices]


class preprocessor:
      
      nlp = spacy.load('en_core_web_lg')
      coref = neuralcoref.NeuralCoref(nlp.vocab)
      nlp.add_pipe(coref, name='neuralcoref')

      ENTITY_ROLES = set(['nsubjpass','nsubj','dobj','pobj'])
      STOP_WORDS = set(stopwords.words('english')) 

      @classmethod
      def coref_resolved_text(self,text):

          doc = preprocessor.nlp(text)
          doc._.coref_resolved
    
          return doc._.coref_resolved

      @classmethod  
      def jaccard_similarity(self,key1, key2):

          list1 = [w for w in key1.split() if w not in preprocessor.STOP_WORDS and w.isalnum]
          list2 = [w for w in key2.split() if w not in preprocessor.STOP_WORDS and w.isalnum]

          #print(list1)
          #print(list2)

          intersection = len(list(set(list1).intersection(list2)))
          union = (len(set(list1)) + len(set(list2))) - intersection

          return float(intersection) / union

      @classmethod
      def get_entity_key(self,entity_text,entity_type,entity_key_dict,entity_type_dict,index):
          
          if entity_text in entity_key_dict:

             return entity_key_dict[entity_text], index

          
          if entity_type == 'CARDINAL':
  
             return '$NUM', index

          if entity_type == 'DATE':
             
             return '$DATE', index

          for key, val in entity_key_dict.items():

             if preprocessor.jaccard_similarity(entity_text,key)>0.45:

                #and entity_type in entity_type_dict[val]
 

                print(entity_text)

                entity_key_dict[entity_text] = val

                return val, index

          entity_key_dict[entity_text]  = 'Entity'+str(index)

          entity_type_dict['@Entity'+str(index)] = entity_type
          
          return entity_key_dict[entity_text],index+1

      @classmethod
      def replace_entities(self,text,entity_key_dict,entity_type_dict,index):

          print(text)

          doc = preprocessor.nlp(text)

          for e in reversed(doc.ents):
 

              print(e.text)
              start = e.start_char
              end = start + len(e.text)
              entity_token,index = preprocessor.get_entity_key(e.text,e.label_,entity_key_dict,entity_type_dict,index)
              text = text[:start] + entity_token + text[end:]
              
          print(entity_key_dict)
          print(entity_type_dict)

          print(text)
 
          return text

      @classmethod
      def replace_entities_in_texts(self,texts):

          entity_key_dict= {}
          entity_type_dict = {}
          index = 0

          processed_texts = [self.replace_entities(text,entity_key_dict,entity_type_dict,index) for text in texts]
           
              
           
          return processed_texts


      
      @classmethod
      def role_label(self,dependency):
          
          if dependency in preprocessor.ENTITY_ROLES:
             return dependency

          return 'OTH'
           
      @classmethod
      def extract_entity_roles(self,sentence):

          parsed_text = preprocessor.nlp(sentence)

          print(parsed_text)

          return [str(text)+':'+self.role_label(text.dep_) for text in parsed_text if text.dep_ in preprocessor.ENTITY_ROLES or 'entity' in str(text)] 

             
'''print(preprocessor.jaccard_similarity('the da vinci','the da vinci code'))

print(preprocessor.extract_entity_roles('$entity6 , who has sold more than 81m copies of $entity2 code worldwide , has been revealed as the most donated author to $entity11 \' s $entity10 high street shops'))
#sys.exit(0)
          

s = "whether a sign of a good read ; or a comment on the ' pulp ' nature of some genres of fiction , the oxfam second-hand book charts have remained in the da vinci code author ' s favour for the past four years . dan brown has topped oxfam ' s ' most donated ' list again , his fourth consecutive year . having sold more than 80 million copies of the da vinci code and had all four of his novels on the new york times bestseller list in the same week , it ' s hardly surprising that brown ' s hefty tomes are being donated to charity by readers keen to make some room on their shelves . another cult crime writer responsible to heavy-weight hardbacks , stieg larsson , is oxfam ' s ' story_separator_special_tag a woman reads a copy of the newly released book ' ' the lost symbol ' ' by dan brown , at a speed reading book launch event in sydney , september 15 , 2009. reuters/tim wimborne san francisco the latest novel from \" da vinci code \" author dan brown , \" the lost symbol , \" broke one-day sales records , its publisher and booksellers said . readers snapped up over one million hardcover copies across the united states , canada and the united kingdom after it was released on tuesday , said publisher knopf doubleday , a division of random house inc. \" we are seeing historic , record-breaking sales across all types of our accounts in north america for ' the lost symbol , \" said sonny mehta , editor in chief of knopf doubleday story_separator_special_tag bestselling author is also the most frequently given away to charity shops dan brown might be one of the world ' s bestselling authors but it turns out that readers aren ' t too keen on keeping his special blend of religious conspiracy and scholarly derring-do on their shelves once they ' ve bought it . brown , who has sold more than 81m copies of the da vinci code worldwide , has been revealed as the most donated author to oxfam ' s 700 high street shops . with just four books to his name – although his long-awaited fifth the lost symbol is published next month – brown did well to see off competition from john grisham , author of more than 20 and the second-most likely writer to be ditched in a charity shop by readers story_separator_special_tag a charity shop is urging people to stop donating the da vinci code after becoming overwhelmed with copies . the oxfam shop in swansea has been receiving an average of one copy of the dan brown novel a week for months , leaving them with little room for any other books . staff who are struggling to sell copies of the book have put a note up in the store saying they would rather donors hand in their vinyl instead ."

#s = 'Deepika has a dog. She loves him. The movie star has always been fond of animals'

text = preprocessor.coref_resolved_text(s)  
text = preprocessor.replace_entities(text)'''
        
