import sys
import shutil

def multple_sentences_to_word_seq(input,word_dict,entity_dict):
    
    output = [' '.join(word_dict.index_to_seq(x)).strip() for x in input]
    output = '. '.join(output)
    #print(output)
    for key,ent in entity_dict.items():
        #print(key+" "+ent.entity_string)
        output = output.replace(key,ent.entity_string)
    #sys.exit(0)
    return output
    
def FREEZE_PARAMETERS(model):

    for param in model.parameters():

      param.requires_grad = False

def jaccard_similarity(list1, list2):

    intersection = len(list(set(list1).intersection(list2)))

    union = (len(list1) + len(list2)) - intersection

    return float(intersection / union)

def write_to_file(file_name,list_of_strings):

    with open(file_name, 'w') as f:

       for item in list_of_strings:

           f.write("%s\n" % item)
    
