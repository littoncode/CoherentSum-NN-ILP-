from preprocess import *
from NNcoherencemodelling import *
import pickle
import config

dbfile = open(config.COHERENCE_TRAINSET,'rb')
pre_sent_entity_roles,entity_dict,role_dict  = pickle.load(dbfile)
dbfile.close()


config.entity_list_size = len(entity_dict.dict)

config.entityrole_list_size = len(role_dict.dict)

print(config.entity_list_size)
print(config.entityrole_list_size)
print(pre_sent_entity_roles)

coh_model = coherencemodelling_nn(config.entity_list_size,config.entity_dim,config.entityrole_list_size,config.entity_role_dim)

coh_model(pre_sent_entity_roles)
