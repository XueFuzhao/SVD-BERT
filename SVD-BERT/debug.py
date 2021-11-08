from pytorch_pretrained_bert.modeling import BertModel,BertConfig,BertForPreTraining
from pytorch_pretrained_bert.modeling import BertPreTrainingHeads, PreTrainedBertModel, BertPreTrainingHeads
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

import torch



bert_config = BertConfig.from_json_file('/home/users/nus/e0792473/scratch/BERT_base_model/config.json')
print(bert_config)
pretrained_dict = torch.load('/home/users/nus/e0792473/scratch/BERT_base_model/pytorch_model.bin')
match=0

    
print("===================================")
    
model = BertForPreTraining(bert_config)
model_dict = model.state_dict()

match=0
total=0
for k,v in pretrained_dict.items():
    #print(k)
    total+=1
    for k1,v1 in model_dict.items():
        if k==k1:
            print(k)
            match+=1
'''
for k,v in pretrained_dict.items():
    print(k)
    match+=1
for k,v in model_dict.items():
    print(k)
    total+=1
'''
print(match)
print(total)


#for k1,v1 in model_dict.items():
#    print(k1)


#bert_encoder = BertModel.from_pretrained(
#                bert_config,
#                cache_dir=PYTORCH_PRETRAINED_BERT_CACHE)

#checkpoint_state_dict = torch.load(args.model_file, map_location=torch.device("cpu"))


