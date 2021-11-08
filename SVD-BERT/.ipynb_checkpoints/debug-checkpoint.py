from pytorch_pretrained_bert.modeling import BertModel,BertConfig
from pytorch_pretrained_bert.modeling import BertPreTrainingHeads, PreTrainedBertModel, BertPreTrainingHeads
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

import torch



bert_config = BertConfig.from_json_file('/home/users/nus/e0792473/scratch/BERT_base_model/config.json')
print(bert_config)
checkpoint_state_dict = torch.load('/home/users/nus/e0792473/scratch/BERT_base_model/pytorch_model.bin')

print(checkpoint_state_dict)
#bert_encoder = BertModel.from_pretrained(
#                bert_config,
#                cache_dir=PYTORCH_PRETRAINED_BERT_CACHE)

#checkpoint_state_dict = torch.load(args.model_file, map_location=torch.device("cpu"))


