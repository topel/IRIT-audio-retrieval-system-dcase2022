import pandas as pd

import os
import time

import numpy
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from utils import criterion_utils, data_utils, eval_utils, model_utils
from eval_script import retrieval_metrics
# from models import ema
from utils.data_utils import EvalQueryAudioDataset

import h5py

torch_seed = 0
numpy.random.seed(42)
torch.manual_seed(torch_seed)

# YAML_FPATH = "choro_conf.yaml"
YAML_FPATH = "my_conf.yaml"

with open(YAML_FPATH, "rb") as stream:
    config = yaml.full_load(stream)


# model ckpt directory
# checkpoint_dir='best_checkpoints/checkpoint_seed0_none_margin_0.4_000018_mAP_2292'
# checkpoint_dir='best_checkpoints/checkpoint_TAGS_sigmoidAS_0.800_margin_0.4_000026_mAP_2342'
# checkpoint_dir='best_checkpoints/checkpoint_TAGS_sigmoidAS_0.800_margin_0.4_000018_mAP_2339'
# checkpoint_dir='best_checkpoints/checkpoint_TAGS_sigmoidAS_0.800_margin_0.4_000018_mAP_2307'
# checkpoint_dir='best_checkpoints/checkpoint_TAGS_sigmoidAS_0.800_margin_0.4_000021'
# checkpoint_dir='best_checkpoints/checkpoint_pretrainedAudioCaps_seed1978_seed0_TAGS_sigmoidAS_0.800_margin_0.4_000053_mAP_2396'
checkpoint_dir='best_checkpoints/checkpoint_pretrainedAudioCaps_seed1978_seed42_TAGS_sigmoidAS_0.800_margin_0.4_noScheduler_000043_mAP_2402'

training_config = config["training"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

use_sentence_embeddings = True
use_auto_caption_embeddings = False
print("Using Etienne's auto caption embeddings?", use_auto_caption_embeddings)

_, vocabulary = data_utils.load_data(config["train_data"])

model_config = config[training_config["model"]]
model = model_utils.get_model(model_config, vocabulary)
print(model)

# Restore model states
model = model_utils.restore(model, checkpoint_dir)
model.eval()
print("Nb of learnable parameters :",
      sum(p.numel() for p in model.parameters() if p.requires_grad))

# Load audio features
split_name = 'evaluation_set_task6b'

eval_data_dir='/baie/corpus/DCASE2022/evaluation_set_task6b'
feats_path = os.path.join(eval_data_dir, 'eval_task6b_audio_passt_torch.hdf5')
audio_feats = h5py.File(feats_path, "r")
print("Load", feats_path)

sent_emb_path = os.path.join(eval_data_dir, 'retrieval_captions_embeddings_mpnet.hdf5')
sentence_embeddings = h5py.File(sent_emb_path, "r")
print("Load", sent_emb_path)

json_path = os.path.join(eval_data_dir, 'test_captions.json')
df = pd.read_json(json_path)
print("Load", json_path)
# print(df)

eval_dataset = EvalQueryAudioDataset(audio_feats, df, sentence_embeddings)

# eval_loader = DataLoader(dataset=eval_dataset, batch_size=training_config["algorithm"]["batch_size"],
#                              shuffle=False, collate_fn=data_utils.audiocaps_collate_fn_sentence_embeddings)


# Retrieve audio files for eval captions
scores_h5_output_file = os.path.join(checkpoint_dir, "{}_scores.h5".format(split_name))
# scores_h5_output_file = None

output = eval_utils.audio_retrieval(model, eval_dataset, K=10,
                                    use_sentence_embeddings=use_sentence_embeddings,
                                    use_auto_caption_embeddings=use_auto_caption_embeddings,
                                    scores_output_fpath=scores_h5_output_file)

csv_fields = ["caption",
              "file_name_1",
              "file_name_2",
              "file_name_3",
              "file_name_4",
              "file_name_5",
              "file_name_6",
              "file_name_7",
              "file_name_8",
              "file_name_9",
              "file_name_10"]

output = pd.DataFrame(data=output, columns=csv_fields)
output.to_csv(os.path.join(checkpoint_dir, "{}.output.csv".format(split_name)),
              index=False)
print("Saved", "{}.output.csv".format(split_name))
