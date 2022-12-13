'''The goal of this script is to run audio retrieval with caption queries from the Clotho dev-test set
with audio files from AudioCaps. Then we will augment the clotho dev-train set with those and train a new model.
'''

import pandas as pd

import wandb

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

torch_seed=0
numpy.random.seed(42)
torch.manual_seed(torch_seed)

import socket
host_name = socket.gethostname()
print(host_name)

if host_name == 'choro':
    YAML_FPATH = "choro_conf.yaml"
    # data_dir = '../clotho-dataset/data'
    # data_dir = '/homelocal/thomas/clotho-dataset/data'
elif host_name == 'legolas':
    YAML_FPATH = "my_conf.yaml"
    # data_dir = '../clotho-dataset/data'
    # data_dir = '/homelocal/thomas/clotho-dataset/data'
else:
    # data_dir = '/tmpdir/pellegri/corpus/clotho-dataset/data/'
    YAML_FPATH = "osirim_conf.yaml"

print(YAML_FPATH)

with open(YAML_FPATH, "rb") as stream:
    config = yaml.full_load(stream)

training_config = config["training"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model ckpt directory
# checkpoint_dir='best_checkpoints/checkpoint_seed0_none_margin_0.4_000018_mAP_2292'
checkpoint_dir='best_checkpoints/checkpoint_TAGS_sigmoidAS_0.800_margin_0.4_000026_mAP_2342'
# checkpoint_dir='best_checkpoints/checkpoint_TAGS_sigmoidAS_0.800_margin_0.4_000018_mAP_2339'
# checkpoint_dir='best_checkpoints/checkpoint_TAGS_sigmoidAS_0.800_margin_0.4_000018_mAP_2307'
# checkpoint_dir='best_checkpoints/checkpoint_TAGS_sigmoidAS_0.800_margin_0.4_000021'
# checkpoint_dir='best_checkpoints/checkpoint_pretrainedAudioCaps_seed1978_seed0_TAGS_sigmoidAS_0.800_margin_0.4_000053_mAP_2396'

# checkpoint_dir='best_checkpoints/checkpoint_pretrainedAudioCaps_seed1978_seed42_TAGS_sigmoidAS_0.800_margin_0.4_noScheduler_000043_mAP_2402'

_, vocabulary = data_utils.load_data(config["train_data"])

model_config = config[training_config["model"]]
model = model_utils.get_model(model_config, vocabulary)
print(model)

# Restore model states
model = model_utils.restore(model, checkpoint_dir)
model.to(device=device)
model.eval()
print("Nb of learnable parameters :",
      sum(p.numel() for p in model.parameters() if p.requires_grad))


use_sentence_embeddings = True
use_auto_caption_embeddings = False
print("Using Etienne's auto caption embeddings?", use_auto_caption_embeddings)


# I - AUDIOCAPS: stack audio embeddings passed through the model

# Load audio features from AudioCaps
feats_path = os.path.join(config["audiocaps_train_data"]["input_path"], config["audiocaps_train_data"]["audio_features"])
audio_feats = h5py.File(feats_path, "r")
print("Load", feats_path)
sent_emb_path = os.path.join(config["audiocaps_train_data"]["input_path"],
                             config["audiocaps_train_data"]["caption_embeddings"])
sentence_embeddings = h5py.File(sent_emb_path, "r")
print("Load", sent_emb_path)

csv_path = os.path.join(config["audiocaps_train_data"]["input_path"],
                        config["audiocaps_train_data"]["data_splits"]['train'])
df = pd.read_csv(csv_path)
print("Load", csv_path)

from utils.data_utils import AudiocapsQueryAudioDataset

audiocaps_dataset = AudiocapsQueryAudioDataset(audio_feats, df, sentence_embeddings)

fid_embs, fid_fnames = {}, {}
with torch.no_grad():

    for index in range(len(audiocaps_dataset)):
        audio, word_query, info, query = audiocaps_dataset[index]

        audio = torch.unsqueeze(audio, dim=0).to(device=device)
        audio_emb, _, _ = model(audio, query)
        audio_emb = torch.squeeze(audio_emb, dim=0).to(device=device)

        fid_embs[info["fid"]] = audio_emb
        fid_fnames[info["fid"]] = info["fname"]

# Stack audio embeddings
audiocaps_embs, audiocaps_fnames = [], []
for fid in fid_embs:
    audiocaps_embs.append(fid_embs[fid])
    audiocaps_fnames.append(fid_fnames[fid])

audiocaps_audio_embs = torch.vstack(audiocaps_embs)  # dim [N, E]


# II - Load queries from Clotho dev-test, and do retrieval on the AUDIOCAPS audio embeddings
#

# nb of audio files to retrieve
K = 10

# Load audio features
feats_path = os.path.join(config["eval_data"]["input_path"], config["eval_data"]["audio_features"])
audio_feats_clotho = h5py.File(feats_path, "r")
print("Load", feats_path)

sent_emb_path = os.path.join(config["eval_data"]["input_path"], config["eval_data"]["caption_embeddings"])
sentence_embeddings_clotho = h5py.File(sent_emb_path, "r")
print("Load", sent_emb_path)

# Load json from Clotho dev-test
json_path = os.path.join(config["train_data"]["input_path"], config["train_data"]["data_splits"]["test"])
df_clotho = pd.read_json(json_path)
print("Load", json_path)

clothotest_dataset = EvalQueryAudioDataset(audio_feats_clotho, df_clotho, sentence_embeddings_clotho)

# Retrieve audio files for eval captions
split_name = 'audiocaps_with_clotho_dev_eval_queries'
scores_output_fpath = os.path.join(checkpoint_dir, "{}_scores.h5".format(split_name))
# scores_h5_output_file = None

with h5py.File(scores_output_fpath, "w") as score_store:
    with torch.no_grad():

        cid_embs, cid_infos = {}, {}

        # Encode audio signals and captions
        for index in range(len(clothotest_dataset)):

            audio, word_query, info, query = clothotest_dataset[index]

            audio = torch.unsqueeze(audio, dim=0).to(device=device)
            query = torch.unsqueeze(query, dim=0).to(device=device)

            _, query_emb, _ = model(audio, query)

            query_emb = torch.squeeze(query_emb, dim=0).to(device=device)

            cid_embs[info["cid"]] = query_emb
            cid_infos[info["cid"]] = info


        # Compute similarities
        output_rows = []
        for cid in cid_embs:

            # print(cid)

            sims = torch.mm(torch.vstack([cid_embs[cid]]), audiocaps_audio_embs.T).flatten().to(device=device)

            # print(cid, sims.size(), sims[:5])

            sorted_idx = torch.argsort(sims, dim=-1, descending=True)

            csv_row = [cid_infos[cid]["caption"]]  # caption
            for idx in sorted_idx[:K]:  # top-K retrieved fnames
                csv_row.append(audiocaps_fnames[idx])

            score_store[str(cid)] = sims.clone().detach().cpu().numpy()

            output_rows.append(csv_row)


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

output = pd.DataFrame(data=output_rows, columns=csv_fields)
output.to_csv(os.path.join(checkpoint_dir, "{}.output.csv".format(split_name)),
              index=False)
print("Saved", "{}.output.csv".format(split_name))
