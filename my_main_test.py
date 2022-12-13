import numpy as np
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

import h5py

torch_seed=0
numpy.random.seed(42)
torch.manual_seed(torch_seed)

# YAML_FPATH = "choro_conf.yaml"
YAML_FPATH = "my_conf.yaml"

with open(YAML_FPATH, "rb") as stream:
    config = yaml.full_load(stream)


# model ckpt directory
# checkpoint_dir='best_checkpoints/checkpoint_TAGS_sigmoidAS_0.800_margin_0.4_000026_mAP_2342'
# checkpoint_dir='best_checkpoints/checkpoint_TAGS_sigmoidAS_0.800_margin_0.4_000018_mAP_2339'
# checkpoint_dir='best_checkpoints/checkpoint_seed2_TAGS_sigmoidAS_0.800_margin_0.4_000021_mAP_2323'
# checkpoint_dir='best_checkpoints/checkpoint_pretrainedAudioCaps_seed1978_seed0_TAGS_sigmoidAS_0.800_margin_0.4_000053_mAP_2396'
# checkpoint_dir='best_checkpoints/checkpoint_pretrainedAudioCaps_seed1978_seed42_TAGS_sigmoidAS_0.800_margin_0.4_noScheduler_000043_mAP_2402'

checkpoint_dir='outputs_passt/checkpoint_margin1_20juin2022/checkpoint_000030'

# checkpoint_dir='best_checkpoints/checkpoint_seed0_none_margin_0.4_000018_mAP_2292'
# checkpoint_dir='best_checkpoints/checkpoint_TAGS_sigmoidAS_0.800_margin_0.4_000018_mAP_2307'
# checkpoint_dir='best_checkpoints/checkpoint_TAGS_sigmoidAS_0.800_margin_0.4_000021'

training_config = config["training"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

use_sentence_embeddings = True
use_auto_caption_embeddings = False
print("Using Etienne's auto caption embeddings?", use_auto_caption_embeddings)

# load Clotho data
# text_datasets, vocabulary = data_utils.load_data(config["train_data"])
caption_datasets, vocabulary = data_utils.load_data(config["eval_data"])

model_config = config[training_config["model"]]
model = model_utils.get_model(model_config, vocabulary)
print(model)
# print(model.audio_encoder.pointwise_conv.weight.data)
# model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
# model.load_state_dict(model_state)
# optimizer.load_state_dict(optimizer_state)
# for state in optimizer.state.values():
#     for k, v in state.items():
#         if isinstance(v, torch.Tensor):
#             state[k] = v.to(device)
#
# Restore model states
model = model_utils.restore(model, checkpoint_dir)
model.eval()


print("Nb of learnable parameters :",
      sum(p.numel() for p in model.parameters() if p.requires_grad))

# alg_config = training_config["algorithm"]
#
# criterion_config = config[alg_config["criterion"]]
# if config["train_data"]["use_mixup"] and criterion_config["name"] != "MixupTripletRankingLoss":
#     print("ERROR: change criterion to MixupTripletRankingLoss in config file")
#
#
# criterion = getattr(criterion_utils, criterion_config["name"], None)(**criterion_config["args"])
#
# optimizer_config = config[alg_config["optimizer"]]
# optimizer = getattr(optim, optimizer_config["name"], None)(
#     model.parameters(), **optimizer_config["args"]
# )

# lr_scheduler = getattr(optim.lr_scheduler, "ReduceLROnPlateau")(optimizer, **optimizer_config["scheduler_args"])



clotho_split = "test"
clotho_dataset = caption_datasets[clotho_split]
# Dataloader not needed since files are processed one by one
# clotho_loader = DataLoader(dataset=clotho_dataset, batch_size=training_config["algorithm"]["batch_size"],
#                              shuffle=False, collate_fn=data_utils.collate_fn_sentence_embeddings)

# Retrieve audio files for evaluation captions
# output = eval_utils.audio_retrieval(model, caption_datasets[clotho_split], K=10,
#                          use_sentence_embeddings=use_sentence_embeddings,
#                          use_auto_caption_embeddings=use_auto_caption_embeddings)


scores_h5_output_file = os.path.join(checkpoint_dir, "{}_scores.h5".format(clotho_split))
# scores_h5_output_file = None

output = eval_utils.audio_retrieval(model, clotho_dataset, K=10,
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
output.to_csv(os.path.join(checkpoint_dir, "{}.output.csv".format(clotho_split)),
              index=False)
print("Saved", "{}.output.csv".format(clotho_split))

gt_csv = "test.gt.csv"  # ground truth for Clotho evaluation data
pred_csv = os.path.join(checkpoint_dir,
                        "{}.output.csv".format(clotho_split))  # baseline system retrieved output for Clotho evaluation data

retrieval_metrics(gt_csv, pred_csv, log_wandb=False, is_ema=False)

# get loss values on train, val and test

text_datasets, vocabulary = data_utils.load_data(config["train_data"])

text_loaders = {}
for split in ["train", "val", "test"]:
    _dataset = text_datasets[split]
    if use_sentence_embeddings:
        if use_auto_caption_embeddings:
            _loader = DataLoader(dataset=_dataset, batch_size=training_config["algorithm"]["batch_size"],
                                 shuffle=True, collate_fn=data_utils.collate_fn_sentence_and_auto_caption_embeddings)
        else:
            _loader = DataLoader(dataset=_dataset, batch_size=training_config["algorithm"]["batch_size"],
                                 shuffle=True, collate_fn=data_utils.collate_fn_sentence_embeddings)
    else:
        _loader = DataLoader(dataset=_dataset, batch_size=training_config["algorithm"]["batch_size"],
                             shuffle=True, collate_fn=data_utils.collate_fn)

    text_loaders[split] = _loader

alg_config = training_config["algorithm"]

criterion_config = config[alg_config["criterion"]]
if config["train_data"]["use_mixup"] and criterion_config["name"] != "MixupTripletRankingLoss":
    print("ERROR: change criterion to MixupTripletRankingLoss in config file")
    exit(-1)

criterion = getattr(criterion_utils, criterion_config["name"], None)(**criterion_config["args"])

split='train'
data_loader = text_loaders[split]

for batch_idx, data in enumerate(data_loader):
    # Get the inputs; data is a list of tuples (audio_feats, audio_lens, queries, query_lens, infos)
    audio_feats, audio_lens, queries, query_lens, infos, caption_embeds = data
    audio_feats, caption_embeds = audio_feats.to(device), caption_embeds.to(device)
    # print(audio_feats.size())
    # Forward
    audio_embeds, query_embeds, audio_embeds2 = model(audio_feats, caption_embeds)
    loss, list_scores = criterion(audio_embeds, query_embeds, infos, is_train_subset=True)

import matplotlib.pyplot as plt

anchor_scores = list_scores[0]
A_imp_scores = list_scores[1]
Q_imp_scores = list_scores[2]

print("anchor=%.1f    A_imp_scores=%.3f    Q_imp_scores=%.3f\n"%(np.mean(anchor_scores), np.mean(A_imp_scores), np.mean(A_imp_scores)))
# test: anchor=0.4    A_imp_scores=0.005    Q_imp_scores=0.005
# train: anchor=0.4    A_imp_scores=0.015    Q_imp_scores=0.015

fig, ax = plt.subplots()
ax.hist(anchor_scores, bins=20, rwidth=0.9,  alpha=0.2, label='anchor')
ax.set_xlabel("Scores")
# ax.set_ylabel("")
ax.hist(A_imp_scores, bins=20, rwidth=0.9, alpha=0.2, label='A imp')
ax.hist(Q_imp_scores, bins=20, rwidth=0.9, alpha=0.2, label='Q imp')
ax.legend()
plt.savefig("histogram_scores_%s.png"%(split))