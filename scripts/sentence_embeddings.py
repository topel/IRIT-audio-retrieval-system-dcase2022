import os

import pandas as pd
import yaml

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

import h5py

"""Generate an h5 file with 
- for groundtruth captions (5 per file) : 
    keys: str(caption id), values: the caption sentence-transformer embedding vector (384-d with miniLM, 768-d with mpnet)

>>> df = pd.read_json("/baie/corpus/DCASE2022/Clotho.v2.1/development_captions.json")
>>> 
>>> df
         cid   fid                         fname                                           original                                            caption                                             tokens
0          1   862  Distorted AM Radio noise.wav        A muddled noise of broken channel of the TV        A muddled noise of broken channel of the TV  [A, muddled, noise, <UNK>, broken, channel, <U...
1          2   862  Distorted AM Radio noise.wav     A television blares the rhythm of a static TV.      A television blares the rhythm of a static TV  [A, television, blares, the, rhythm, <UNK>, <U...
2          3   862  Distorted AM Radio noise.wav    Loud television static dips in and out of focus    Loud television static dips in and out of focus  [<UNK>, television, static, dips, in, <UNK>, o...
3          4   862  Distorted AM Radio noise.wav  The loud buzz of static constantly changes pit...  The loud buzz of static constantly changes pit...  [The, loud, buzz, <UNK>, static, constantly, c...
4          5   862  Distorted AM Radio noise.wav  heavy static and the beginnings of a signal on...  heavy static and the beginnings of a signal on...  [heavy, static, <UNK>, the, beginnings, <UNK>,...
...      ...   ...                           ...                                                ...                                                ...                                                ...
19190  19191  2458                Wood chips.wav  Some pounding in a room follows breaking of ma...  Some pounding in a room follows breaking of ma...  [Some, pounding, in, <UNK>, room, follows, bre...
19191  19192  2458                Wood chips.wav  A stack of wooden blocks fall over and crash t...  A stack of wooden blocks fall over and crash t...  [A, stack, <UNK>, wooden, blocks, fall, over, ...
19192  19193  2458                Wood chips.wav  They are knocking down the toy blocks of the kid.   They are knocking down the toy blocks of the kid  [They, are, knocking, down, the, toy, blocks, ...
19193  19194  2458                Wood chips.wav  Small blocks or chips are moving around and a ...  Small blocks or chips are moving around and a ...  [Small, blocks, or, chips, are, moving, around...
19194  19195  2458                Wood chips.wav         Crushing up food to make a dish for others         Crushing up food to make a dish for others  [Crushing, up, food, <UNK>, make, <UNK>, dish,...

[19195 rows x 6 columns]

  """

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


with open("../my_conf.yaml", "rb") as stream:
    config = yaml.full_load(stream)

sentences, audio_file_ids, caption_ids = [], [], []

for split in ["train", "val", "test"]:
# for split in ["train"]:
    json_path = os.path.join(config["train_data"]["input_path"], config["train_data"]["data_splits"][split])
    df = pd.read_json(json_path)
    print("Load", json_path)

    # print(df["fname"])
    # print(df["original"])

    for cid, fid, el in zip(df["cid"], df["fid"], df["caption"]):
        sentences.append(el)
        audio_file_ids.append(fid)
        caption_ids.append(cid)
        # print(el)

    # print(len(df["fid"]), len(df["cid"]), len(df["fname"]), len(df["original"]), len(df["caption"]))
    # for i in range(0,11):
    #     print(df["fid"][i], df["cid"][i], df["fname"][i])
    # print(min(df["fid"]), min(df["cid"]), max(df["fid"]), max(df["cid"]))
    # train: 1 1 3839 19195
    # val: 3840 19196 4884 24420
    # test:4885 24421 5929 29645
print(len(sentences))

# # Load model from HuggingFace Hub

# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
# tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
# model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

print("Nb of learnable parameters :",
      sum(p.numel() for p in model.parameters() if p.requires_grad))

# bs=100
nb_processed = 0
output_file = os.path.join(config["train_data"]["input_path"], 'sentence_embeddings_mpnet.hdf5')
# output_file = os.path.join(config["train_data"]["input_path"], 'sentence_embeddings_MiniLM.json')

# # test on sentence[0] to know the embed output dim
# encoded_input = tokenizer(sentences[0], padding=True, truncation=True, return_tensors='pt')
# # Compute token embeddings
# with torch.no_grad():
#     model_output = model(**encoded_input)
# # Perform pooling
# sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
# print(sentence_embeddings.size())
# mpnet: 768

with h5py.File(output_file, "w") as feature_store:

    for i in range(len(sentences)):
        # batch = sentences[i:i+bs]
        # cid_batch = caption_ids[i:i+bs]
        # Tokenize sentences
        encoded_input = tokenizer(sentences[i], padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        # print("Sentence embeddings:")
        # print(sentence_embeddings.size())
        # torch.Size([1, 384])

        feature_store[str(caption_ids[i])] = torch.squeeze(sentence_embeddings)
        print(caption_ids[i])

        # for j,cid_ in enumerate(cid_batch):
        #     feature_store[str(cid_)] = sentence_embeddings[j]
        #     print(str(cid_))

        nb_processed += 1
        # break

print("nb_processed OK?", nb_processed == len(sentences))