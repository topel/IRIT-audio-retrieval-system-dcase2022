import os

import pandas as pd
import torch
import yaml
import json

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

import h5py

# >>> import pandas as pd
# >>> df = pd.read_json("retrieval_captions.json")
# >>> df
#       cid   fid  ...                                            caption                                             tokens
# 0       1     1  ...  A liquid continuously being poured out and hit...  [A, liquid, continuously, being, poured, out, ...
# 1       2     2  ...  Water coming out of a shower head and hitting ...  [Water, coming, out, <UNK>, <UNK>, shower, hea...
# 2       3     3  ...  A grass cutter is roaring very loudly when rev...  [A, grass, cutter, is, roaring, very, loudly, ...
# 3       4     4  ...    Heavy machinery is being used inside a building  [Heavy, machinery, is, being, used, inside, <U...
# 4       5     5  ...       Calm water flows down stream two rocks clash  [<UNK>, water, flows, down, stream, two, rocks...
# ..    ...   ...  ...                                                ...                                                ...
# 995   996   996  ...    The sound in something is reduced from too much  [The, sound, in, something, is, reduced, from,...
# 996   997   997  ...  People talks at a party as some are playing vi...  [People, talks, at, <UNK>, party, as, some, ar...
# 997   998   998  ...       A dog barks and whines as some children talk  [A, dog, barks, <UNK>, whines, as, some, child...
# 998   999   999  ...  Splashing water running fast and hard suddenly...  [Splashing, water, running, fast, <UNK>, hard,...
# 999  1000  1000  ...  Several small wooden objects are being tossed ...  [Several, small, wooden, objects, are, being, ...
#
# [1000 rows x 6 columns]

# model: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

# Mean Pooling - Take attention mask into account for correct averaging

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


sentences, caption_ids = [], []

split='test'
# for split in ["train"]:
json_path='/baie/corpus/DCASE2022/evaluation_set_task6b/retrieval_captions.json'
df = pd.read_json(json_path)
print("Load", json_path)

# print(df["fname"])
# print(df["original"])

for cid, fid, el in zip(df["cid"], df["fid"], df["caption"]):
    sentences.append(el)
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
# tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
# model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')


# bs=100
nb_processed = 0
output_file = '/baie/corpus/DCASE2022/evaluation_set_task6b/retrieval_captions_embeddings_mpnet.hdf5'
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

