import os

import pandas as pd
import torch
import yaml
import json
from nltk.tokenize import sent_tokenize

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

import csv

import h5py

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

"""Generate an h5 file with keys: str(caption id), values: the caption sentence-transformer embedding vector"""

def read_audioset_label_tags():
    with open('../metadata/class_labels_indices.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        lines = list(reader)

    labels = []
    ids = []    # Each label has a unique id such as "/m/068hy"
    for i1 in range(1, len(lines)):
        id = lines[i1][1]
        label = lines[i1][2]
        ids.append(id)
        labels.append(label)

    classes_num = len(labels)

    lb_to_ix = {label : i for i, label in enumerate(labels)}
    ix_to_lb = {i : label for i, label in enumerate(labels)}

    id_to_ix = {id : i for i, id in enumerate(ids)}
    ix_to_id = {i : id for i, id in enumerate(ids)}

    return lb_to_ix, ix_to_lb, id_to_ix, ix_to_id


def read_audioset_ontology(id_to_ix):
    with open('../metadata/audioset_ontology.json', 'r') as f:
        data = json.load(f)

    # Output: {'name': 'Bob', 'languages': ['English', 'French']}
    sentences = []
    for el in data:
        id = el['id']
        if id in id_to_ix:
            name = el['name']
            desc = sent_tokenize(el['description'])[0]
            # if '(' in desc:
                # print(name, '---', desc)
            # print(id_to_ix[id], name, '---', )

            # sent = name
            # sent = name + ', ' + desc.replace('(', '').replace(')', '').lower()
            sent = desc.replace('(', '').replace(')', '').lower()
            sentences.append(sent)
            # print(sent)
            # break
    return sentences

    # print('/m/0dgw9r' in id_to_ix)
    # print(data[0]['name'], '--', data[0]['id'], '---', data[0]['description'])

    nb_classes = len(data)
    print(nb_classes)

lb_to_ix, ix_to_lb, id_to_ix, ix_to_id = read_audioset_label_tags()
print(len(id_to_ix))
sentences = read_audioset_ontology(id_to_ix)


# model: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

with open("../my_conf.yaml", "rb") as stream:
    config = yaml.full_load(stream)

# Load model from HuggingFace Hub
# tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
# model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')


nb_processed = 0

output_file = os.path.join(config["train_data"]["input_path"], 'audioset_527_sentenceNoTags_embeddings_mpnet.hdf5')
# output_file = os.path.join(config["train_data"]["input_path"], 'audioset_527_sentence_embeddings_MiniLM.hdf5')

with h5py.File(output_file, "w") as feature_store:

    # X_tsne = []
    for i in range(len(sentences)):
        # batch = sentences[i:i+bs]
        # cid_batch = caption_ids[i:i+bs]
        # Tokenize sentences

        # print(sentences[i])

        encoded_input = tokenizer(sentences[i], padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        # X_tsne.append(sentence_embeddings.clone().detach().numpy())
        # print("Sentence embeddings:")
        # print(sentence_embeddings.size())
        # torch.Size([1, 384])
        # print(sentence_embeddings)
        feature_store[str(i)] = torch.squeeze(sentence_embeddings)
        print(i, sentences[i])

        # for j,cid_ in enumerate(cid_batch):
        #     feature_store[str(cid_)] = sentence_embeddings[j]
        #     print(str(cid_))

        nb_processed += 1
        # if nb_processed > 9:  break

print("nb_processed OK?", nb_processed == len(sentences))

# X_tsne = np.squeeze(np.array(X_tsne))
# print(X_tsne.shape)
#
# X_embedded = TSNE(n_components=2, learning_rate='auto',
#                   init='random').fit_transform(X_tsne)
#
# plt.plot(X_embedded[:,0], X_embedded[:,1], 'x')
# plt.show()
#
# print(X_embedded.shape)

