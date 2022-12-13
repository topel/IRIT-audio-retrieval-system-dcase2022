import os
import pickle

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class Vocabulary(object):

    def __init__(self):
        self.word2vec = {}
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

        self.weights = None

    def add_word(self, word, word_vector):
        if word not in self.word2idx:
            self.word2vec[word] = word_vector
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def get_weights(self):
        for idx in range(self.idx):
            if self.weights is None:
                self.weights = self.word2vec[self.idx2word[idx]]
            else:
                self.weights = np.vstack((self.weights, self.word2vec[self.idx2word[idx]]))

        return self.weights

    def __call__(self, word):
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


# class MixupDataset(Dataset):
#     """ Mixing Up wave forms
#     """
#
#     def __init__(self, dataset, beta=2, rate=0.5):
#         self.beta = beta
#         self.rate = rate
#         self.dataset = dataset
#         print(f"Mixing up waveforms from dataset of len {len(dataset)}")
#
#     def __getitem__(self, index):
#         if torch.rand(1) < self.rate:
#             x1, f1, y1 = self.dataset[index]
#             idx2 = torch.randint(len(self.dataset), (1,)).item()
#             x2, f2, y2 = self.dataset[idx2]
#             l = np.random.beta(self.beta, self.beta)
#             l = max(l, 1. - l)
#             x1 = x1-x1.mean()
#             x2 = x2-x2.mean()
#             x = (x1 * l + x2 * (1. - l))
#             x = x - x.mean()
#             return x, f1, (y1 * l + y2 * (1. - l))
#         return self.dataset[index]
#
#     def __len__(self):
#         return len(self.dataset)


class QueryAudioDataset(Dataset):

    def __init__(self, audio_feature, data_df, query_col, vocabulary=None, audio_feat_type='mel',
                 use_sentence_embeddings=False, sentence_embeddings=None,
                 use_auto_caption_embeddings=False, auto_caption_embeddings=None,
                 use_scene_passt_embeddings=False, scene_audio_feats=None):

        self.audio_feature = audio_feature
        self.data_df = data_df
        self.query_col = query_col
        self.vocabulary = vocabulary
        self.audio_feat_type = audio_feat_type
        self.use_sentence_embeddings = use_sentence_embeddings
        self.sentence_embeddings = sentence_embeddings
        self.use_auto_caption_embeddings = use_auto_caption_embeddings
        self.auto_caption_embeddings = auto_caption_embeddings
        self.use_scene_passt_embeddings = use_scene_passt_embeddings
        self.scene_audio_feats = scene_audio_feats

    def __getitem__(self, index):
        item = self.data_df.iloc[index]

        audio_feat = torch.as_tensor(self.audio_feature[str(item["fid"])][()])
        if self.audio_feat_type == 'raw':
            audio_feat = torch.squeeze(audio_feat)

        if self.vocabulary is not None:
            query = torch.as_tensor([self.vocabulary(token) for token in item[self.query_col]])
        else: query = None

        info = {"cid": item["cid"], "fid": item["fid"], "fname": item["fname"], "caption": item["original"], "fnames": {item["fid"]}}

        if self.use_sentence_embeddings:
            query_sent_embed = torch.as_tensor(self.sentence_embeddings[str(item["cid"])][()])
            # print("type(query_sent_embed)", type(query_sent_embed))
            if self.use_auto_caption_embeddings:
                auto_caption_embed = torch.as_tensor(self.auto_caption_embeddings[str(item["fname"])][()])
                return audio_feat, query, info, query_sent_embed, auto_caption_embed
            if self.use_scene_passt_embeddings:
                scene_passt_embed = torch.as_tensor(self.scene_audio_feats[str(item["fid"])][()])
                return audio_feat, query, info, query_sent_embed, scene_passt_embed
            return audio_feat, query, info, query_sent_embed

        return audio_feat, query, info

    def __len__(self):
        return len(self.data_df)



class QueryAudioMixupDataset(Dataset):

    def __init__(self, dataset, beta=0.4, rate=0.5):
        self.beta = beta
        self.rate = rate
        self.dataset = dataset
        self.use_sentence_embeddings = dataset.use_sentence_embeddings
        self.use_auto_caption_embeddings = dataset.use_auto_caption_embeddings
        self.auto_caption_embeddings = dataset.auto_caption_embeddings

        print(f"Using MixUp with rate=%.1f and beta=%.1f"%(rate, beta))

    def __getitem__(self, index):

        if torch.rand(1) < self.rate:

            l = np.random.beta(self.beta, self.beta)
            # l = max(l, 1. - l)

            index2 = torch.randint(len(self.dataset), (1,)).item()

            if self.use_sentence_embeddings:

                if self.use_auto_caption_embeddings:

                    audio_feat, query, info, query_sent_embed, auto_caption_embed = self.dataset[index]
                    audio_feat2, query2, info2, query_sent_embed2, auto_caption_embed2 = self.dataset[index2]
                    while info["fid"] == info2["fid"]:
                        index2 = torch.randint(len(self.dataset), (1,)).item()
                        audio_feat2, query2, info2, query_sent_embed2, auto_caption_embed2 = self.dataset[index2]

                    # print("l=%.3f, i=%i, i2=%i"%(l, index, index2))
                    # print("audio", audio_feat.size(), audio_feat2.size())
                    mixup_audio_feat = audio_feat * l + audio_feat2 * (1. - l)
                    # print("query", query.size(), query2.size())

                    # I cannot mixup the word-level queries from the baseline challenge approach without padding them for them to have the same length...
                    mixup_query = query_sent_embed  # * l + query_sent_embed2 * (1. - l)

                    mixup_query_sent_embed = F.normalize(query_sent_embed * l + query_sent_embed2 * (1. - l), p=2,
                                                         dim=-1)
                    mixup_auto_caption_embed = F.normalize(auto_caption_embed * l + auto_caption_embed2 * (1. - l), p=2,
                                                         dim=-1)

                    mixup_info = {"cid": info["cid"], "fid": info["fid"], "fname": info["fname"],
                                  "caption": info["caption"],
                                  "cid2": info2["cid"], "fid2": info2["fid"], "fname2": info2["fname"],
                                  "caption2": info2["caption"],
                                  "fnames": {info["fname"], info2["fname"]}, "lambda": l}
                    return mixup_audio_feat, mixup_query, mixup_info, mixup_query_sent_embed, mixup_auto_caption_embed
                else:
                    audio_feat, query, info, query_sent_embed = self.dataset[index]
                    audio_feat2, query2, info2, query_sent_embed2 = self.dataset[index2]
                    while info["fid"] == info2["fid"]:
                        index2 = torch.randint(len(self.dataset), (1,)).item()
                        audio_feat2, query2, info2, query_sent_embed2 = self.dataset[index2]

                    # print("l=%.3f, i=%i, i2=%i"%(l, index, index2))
                    # print("audio", audio_feat.size(), audio_feat2.size())
                    mixup_audio_feat = audio_feat * l + audio_feat2 * (1. - l)
                    # print("query", query.size(), query2.size())
                    mixup_query = query_sent_embed # * l + query_sent_embed2 * (1. - l)
                    mixup_query_sent_embed = F.normalize(query_sent_embed * l + query_sent_embed2 * (1. - l), p=2, dim=-1)
                    mixup_info = {"cid": info["cid"], "fid": info["fid"], "fname": info["fname"], "caption": info["caption"],
                                  "cid2": info2["cid"], "fid2": info2["fid"], "fname2": info2["fname"], "caption2": info2["caption"],
                                  "fnames": {info["fname"], info2["fname"]}, "lambda": l}
                    return mixup_audio_feat, mixup_query, mixup_info, mixup_query_sent_embed
            else:
                audio_feat, query, info = self.dataset[index]
                audio_feat2, query2, info2 = self.dataset[index2]
                while info["fid"] == info2["fid"]:
                    index2 = torch.randint(len(self.dataset), (1,)).item()
                    audio_feat2, query2, info2 = self.dataset[index2]

                mixup_audio_feat = audio_feat * l + audio_feat2 * (1. - l)
                mixup_query = query * l + query2 * (1. - l)
                mixup_info = {"cid": info["cid"], "fid": info["fid"], "fname": info["fname"], "caption": info["original"],
                              "cid2": info2["cid"], "fid2": info2["fid"], "fname2": info2["fname"], "caption2": info2["original"],
                              "fnames": {info["fname"], info2["fname"]}, "lambda": l}
                return mixup_audio_feat, mixup_query, mixup_info

            # x1 = x1-x1.mean()
            # x2 = x2-x2.mean()
            # x = (x1 * l + x2 * (1. - l))
            # x = x - x.mean()
            # return x, f1, (y1 * l + y2 * (1. - l))
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class AudiocapsQueryAudioDataset(Dataset):

    def __init__(self, audio_feature, data_df, sentence_embeddings=None):

        self.audio_feature = audio_feature
        self.data_df = data_df
        self.sentence_embeddings = sentence_embeddings

    def __getitem__(self, index):
        item = self.data_df.iloc[index]

        audio_feat = torch.as_tensor(self.audio_feature[str(item["audiocap_id"])][()])

        info = {"cid": item["audiocap_id"], "fid": item["audiocap_id"], "fname": item["youtube_id"], "caption": item["caption"]}

        query_sent_embed = torch.as_tensor(self.sentence_embeddings[str(item["audiocap_id"])][()])

        # fake word2vec query
        word_query = None

        return audio_feat, word_query, info, query_sent_embed

    def __len__(self):
        return len(self.data_df)


class EvalQueryAudioDataset(Dataset):

    def __init__(self, audio_feature, data_df, sentence_embeddings=None, scene_passt_embeddings=None):

        self.audio_feature = audio_feature
        self.data_df = data_df
        self.sentence_embeddings = sentence_embeddings
        self.scene_passt_embeddings = scene_passt_embeddings

    def __getitem__(self, index):
        item = self.data_df.iloc[index]

        # print(item)

        audio_feat = torch.as_tensor(self.audio_feature[str(item["fid"])][()])

        info = {"cid": item["cid"], "fid": item["fid"], "fname": item["fname"], "caption": item["caption"]}

        query_sent_embed = torch.as_tensor(self.sentence_embeddings[str(item["cid"])][()])

        scene_passt_embeddings = torch.as_tensor(self.scene_passt_embeddings[str(item["cid"])][()])

        # fake word2vec query
        word_query = None

        return audio_feat, word_query, info, query_sent_embed, scene_passt_embeddings

    def __len__(self):
        return len(self.data_df)


def collate_fn(data_batch):
    """
    :param data_batch: a list of tensor tuples (audio_feat, query, info).
    :return:
    """
    audio_feat_batch = []
    query_batch = []
    info_batch = []

    for a, q, i in data_batch:
        audio_feat_batch.append(a)
        query_batch.append(q)
        info_batch.append(i)

    audio_feat_batch, audio_feat_lens = pad_tensors(audio_feat_batch)
    query_batch, query_lens = pad_tensors(query_batch)

    return audio_feat_batch.float(), audio_feat_lens, query_batch.long(), query_lens, info_batch


def audiocaps_collate_fn_sentence_embeddings(data_batch):
    """
    :param data_batch: a list of tensor tuples (audio_feat, query, info, query_sent_embed).
    :return:
    """
    audio_feat_batch = []
    query_batch = []
    info_batch = []
    query_sent_batch = []
    # print(data_batch[0])

    for a, q, i, s in data_batch:
        audio_feat_batch.append(a)
        # query_batch.append(q)
        info_batch.append(i)
        query_sent_batch.append(s)

    audio_feat_batch, audio_feat_lens = pad_tensors(audio_feat_batch)
    # query_batch, query_lens = pad_tensors(query_batch)
    # query_sent_batch = torch.from_numpy(np.array(query_sent_batch))
    query_sent_batch, _ = pad_tensors(query_sent_batch)
    query_batch, query_lens = None, None

    return audio_feat_batch.float(), audio_feat_lens, query_batch, query_lens, info_batch, query_sent_batch.float()


def collate_fn_sentence_embeddings(data_batch):
    """
    :param data_batch: a list of tensor tuples (audio_feat, query, info, query_sent_embed).
    :return:
    """
    audio_feat_batch = []
    query_batch = []
    info_batch = []
    query_sent_batch = []
    # print(data_batch[0])

    for a, q, i, s in data_batch:
        audio_feat_batch.append(a)
        query_batch.append(q)
        info_batch.append(i)
        query_sent_batch.append(s)

    audio_feat_batch, audio_feat_lens = pad_tensors(audio_feat_batch)
    query_batch, query_lens = pad_tensors(query_batch)
    # query_sent_batch = torch.from_numpy(np.array(query_sent_batch))
    query_sent_batch, _ = pad_tensors(query_sent_batch)

    return audio_feat_batch.float(), audio_feat_lens, query_batch.long(), query_lens, info_batch, query_sent_batch.float()


def collate_fn_sentence_and_scene_embeddings(data_batch):
    """
    :param data_batch: a list of tensor tuples (audio_feat, query, info, query_sent_embed).
    :return:
    """
    audio_feat_batch = []
    query_batch = []
    info_batch = []
    query_sent_batch = []
    scene_embed_batch = []

    # print(data_batch[0])

    for a, q, i, s, p in data_batch:
        audio_feat_batch.append(a)
        query_batch.append(q)
        info_batch.append(i)
        query_sent_batch.append(s)
        scene_embed_batch.append(p)

    audio_feat_batch, audio_feat_lens = pad_tensors(audio_feat_batch)
    query_batch, query_lens = pad_tensors(query_batch)
    # query_sent_batch = torch.from_numpy(np.array(query_sent_batch))
    query_sent_batch, _ = pad_tensors(query_sent_batch)
    scene_embed_batch, _ = pad_tensors(scene_embed_batch)

    return audio_feat_batch.float(), audio_feat_lens, query_batch.long(), query_lens, info_batch, query_sent_batch.float(), scene_embed_batch.float()


def collate_fn_sentence_and_auto_caption_embeddings(data_batch):
    """
    :param data_batch: a list of tensor tuples (audio_feat, query, info, query_sent_embed).
    :return:
    """
    audio_feat_batch = []
    query_batch = []
    info_batch = []
    query_sent_batch = []
    auto_capt_batch = []
    # print(data_batch[0])

    for a, q, i, s, ac in data_batch:
        audio_feat_batch.append(a)
        query_batch.append(q)
        info_batch.append(i)
        query_sent_batch.append(s)
        auto_capt_batch.append(ac)

    audio_feat_batch, audio_feat_lens = pad_tensors(audio_feat_batch)
    query_batch, query_lens = pad_tensors(query_batch)
    # query_sent_batch = torch.from_numpy(np.array(query_sent_batch))
    query_sent_batch, _ = pad_tensors(query_sent_batch)
    auto_capt_batch, _ = pad_tensors(auto_capt_batch)

    return audio_feat_batch.float(), audio_feat_lens, query_batch.long(), query_lens, info_batch, query_sent_batch.float(), auto_capt_batch.float()


def pad_tensors(tensor_list):
    tensor_lens = [tensor.shape for tensor in tensor_list]

    dim_max_lens = tuple(np.max(tensor_lens, axis=0))

    tensor_lens = np.array(tensor_lens)[:, 0]

    padded_tensor = torch.zeros((len(tensor_list),) + dim_max_lens)
    for i, t in enumerate(tensor_list):
        end = tensor_lens[i]
        padded_tensor[i, :end] = t[:end]

    return padded_tensor, tensor_lens


def load_data(config):
    # Load audio features
    feats_path = os.path.join(config["input_path"], config["audio_features"])
    audio_feats = h5py.File(feats_path, "r")
    print("Load", feats_path)

    # Load pretrained word embeddings
    emb_path = os.path.join(config["input_path"], config["word_embeddings"])
    with open(emb_path, "rb") as emb_reader:
        word_vectors = pickle.load(emb_reader)
    print("Load", emb_path)

    if 'caption_embeddings' in config:
        use_sentence_embeddings = True
        sent_emb_path = os.path.join(config["input_path"], config["caption_embeddings"])
        sentence_embeddings = h5py.File(sent_emb_path, "r")
        print("Load", sent_emb_path)
    else:
        use_sentence_embeddings = False
        sentence_embeddings = None
        # : sentence_embeddings_MiniLM.hdf5

    # load Etienne's auto captions to be used in the audio_encoder part
    use_auto_caption_embeddings = config['use_auto_caption_embeddings']

    if use_auto_caption_embeddings:
        auto_caption_emb_path = os.path.join(config["input_path"], config["auto_caption_embeddings"])
        auto_caption_embeddings = h5py.File(auto_caption_emb_path, "r")
        print("Load", auto_caption_emb_path)
    else:
        auto_caption_embeddings = None

    use_scene_passt_embeddings = config['use_scene_passt_embeddings']

    if use_scene_passt_embeddings:
        feats_path = os.path.join(config["input_path"], config["scene_audio_feats"])
        scene_audio_feats = h5py.File(feats_path, "r")
        print("Load", feats_path)
    else:
        scene_audio_feats = None

    # Construct vocabulary
    vocabulary = Vocabulary()
    for word in word_vectors:
        if len(vocabulary) == 0:
            vocabulary.add_word("<pad>", np.zeros_like(word_vectors[word]))
        vocabulary.add_word(word, word_vectors[word])

    # Load data splits
    if 'raw_waveforms' in config["audio_features"]:
        audio_feat_type='raw'
    else: audio_feat_type='mel'

    text_datasets = {}

    for split in ["train", "val", "test"]:
        json_path = os.path.join(config["input_path"], config["data_splits"][split])
        df = pd.read_json(json_path)
        print("Load", json_path)

        dataset = QueryAudioDataset(audio_feats, df, config["text_tokens"], vocabulary, audio_feat_type,
                                     use_sentence_embeddings, sentence_embeddings,
                                    use_auto_caption_embeddings, auto_caption_embeddings,
                                    use_scene_passt_embeddings, scene_audio_feats)

        if split == 'train' and config["use_mixup"]:
            print("QueryAudioMixupDataset is used for the train subset!")
            dataset = QueryAudioMixupDataset(dataset, beta=config["beta"], rate=0.5)

        text_datasets[split] = dataset

    return text_datasets, vocabulary
