import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# import sys
# sys.path.append("/homelocal/thomas/research/dcase2022/hear21passt")
# import hear21passt.base as hear21passt

import os
import h5py

import wandb

class PasstEncoder(nn.Module):

    def __init__(self, *args, **kwargs):
        super(PasstEncoder, self).__init__()

        self.args = args
        self.kwargs = kwargs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.embed_dim = kwargs["out_dim"]
        # # self.passt_encoder = hear21passt.load_model(mode="logits")
        # # self.dropout_input = nn.Dropout(p=0.2)
        #
        # self.bn = nn.BatchNorm1d(kwargs["passt_dim"])
        # init_weights(self.bn)
        # self.bn_audioset_sentences = nn.BatchNorm1d(self.embed_dim)

        self.layer_norm = nn.LayerNorm(kwargs["passt_dim"], elementwise_affine=True)
        # self.layer_norm_audioset_sentences = nn.LayerNorm(self.embed_dim, elementwise_affine=True)

        self.projection_layer = nn.Linear(kwargs["passt_dim"], self.embed_dim)
        init_weights(self.projection_layer)

        self.scene_embedding_layer_norm = nn.LayerNorm(kwargs["scene_embed_dim"], elementwise_affine=True)
        self.scene_embedding_projection_layer = nn.Linear(kwargs["scene_embed_dim"], self.embed_dim)
        init_weights(self.scene_embedding_projection_layer)

        if "ratio_embed1" in kwargs:
            self.ratio_embed1 = kwargs["ratio_embed1"]
        else:
            # here it's the case where we use sweep from wandb in training mode
            self.ratio_embed1 = wandb.config["ratio_embed1"]
        print("audio encoder ratio_embed1:", self.ratio_embed1)

        self.ratio_embed2 = 1 - self.ratio_embed1

        # self.pointwise_conv = nn.Conv1d(2, 1, 1, bias=False)
        # self.pointwise_conv.weight.data.fill_(0.5)
        # hidden_dim = 50
        #
        # self.fc1 = nn.Linear(kwargs["passt_dim"], hidden_dim)
        # init_weights(self.fc1)
        # self.bn1 = nn.BatchNorm1d(hidden_dim)
        # init_weights(self.bn1)
        # self.gelu = nn.GELU()
        #
        # self.fc2 = nn.Linear(hidden_dim, kwargs["out_dim"])
        # init_weights(self.fc2)
        # self.bn2 = nn.BatchNorm1d(kwargs["out_dim"])
        # init_weights(self.bn2)

        # freeze PASST
        # for param in self.passt_encoder.parameters():
        #     param.requires_grad = False

        # fpath = '/baie/corpus/DCASE2022/Clotho.v2.1/audioset_527_sentence_embeddings_MiniLM.hdf5'

        # Legolas
        # fpath = '/baie/corpus/DCASE2022/Clotho.v2.1/audioset_527_sentence_embeddings_mpnet.hdf5'
        # tags only
        # fpath = '/baie/corpus/DCASE2022/Clotho.v2.1/audioset_527_tags_embeddings_mpnet.hdf5'
        # audioset sentences without the tags
        # fpath = '/baie/corpus/DCASE2022/Clotho.v2.1/audioset_527_sentenceNoTags_embeddings_mpnet.hdf5'

        # on choro
        # fpath = '/home/pellegri/research/PROJETS/dcase2022/mydcase2022-audio-retrieval/Clotho.v2.1/audioset_527_sentence_embeddings_MiniLM.hdf5'
        # fpath = '/home/pellegri/research/PROJETS/dcase2022/mydcase2022-audio-retrieval/Clotho.v2.1/audioset_527_sentence_embeddings_mpnet.hdf5'

        # on osirim
        # audioset sentences with the tags
        # sentences with tags
        # fpath = '/projets/samova/pellegri/dcase2022/Clotho.v2.1/audioset_527_sentence_embeddings_mpnet.hdf5'
        # tags only
        # fpath = '/projets/samova/pellegri/dcase2022/Clotho.v2.1/audioset_527_tags_embeddings_mpnet.hdf5'
        # audioset sentences without the tags
        # fpath = '/projets/samova/pellegri/dcase2022/Clotho.v2.1/audioset_527_sentenceNoTags_embeddings_mpnet.hdf5'

        fpath = os.path.join(kwargs["audioset_path"], kwargs["audioset_embeddings"])

        hfile = h5py.File(fpath, "r")

        # self.audioset_sent_embeds = torch.zeros(527, 384)
        self.audioset_sent_embeds = torch.zeros(527, 768)
        for ind in range(527):
            self.audioset_sent_embeds[ind] = torch.as_tensor(hfile[str(ind)][()])
        self.audioset_sent_embeds = self.audioset_sent_embeds.to(self.device)
        # self.audioset_sent_embeds.requires_grad_()

        print("Load", fpath)

    def forward(self, x):
        """
        :param x: tensor, (batch_size, time_steps). RAW WAVEFORM!
        :return: tensor, (batch_size, embed_dim).
        """
        # batch, time = x.shape
        # with torch.no_grad():
        #     embed = hear21passt.get_scene_embeddings(x, self.passt_encoder)
        # print('1', embed.size())
        # embed = embed.to(self.device)
        # embed = self.projection_layer(embed)

        # when using auto captions
        # passt_embed, auto_caption_embed = x
        # passt_embed = x

        # print(len(x))
        logits, scene_embeds = x
        # print(logits.size(), scene_embeds.size())

        with torch.no_grad():
        #### predicted_audioset_class = torch.argmax(logits, dim=-1)
            predicted_audioset_probs = torch.sigmoid(logits)
            ## embed2 = self.audioset_sent_embeds[predicted_audioset_class]

            # embed2 = self.layer_norm_audioset_sentences(embed2)
            # embed2 = self.bn_audioset_sentences(embed2)

            predicted_audioset_probs = predicted_audioset_probs.to(self.device)
            denom = torch.sum(predicted_audioset_probs, dim=-1, keepdim=True)
            denom = denom.to(self.device)
            embed2 = torch.matmul(predicted_audioset_probs, self.audioset_sent_embeds)
            embed2 = embed2 / denom
            embed2 = embed2.to(self.device)
            embed2 = F.normalize(embed2, p=2, dim=-1)

        ### embed = self.bn(passt_embed)
        ### embed = self.projection_layer(embed)

        # x = self.bn(passt_embed)
        # print(passt_embed[0])
        # x = torch.sigmoid(passt_embed)

        # embed = self.layer_norm(logits)
        embed3 = self.scene_embedding_layer_norm(scene_embeds)

        # embed = self.bn(x)

        # print(x[0])
        # x =  F.normalize(passt_embed, p=2, dim=-1)

        # embed = self.projection_layer(embed)
        # embed = F.normalize(embed, p=2, dim=-1)

        embed3 = self.scene_embedding_projection_layer(embed3)
        embed3 = F.normalize(embed3, p=2, dim=-1)

        # channel_embed = torch.zeros(passt_embed.size(0), 2, self.embed_dim)
        # channel_embed[:, 0] = embed
        # channel_embed[:, 1] = auto_caption_embed
        # channel_embed = channel_embed.to(self.device)
        # final_embed = self.pointwise_conv(channel_embed)
        # final_embed = torch.squeeze(F.normalize(final_embed, p=2, dim=-1))

        # embed = self.ratio_embed1 * embed + self.ratio_embed2 * embed2
        # embed = F.normalize(embed, p=2, dim=-1)

        # embed = 0.2 * embed + 0.8 * embed3
        # embed = F.normalize(embed, p=2, dim=-1)

        embed = 0.8 * embed3 + 0.2 * embed2
        embed = F.normalize(embed, p=2, dim=-1)

        # print('2', embed.size())
        # embed = self.dropout_input(embed)
        # print(x.size())

        # return embed, embed2
        return embed, embed3


class PasstEncoderSauvegarde(nn.Module):

    def __init__(self, *args, **kwargs):
        super(PasstEncoder, self).__init__()

        self.args = args
        self.kwargs = kwargs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.embed_dim = kwargs["out_dim"]
        # # self.passt_encoder = hear21passt.load_model(mode="logits")
        # # self.dropout_input = nn.Dropout(p=0.2)
        #
        # self.bn = nn.BatchNorm1d(kwargs["passt_dim"])
        # init_weights(self.bn)
        # self.bn_audioset_sentences = nn.BatchNorm1d(self.embed_dim)

        self.layer_norm = nn.LayerNorm(kwargs["passt_dim"], elementwise_affine=True)
        # self.layer_norm_audioset_sentences = nn.LayerNorm(self.embed_dim, elementwise_affine=True)

        self.projection_layer = nn.Linear(kwargs["passt_dim"], self.embed_dim)
        init_weights(self.projection_layer)

        if "ratio_embed1" in kwargs:
            self.ratio_embed1 = kwargs["ratio_embed1"]
        else:
            # here it's the case where we use sweep from wandb in training mode
            self.ratio_embed1 = wandb.config["ratio_embed1"]
        print("audio encoder ratio_embed1:", self.ratio_embed1)

        self.ratio_embed2 = 1 - self.ratio_embed1

        # self.pointwise_conv = nn.Conv1d(2, 1, 1, bias=False)
        # self.pointwise_conv.weight.data.fill_(0.5)
        # hidden_dim = 50
        #
        # self.fc1 = nn.Linear(kwargs["passt_dim"], hidden_dim)
        # init_weights(self.fc1)
        # self.bn1 = nn.BatchNorm1d(hidden_dim)
        # init_weights(self.bn1)
        # self.gelu = nn.GELU()
        #
        # self.fc2 = nn.Linear(hidden_dim, kwargs["out_dim"])
        # init_weights(self.fc2)
        # self.bn2 = nn.BatchNorm1d(kwargs["out_dim"])
        # init_weights(self.bn2)

        # freeze PASST
        # for param in self.passt_encoder.parameters():
        #     param.requires_grad = False

        # fpath = '/baie/corpus/DCASE2022/Clotho.v2.1/audioset_527_sentence_embeddings_MiniLM.hdf5'

        # Legolas
        # fpath = '/baie/corpus/DCASE2022/Clotho.v2.1/audioset_527_sentence_embeddings_mpnet.hdf5'
        # tags only
        # fpath = '/baie/corpus/DCASE2022/Clotho.v2.1/audioset_527_tags_embeddings_mpnet.hdf5'
        # audioset sentences without the tags
        # fpath = '/baie/corpus/DCASE2022/Clotho.v2.1/audioset_527_sentenceNoTags_embeddings_mpnet.hdf5'

        # on choro
        # fpath = '/home/pellegri/research/PROJETS/dcase2022/mydcase2022-audio-retrieval/Clotho.v2.1/audioset_527_sentence_embeddings_MiniLM.hdf5'
        # fpath = '/home/pellegri/research/PROJETS/dcase2022/mydcase2022-audio-retrieval/Clotho.v2.1/audioset_527_sentence_embeddings_mpnet.hdf5'

        # on osirim
        # audioset sentences with the tags
        # sentences with tags
        # fpath = '/projets/samova/pellegri/dcase2022/Clotho.v2.1/audioset_527_sentence_embeddings_mpnet.hdf5'
        # tags only
        # fpath = '/projets/samova/pellegri/dcase2022/Clotho.v2.1/audioset_527_tags_embeddings_mpnet.hdf5'
        # audioset sentences without the tags
        # fpath = '/projets/samova/pellegri/dcase2022/Clotho.v2.1/audioset_527_sentenceNoTags_embeddings_mpnet.hdf5'

        fpath = os.path.join(kwargs["audioset_path"], kwargs["audioset_embeddings"])

        hfile = h5py.File(fpath, "r")

        # self.audioset_sent_embeds = torch.zeros(527, 384)
        self.audioset_sent_embeds = torch.zeros(527, 768)
        for ind in range(527):
            self.audioset_sent_embeds[ind] = torch.as_tensor(hfile[str(ind)][()])
        self.audioset_sent_embeds = self.audioset_sent_embeds.to(self.device)
        # self.audioset_sent_embeds.requires_grad_()

        print("Load", fpath)

    def forward(self, x):
        """
        :param x: tensor, (batch_size, time_steps). RAW WAVEFORM!
        :return: tensor, (batch_size, embed_dim).
        """
        # batch, time = x.shape
        # with torch.no_grad():
        #     embed = hear21passt.get_scene_embeddings(x, self.passt_encoder)
        # print('1', embed.size())
        # embed = embed.to(self.device)
        # embed = self.projection_layer(embed)

        # when using auto captions
        # passt_embed, auto_caption_embed = x
        # passt_embed = x

        with torch.no_grad():
        #### predicted_audioset_class = torch.argmax(passt_embed, dim=-1)
            predicted_audioset_probs = torch.sigmoid(x)
            ## embed2 = self.audioset_sent_embeds[predicted_audioset_class]

            # embed2 = self.layer_norm_audioset_sentences(embed2)
            # embed2 = self.bn_audioset_sentences(embed2)

            predicted_audioset_probs = predicted_audioset_probs.to(self.device)
            denom = torch.sum(predicted_audioset_probs, dim=-1, keepdim=True)
            denom = denom.to(self.device)
            embed2 = torch.matmul(predicted_audioset_probs, self.audioset_sent_embeds)
            embed2 = embed2 / denom
            embed2 = embed2.to(self.device)
            embed2 = F.normalize(embed2, p=2, dim=-1)

        ### embed = self.bn(passt_embed)
        ### embed = self.projection_layer(embed)

        # x = self.bn(passt_embed)
        # print(passt_embed[0])
        # x = torch.sigmoid(passt_embed)

        embed = self.layer_norm(x)
        # embed = self.bn(x)

        # print(x[0])
        # x =  F.normalize(passt_embed, p=2, dim=-1)
        embed = self.projection_layer(embed)

        embed = F.normalize(embed, p=2, dim=-1)

        # channel_embed = torch.zeros(passt_embed.size(0), 2, self.embed_dim)
        # channel_embed[:, 0] = embed
        # channel_embed[:, 1] = auto_caption_embed
        # channel_embed = channel_embed.to(self.device)
        # final_embed = self.pointwise_conv(channel_embed)
        # final_embed = torch.squeeze(F.normalize(final_embed, p=2, dim=-1))

        embed = self.ratio_embed1 * embed + self.ratio_embed2 * embed2
        embed = F.normalize(embed, p=2, dim=-1)
        # print('2', embed.size())
        # embed = self.dropout_input(embed)
        # print(x.size())
        return embed, embed2


class CNNModule(nn.Module):

    def __init__(self):
        super(CNNModule, self).__init__()

        self.features = nn.Sequential(
            # Conv2D block
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),

            # LPPool
            nn.LPPool2d(4, (2, 4)),

            # Conv2D block
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 128, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),

            # Conv2D block
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),

            # LPPool
            nn.LPPool2d(4, (2, 4)),

            # Conv2D block
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),

            # Conv2D block
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),

            # LPPool
            nn.LPPool2d(4, (1, 4)),

            nn.Dropout(0.3)
        )

        self.features.apply(init_weights)

    def forward(self, x):
        """
        :param x: tensor, (batch_size, time_steps, Mel_bands).
        :return: tensor, (batch_size, time_steps / 4, 128 * Mel_bands / 64).
        """
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.transpose(1, 2).contiguous().flatten(-2)

        return x


class CRNNEncoder(nn.Module):

    def __init__(self, *args, **kwargs):
        super(CRNNEncoder, self).__init__()

        self.args = args
        self.kwargs = kwargs

        self.cnn = CNNModule()

        with torch.no_grad():
            rnn_in_dim = self.cnn(torch.randn(1, 500, kwargs["in_dim"])).shape
            rnn_in_dim = rnn_in_dim[-1]

        self.gru = nn.GRU(rnn_in_dim, kwargs["out_dim"] // 2, bidirectional=True, batch_first=True)

    def forward(self, x):
        """
        :param x: tensor, (batch_size, time_steps, Mel_bands).
        :return: tensor, (batch_size, embed_dim).
        """
        batch, time, dim = x.shape

        x = self.cnn(x)
        x, _ = self.gru(x)

        if self.kwargs.get("up_sampling", False):
            x = F.interpolate(x.transpose(1, 2), time, mode="linear", align_corners=False).transpose(1, 2)

        x = torch.mean(x, dim=1, keepdim=False)

        return x


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Conv1d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
