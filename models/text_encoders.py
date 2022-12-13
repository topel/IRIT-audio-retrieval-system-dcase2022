import torch
import torch.nn as nn
import torch.nn.functional as F

class WordEmbedding(nn.Module):

    def __init__(self, *args, **kwargs):
        super(WordEmbedding, self).__init__()

        self.args = args
        self.kwargs = kwargs

        self.embedding = nn.Embedding(kwargs["num_word"], kwargs["embed_dim"])

        if kwargs.get("word_embeds", None) is not None:
            self.load_pretrained_embedding(kwargs["word_embeds"])
        else:
            nn.init.kaiming_uniform_(self.embedding.weight)

        for para in self.embedding.parameters():
            para.requires_grad = kwargs.get("trainable", False)

    def load_pretrained_embedding(self, weight):
        assert weight.shape[0] == self.embedding.weight.size()[0], "vocabulary size mismatch!"

        weight = torch.as_tensor(weight).float()
        self.embedding.weight = nn.Parameter(weight)

    def forward(self, queries, query_lens):
        """
        :param queries: tensor, (batch_size, query_max_len).
        :param query_lens: list, [N_{1}, ..., N_{batch_size}].
        :return: (batch_size, query_max_len, embed_dim).
        """

        query_lens = torch.as_tensor(query_lens)
        batch_size, query_max = queries.size()

        query_embeds = self.embedding(queries)

        mask = torch.arange(query_max, device="cpu").repeat(batch_size).view(batch_size, query_max)
        mask = (mask < query_lens.view(-1, 1)).to(query_embeds.device)

        query_embeds = query_embeds * mask.unsqueeze(-1)

        return query_embeds


class WordEncoder(nn.Module):

    def __init__(self, *args, **kwargs):
        super(WordEncoder, self).__init__()

        self.args = args
        self.kwargs = kwargs

        self.word_embedding = WordEmbedding(*args, **kwargs["word_embedding"])

        self.dropout_input = nn.Dropout(p=0.2)

        # self.bn = nn.BatchNorm1d(kwargs["word_embedding"]["embed_dim"])
        # init_weights(self.bn)

        # self.rnn = nn.GRU(kwargs["word_embedding"]["embed_dim"], kwargs["word_embedding"]["embed_dim"], num_layers=1, batch_first=True, dropout=0, bidirectional=False)
        # init_weights(self.rnn)

    def forward(self, queries, query_lens):
        """
        :param queries: tensor, (batch_size, query_max_len).
        :param query_lens: list, [N_{1}, ..., N_{batch_size}].
        :return: (batch_size, embed_dim).
        """

        query_embeds = self.word_embedding(queries, query_lens)
        # print('1', query_embeds.size())
        # query_embeds = self.bn(query_embeds.transpose(1, 2))
        # query_embeds = query_embeds.transpose(1, 2)
        # print('2', query_embeds.size())

        # _, query_embeds = self.rnn(query_embeds)
        query_embeds = torch.mean(query_embeds, dim=1, keepdim=False)
        query_embeds = torch.squeeze(query_embeds)
        # print('2', query_embeds.size())

        query_embeds = self.dropout_input(query_embeds)

        return query_embeds


class SentEncoder(nn.Module):

    def __init__(self, *args, **kwargs):
        super(SentEncoder, self).__init__()

        self.args = args
        self.kwargs = kwargs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.embed_dim = kwargs["word_embedding"]["embed_dim"]
        self.layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False)

        # self.proj_dim = kwargs["out_dim"]
        # self.bn = nn.BatchNorm1d(self.embed_dim)
        # init_weights(self.bn)
        # self.dropout_input = nn.Dropout(p=0.2)
        #
        # self.projection_layer = nn.Linear(self.embed_dim, self.proj_dim)
        # init_weights(self.projection_layer)

    def forward(self, caption_embeds):
        """
        :param caption_embeds: tensor, (batch_size, caption_embed_dim).
        :return: (batch_size, embed_dim).
        """

        # print('1', caption_embeds.size())
        caption_embeds = self.layer_norm(caption_embeds)
        # caption_embeds = self.projection_layer(caption_embeds)
        caption_embeds = F.normalize(caption_embeds, p=2, dim=-1)

        # caption_embeds = self.bn(caption_embeds.transpose(1, 2))
        # caption_embeds = caption_embeds.transpose(1, 2)
        # print('2', query_embeds.size())

        # _, query_embeds = self.rnn(query_embeds)
        # query_embeds = torch.mean(query_embeds, dim=1, keepdim=False)
        # query_embeds = torch.squeeze(query_embeds)
        # print('2', query_embeds.size())

        # query_embeds = self.dropout_input(query_embeds)

        return caption_embeds


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Conv1d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
