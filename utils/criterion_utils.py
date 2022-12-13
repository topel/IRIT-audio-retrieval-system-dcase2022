import numpy as np
import torch
import torch.nn as nn
import wandb

# import matplotlib.pyplot as plt

class TripletRankingLoss(nn.Module):

    def __init__(self, margin=1.0, log_wandb=False):
        super().__init__()

        self.margin = margin
        self.log_wandb = log_wandb
        print("TripletRankingLoss margin:", margin)

    def forward(self, audio_embeds, query_embeds, infos, is_train_subset=False):
        """
        :param audio_embeds: tensor, (N, E).
        :param query_embeds: tensor, (N, E).
        :param infos: list of audio infos.
        :return:
        """
        N = audio_embeds.size(0)

        # Computes the triplet margin ranking loss for each anchor audio/query pair.
        # The impostor audio/query is randomly sampled from the mini-batch.
        loss = torch.tensor(0., device=audio_embeds.device, requires_grad=True)

        # if is_train_subset and self.log_wandb:
        # anchor_scores, A_imp_scores, Q_imp_scores = [], [], []

        for i in range(N):
            A_imp_idx = i
            while infos[A_imp_idx]["fid"] == infos[i]["fid"]:
                A_imp_idx = np.random.randint(0, N)

            Q_imp_idx = i
            while infos[Q_imp_idx]["fid"] == infos[i]["fid"]:
                Q_imp_idx = np.random.randint(0, N)

            # using the
            # Q_imp_idx = A_imp_idx

            anchor_score = score(audio_embeds[i], query_embeds[i])

            A_imp_score = score(audio_embeds[A_imp_idx], query_embeds[i])

            Q_imp_score = score(audio_embeds[i], query_embeds[Q_imp_idx])

            # if is_train_subset and self.log_wandb:
            # anchor_scores.append(anchor_score.item())
            # A_imp_scores.append(A_imp_score.item())
            # Q_imp_scores.append(Q_imp_score.item())

            A2Q_diff = self.margin + Q_imp_score - anchor_score
            if (A2Q_diff.data > 0.).all():
                loss = loss + A2Q_diff

            Q2A_diff = self.margin + A_imp_score - anchor_score
            if (Q2A_diff.data > 0.).all():
                loss = loss + Q2A_diff

        loss = loss / N


        # plt.hist(anchor_scores)
        # plt.ylabel("Anchor scores")
        # plt.savefig("toto.png")
        # print("toto.png")

        # if is_train_subset and self.log_wandb:
        #     # hist = np.histogram(anchor_scores)
        #     # wandb.Histogram(np_histogram=hist)
        #     # hist = np.histogram(A_imp_scores)
        #     # wandb.Histogram(np_histogram=hist)
        #     # hist = np.histogram(Q_imp_scores)
        #     # wandb.Histogram(np_histogram=hist)
        #
        #     # combined_scores = [[el1, el2, el3] for el1, el2, el3 in zip(anchor_scores, A_imp_scores, Q_imp_scores)]
        #     # table = wandb.Table(data=combined_scores, columns=["Anchor", "Aimp", "Qimp"])
        #     # histogram = wandb.plot.histogram(table, value='Anchor', title='Scores Anchor')
        #     # wandb.log({'score_histo': histogram})
        #
        #     wandb.log({"Anchor hist":
        #                    wandb.Histogram(np.array(anchor_scores))})
        #     wandb.log({"A imp hist":
        #                    wandb.Histogram(np.array(A_imp_scores))})
        #     wandb.log({"Q imp hist":
        #                    wandb.Histogram(np.array(Q_imp_scores))})
        #
        #     # wandb.log({'scores': table})
        #
        #     # wandb.log({'my_histogram': wandb.plot.histogram(table, "scores",
        #     # title ="Anchor score Distribution")})
        #
        #     # fig, ax = plt.subplots()
        #     # ax.hist(anchor_scores, bins=20, rwidth=0.9,  alpha=0.2, label='anchor')
        #     # ax.set_ylabel("Scores")
        #     # ax.hist(A_imp_scores, bins=20, rwidth=0.9, alpha=0.2, label='A imp')
        #     # ax.hist(Q_imp_scores, bins=20, rwidth=0.9, alpha=0.2, label='Q imp')
        #     # ax.legend()
        #     # # Log the plot
        #     # wandb.log({"plot": fig})

        # return loss, [anchor_scores, A_imp_scores, Q_imp_scores]
        return loss


class MixupTripletRankingLoss(nn.Module):

    def __init__(self, margin=1.0):
        super().__init__()

        self.margin = margin

    def forward(self, audio_embeds, query_embeds, infos):
        """
        :param audio_embeds: tensor, (N, E).
        :param query_embeds: tensor, (N, E).
        :param infos: list of audio infos.
        :return:
        """
        N = audio_embeds.size(0)

        # Computes the triplet margin ranking loss for each anchor audio/query pair.
        # The impostor audio/query is randomly sampled from the mini-batch.
        loss = torch.tensor(0., device=audio_embeds.device, requires_grad=True)

        for i in range(N):
            A_imp_idx = i
            while not infos[A_imp_idx]["fnames"].isdisjoint(infos[i]["fnames"]):
                A_imp_idx = np.random.randint(0, N)

            Q_imp_idx = i
            while not infos[Q_imp_idx]["fnames"].isdisjoint(infos[i]["fnames"]):
                Q_imp_idx = np.random.randint(0, N)

            anchor_score = score(audio_embeds[i], query_embeds[i])

            A_imp_score = score(audio_embeds[A_imp_idx], query_embeds[i])

            Q_imp_score = score(audio_embeds[i], query_embeds[Q_imp_idx])

            A2Q_diff = self.margin + Q_imp_score - anchor_score
            if (A2Q_diff.data > 0.).all():
                loss = loss + A2Q_diff

            Q2A_diff = self.margin + A_imp_score - anchor_score
            if (Q2A_diff.data > 0.).all():
                loss = loss + Q2A_diff

        loss = loss / N

        return loss


def score(audio_embed, query_embed):
    """
    Compute an audio-query score.

    :param audio_embed: tensor, (E, ).
    :param query_embed: tensor, (E, ).
    :return: similarity score: tensor, (1, ).
    """

    sim = torch.dot(audio_embed, query_embed)

    return sim
