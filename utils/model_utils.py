import copy
import os

import torch

from models import core

import wandb

def get_model(config, vocabulary):
    model_args = copy.deepcopy(config["args"])

    if config["name"] in ["CRNNWordModel"] or config["name"] in ["PasstWordModel"]:
        embed_args = model_args["text_encoder"]["word_embedding"]
        embed_args["num_word"] = len(vocabulary)
        embed_args["word_embeds"] = vocabulary.get_weights() if embed_args["pretrained"] else None

        return getattr(core, config["name"], None)(**model_args)

    elif config["name"] in ["PasstSentModel"]:

        return getattr(core, config["name"], None)(**model_args)

    return None


def train(model, optimizer, criterion, data_loader, epoch, use_sentence_embeddings=False, use_auto_caption_embeddings=False, use_scene_passt_embeddings=False, log_wandb=False):

    # if mixup_lambdas is None:
    #     mixup_lambdas = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion.to(device=device)
    model.to(device=device)

    train_loss, train_steps = 0.0, 0

    model.train()
    if use_sentence_embeddings:
        if use_auto_caption_embeddings:
            for batch_idx, data in enumerate(data_loader, 0):
                # Get the inputs; data is a list of tuples (audio_feats, audio_lens, queries, query_lens, infos)
                audio_feats, audio_lens, queries, query_lens, infos, caption_embeds, auto_captions_embeds = data
                audio_feats, caption_embeds, auto_captions_embeds = audio_feats.to(device), caption_embeds.to(device), auto_captions_embeds.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward + backward + optimize
                audio_embeds, query_embeds = model([audio_feats, auto_captions_embeds], caption_embeds)
                loss = criterion(audio_embeds, query_embeds, infos)
                # print("   batch", batch_idx, "loss=%.3f"%loss)
                if log_wandb:
                    wandb.log({"train_step_loss": loss})
                    wandb.log({"epoch": epoch})

                loss.backward()
                # print("bck done")
                optimizer.step()
                # print("step done")
                # for el in infos:
                #     if "lambda" in el:
                #         mixup_lambdas.append(el["lambda"])
                # print(mixup_lambdas)
                # break
                train_loss += loss.clone().detach().cpu().numpy()
                train_steps += 1

        elif use_scene_passt_embeddings:

            for batch_idx, data in enumerate(data_loader, 0):
                # Get the inputs; data is a list of tuples (audio_feats, audio_lens, queries, query_lens, infos)
                audio_feats, audio_lens, queries, query_lens, infos, caption_embeds, scene_embeds = data
                audio_feats, caption_embeds, scene_embeds = audio_feats.to(device), caption_embeds.to(device), scene_embeds.to(device)

                # print(audio_feats.size())

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward + backward + optimize
                audio_embeds, query_embeds, audio_embeds2 = model([audio_feats, scene_embeds], caption_embeds)
                loss = criterion(audio_embeds, query_embeds, infos, is_train_subset=True)
                # print("   batch", batch_idx, "loss=%.3f"%loss)

                # wandb.log({"train_step_loss": loss})

                # if batch_idx < 1 :
                #     for index in range(10):
                #         print('norm(audio_embed)', torch.norm(audio_embeds[index], p=2).item(), 'norm(audio_embed2)', torch.norm(audio_embeds2[index], p=2).item())

                loss.backward()
                # print("bck done")
                optimizer.step()
                # print("step done")
                train_loss += loss.clone().detach().cpu().numpy()
                train_steps += 1
                if log_wandb:
                    wandb.log({"train_step_loss": loss})
                    wandb.log({"epoch": epoch})

        else:
            for batch_idx, data in enumerate(data_loader, 0):
                # Get the inputs; data is a list of tuples (audio_feats, audio_lens, queries, query_lens, infos)
                audio_feats, audio_lens, queries, query_lens, infos, caption_embeds = data
                audio_feats, caption_embeds = audio_feats.to(device), caption_embeds.to(device)

                # print(audio_feats.size())

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward + backward + optimize
                audio_embeds, query_embeds, audio_embeds2 = model(audio_feats, caption_embeds)
                loss = criterion(audio_embeds, query_embeds, infos, is_train_subset=True)
                # print("   batch", batch_idx, "loss=%.3f"%loss)

                # wandb.log({"train_step_loss": loss})

                # if batch_idx < 1 :
                #     for index in range(10):
                #         print('norm(audio_embed)', torch.norm(audio_embeds[index], p=2).item(), 'norm(audio_embed2)', torch.norm(audio_embeds2[index], p=2).item())

                loss.backward()
                # print("bck done")
                optimizer.step()
                # print("step done")
                train_loss += loss.clone().detach().cpu().numpy()
                train_steps += 1
                if log_wandb:
                    wandb.log({"train_step_loss": loss})
                    wandb.log({"epoch": epoch})

    else:
        for batch_idx, data in enumerate(data_loader, 0):
                # Get the inputs; data is a list of tuples (audio_feats, audio_lens, queries, query_lens, infos)
                audio_feats, audio_lens, queries, query_lens, infos = data
                audio_feats, queries = audio_feats.to(device), queries.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward + backward + optimize
                audio_embeds, query_embeds = model(audio_feats, queries, query_lens)
                loss = criterion(audio_embeds, query_embeds, infos)
                # print("   batch", batch_idx, "loss=%.3f"%loss)

                loss.backward()
                # print("bck done")
                optimizer.step()
                # print("step done")
                train_loss += loss.clone().detach().cpu().numpy()
                train_steps += 1

                if log_wandb:
                    wandb.log({"train_step_loss": loss})
                    wandb.log({"epoch": epoch})

    return train_loss / (train_steps + 1e-20)


def eval(model, criterion, data_loader, use_sentence_embeddings=False, use_auto_caption_embeddings=False, use_scene_passt_embeddings=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion.to(device=device)
    model.to(device=device)

    model.eval()

    eval_loss, eval_steps = 0.0, 0
    if use_sentence_embeddings:
        if use_auto_caption_embeddings:
            with torch.no_grad():
                for batch_idx, data in enumerate(data_loader, 0):
                    audio_feats, audio_lens, queries, query_lens, infos, caption_embeds, auto_captions_embeds = data
                    audio_feats, caption_embeds, auto_captions_embeds = audio_feats.to(device), caption_embeds.to(
                        device), auto_captions_embeds.to(device)

                    audio_embeds, query_embeds = model([audio_feats, auto_captions_embeds], caption_embeds)

                    loss = criterion(audio_embeds, query_embeds, infos)
                    eval_loss += loss.cpu().numpy()
                    eval_steps += 1
        elif use_scene_passt_embeddings:

            with torch.no_grad():
                for batch_idx, data in enumerate(data_loader, 0):
                    audio_feats, audio_lens, queries, query_lens, infos, caption_embeds, scene_embeds = data
                    audio_feats, caption_embeds, scene_embeds = audio_feats.to(device), caption_embeds.to(
                        device), scene_embeds.to(device)

                    audio_embeds, query_embeds, audio_embeds2 = model([audio_feats, scene_embeds], caption_embeds)

                    loss = criterion(audio_embeds, query_embeds, infos)
                    eval_loss += loss.cpu().numpy()
                    eval_steps += 1


        else:
            with torch.no_grad():
                for batch_idx, data in enumerate(data_loader, 0):
                    audio_feats, audio_lens, queries, query_lens, infos, caption_embeds = data
                    audio_feats, caption_embeds = audio_feats.to(device), caption_embeds.to(device)

                    audio_embeds, query_embeds, audio_embeds2 = model(audio_feats, caption_embeds)

                    loss = criterion(audio_embeds, query_embeds, infos)
                    eval_loss += loss.cpu().numpy()
                    eval_steps += 1
    else:
        with torch.no_grad():
            for batch_idx, data in enumerate(data_loader, 0):
                audio_feats, audio_lens, queries, query_lens, infos = data
                audio_feats, queries = audio_feats.to(device), queries.to(device)

                audio_embeds, query_embeds = model(audio_feats, queries, query_lens)

                loss = criterion(audio_embeds, query_embeds, infos)
                eval_loss += loss.cpu().numpy()
                eval_steps += 1

    return eval_loss / (eval_steps + 1e-20)


def restore(model, checkpoint_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"), map_location=device)
    model.load_state_dict(model_state)
    return model
