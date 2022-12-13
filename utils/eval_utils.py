import os

import pandas as pd
import torch

from utils import data_utils, model_utils

import h5py

def transform(model, dataset, index, device=None, use_sentence_embeddings=False, use_auto_caption_embeddings=False, use_scene_passt_embeddings=False):
    if use_sentence_embeddings:
        # here, query is the caption embedding, instead of the word-based query of the baseline
        if use_auto_caption_embeddings:
            audio, word_query, info, query, auto_caption = dataset[index]

            audio = torch.unsqueeze(audio, dim=0).to(device=device)
            auto_caption = torch.unsqueeze(auto_caption, dim=0).to(device=device)
            query = torch.unsqueeze(query, dim=0).to(device=device)
            audio_emb, query_emb = model([audio, auto_caption], query)

        elif use_scene_passt_embeddings:

            audio, word_query, info, query, scene_passt_embed = dataset[index]

            audio = torch.unsqueeze(audio, dim=0).to(device=device)
            query = torch.unsqueeze(query, dim=0).to(device=device)
            scene_passt_embed = torch.unsqueeze(scene_passt_embed, dim=0).to(device=device)

            audio_emb, query_emb, audio_emb2 = model([audio, scene_passt_embed], query)

        else:

            audio, word_query, info, query = dataset[index]

            audio = torch.unsqueeze(audio, dim=0).to(device=device)
            query = torch.unsqueeze(query, dim=0).to(device=device)

            audio_emb, query_emb, audio_emb2 = model(audio, query)

    else:
        audio, query, info = dataset[index]

        audio = torch.unsqueeze(audio, dim=0).to(device=device)
        query = torch.unsqueeze(query, dim=0).to(device=device)

        audio_emb, query_emb = model(audio, query, [query.size(-1)])

    audio_emb = torch.squeeze(audio_emb, dim=0).to(device=device)
    query_emb = torch.squeeze(query_emb, dim=0).to(device=device)

    return audio_emb, query_emb, info


def audio_retrieval(model, caption_dataset, K=10, use_sentence_embeddings=False,
                    use_auto_caption_embeddings=False, use_scene_passt_embeddings=False,
                    scores_output_fpath=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device=device)

    model.eval()

    print('use_auto_caption_embeddings', use_auto_caption_embeddings)

    if scores_output_fpath is not None:

        with h5py.File(scores_output_fpath, "w") as score_store:

            with torch.no_grad():

                fid_embs, fid_fnames = {}, {}
                cid_embs, cid_infos = {}, {}

                # Encode audio signals and captions
                for cap_ind in range(len(caption_dataset)):
                    audio_emb, query_emb, info = transform(model, caption_dataset, cap_ind, device, use_sentence_embeddings, use_auto_caption_embeddings, use_scene_passt_embeddings)

                    fid_embs[info["fid"]] = audio_emb
                    fid_fnames[info["fid"]] = info["fname"]

                    cid_embs[info["cid"]] = query_emb
                    cid_infos[info["cid"]] = info

                # Stack audio embeddings
                audio_embs, fnames = [], []
                for fid in fid_embs:
                    audio_embs.append(fid_embs[fid])
                    fnames.append(fid_fnames[fid])

                audio_embs = torch.vstack(audio_embs)  # dim [N, E]

                # Compute similarities
                output_rows = []
                for cid in cid_embs:

                    # print(cid)

                    sims = torch.mm(torch.vstack([cid_embs[cid]]), audio_embs.T).flatten().to(device=device)

                    # print(cid, sims.size(), sims[:5])

                    sorted_idx = torch.argsort(sims, dim=-1, descending=True)

                    csv_row = [cid_infos[cid]["caption"]]  # caption
                    for idx in sorted_idx[:K]:  # top-K retrieved fnames
                        csv_row.append(fnames[idx])

                    score_store[str(cid)] = sims.clone().detach().cpu().numpy()

                    output_rows.append(csv_row)

    else:

        with torch.no_grad():

            fid_embs, fid_fnames = {}, {}
            cid_embs, cid_infos = {}, {}

            # Encode audio signals and captions
            for cap_ind in range(len(caption_dataset)):
                audio_emb, query_emb, info = transform(model, caption_dataset, cap_ind, device, use_sentence_embeddings,
                                                       use_auto_caption_embeddings, use_scene_passt_embeddings)

                fid_embs[info["fid"]] = audio_emb
                fid_fnames[info["fid"]] = info["fname"]

                cid_embs[info["cid"]] = query_emb
                cid_infos[info["cid"]] = info

            # Stack audio embeddings
            audio_embs, fnames = [], []
            for fid in fid_embs:
                audio_embs.append(fid_embs[fid])
                fnames.append(fid_fnames[fid])

            audio_embs = torch.vstack(audio_embs)  # dim [N, E]

            # Compute similarities
            output_rows = []
            for cid in cid_embs:

                sims = torch.mm(torch.vstack([cid_embs[cid]]), audio_embs.T).flatten().to(device=device)

                sorted_idx = torch.argsort(sims, dim=-1, descending=True)

                csv_row = [cid_infos[cid]["caption"]]  # caption
                for idx in sorted_idx[:K]:  # top-K retrieved fnames
                    csv_row.append(fnames[idx])

                output_rows.append(csv_row)

    return output_rows


def eval_checkpoint(config, checkpoint_dir):
    # Load config
    training_config = config["training"]

    if 'caption_embeddings' in config['train_data']:
        use_sentence_embeddings = True
    else:
        use_sentence_embeddings = False
    # print("Using caption embeddings?", use_sentence_embeddings)

    use_auto_caption_embeddings = config['train_data']['use_auto_caption_embeddings']
    print("Using Etienne's auto caption embeddings?", use_auto_caption_embeddings)

    use_scene_passt_embeddings = config['train_data']['use_scene_passt_embeddings']
    print("Using scene PaSST embeddings?", use_scene_passt_embeddings)

    # Load evaluation
    caption_datasets, vocabulary = data_utils.load_data(config["eval_data"])

    # Initialize a model instance
    model_config = config[training_config["model"]]
    model = model_utils.get_model(model_config, vocabulary)
    # print(model)

    # Restore model states
    model = model_utils.restore(model, checkpoint_dir)
    model.eval()

    # Retrieve audio files for evaluation captions
    for split in ["test"]:
        output = audio_retrieval(model, caption_datasets[split], K=10,
                                 use_sentence_embeddings=use_sentence_embeddings,
                                 use_auto_caption_embeddings=use_auto_caption_embeddings,
                                 use_scene_passt_embeddings=use_scene_passt_embeddings)

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
        output.to_csv(os.path.join(checkpoint_dir, "{}.output.csv".format(split)),
                      index=False)
        print("Saved", "{}.output.csv".format(split))
