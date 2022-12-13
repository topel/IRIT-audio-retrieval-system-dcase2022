import pandas as pd

import wandb

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

torch_seed = 0
numpy.random.seed(0)
torch.manual_seed(torch_seed)

YAML_FPATH = "conf.yaml"
print(YAML_FPATH)
# YAML_FPATH = "choro_conf.yaml"
# YAML_FPATH = "my_conf.yaml"

def train_audiocaps(config, checkpoint_dir=None):

    training_config = config["training"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_wandb = conf["wandb"]

    if log_wandb:
        wandb.config = {
            "learning_rate": config["AdamOptimizer"]["args"]["lr"],
            "epochs": training_config["algorithm"]["epochs"],
            "batch_size": training_config["algorithm"]["batch_size"],
            "mixup": config["train_data"]["use_mixup"]
        }

    use_sentence_embeddings = True
    use_auto_caption_embeddings = False
    print("Using Etienne's auto caption embeddings?", use_auto_caption_embeddings)

    # Load audio features
    feats_path = os.path.join(config["audiocaps_train_data"]["input_path"], config["audiocaps_train_data"]["audio_features"])
    audio_feats = h5py.File(feats_path, "r")
    print("Load", feats_path)

    sent_emb_path = os.path.join(config["audiocaps_train_data"]["input_path"], config["audiocaps_train_data"]["caption_embeddings"])
    sentence_embeddings = h5py.File(sent_emb_path, "r")
    print("Load", sent_emb_path)

    csv_path = os.path.join(config["audiocaps_train_data"]["input_path"], config["audiocaps_train_data"]["data_splits"]['train'])
    df = pd.read_csv(csv_path)
    print("Load", csv_path)

    from utils.data_utils import AudiocapsQueryAudioDataset

    audiocaps_dataset = AudiocapsQueryAudioDataset(audio_feats, df, sentence_embeddings)

    audiocaps_loader = DataLoader(dataset=audiocaps_dataset, batch_size=training_config["algorithm"]["batch_size"],
                                 shuffle=False, collate_fn=data_utils.audiocaps_collate_fn_sentence_embeddings)

    # ---------------

    text_datasets, vocabulary = data_utils.load_data(config["train_data"])

    clotho_split = "test"
    clotho_dataset = text_datasets[clotho_split]
    clotho_loader = DataLoader(dataset=clotho_dataset, batch_size=training_config["algorithm"]["batch_size"],
                                 shuffle=False, collate_fn=data_utils.collate_fn_sentence_embeddings)

    model_config = config[training_config["model"]]
    model = model_utils.get_model(model_config, vocabulary)
    print(model)
    # print(model.audio_encoder.pointwise_conv.weight.data)

    print("Nb of learnable parameters :",
          sum(p.numel() for p in model.parameters() if p.requires_grad))

    alg_config = training_config["algorithm"]

    criterion_config = config[alg_config["criterion"]]
    if config["train_data"]["use_mixup"] and criterion_config["name"] != "MixupTripletRankingLoss":
        print("ERROR: change criterion to MixupTripletRankingLoss in config file")
        return -1

    criterion = getattr(criterion_utils, criterion_config["name"], None)(**criterion_config["args"])

    optimizer_config = config[alg_config["optimizer"]]
    optimizer = getattr(optim, optimizer_config["name"], None)(
        model.parameters(), **optimizer_config["args"]
    )

    # lr_scheduler = getattr(optim.lr_scheduler, "ReduceLROnPlateau")(optimizer, **optimizer_config["scheduler_args"])

    if checkpoint_dir is not None:
        model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    print("nb epochs:", alg_config["epochs"])

    for epoch in range(alg_config["epochs"] + 1):
        # print('epoch %d'%epoch)
        # if epoch > 0:
        # if epoch < 1 :
        #     mixup_lambdas = []
        # else: mixup_lambdas = None

        # current_lr_value = optimizer.state_dict()['param_groups'][0]['lr']
        # if log_wandb:
        #     wandb.log({"lr": current_lr_value})
        # print('lr = {0}'.format(current_lr_value))

        epoch_results = {}

        epoch_results["train_loss"] = model_utils.train(model, optimizer, criterion, audiocaps_loader, epoch, use_sentence_embeddings,
                          use_auto_caption_embeddings, log_wandb)
        # print(model.audio_encoder.pointwise_conv.weight.data)


        epoch_results["test_loss"] = model_utils.eval(model, criterion, clotho_loader,
                                                                   use_sentence_embeddings,
                                                                   use_auto_caption_embeddings)
        print('epoch %d --- train %.3f   test %.3f' % (
        epoch, epoch_results["train_loss"], epoch_results["test_loss"]))

        # Reduce learning rate based on validation loss
        # lr_scheduler.step(epoch_results[config["ray_conf"]["stopper_args"]["metric"]])

        if log_wandb:
            wandb.log({"train_loss": epoch_results["train_loss"]})
            # wandb.log({"val_loss": epoch_results["val_loss"]})
            wandb.log({"test_loss": epoch_results["test_loss"]})

        checkpoint_dir_ = config['output_path'] + '/audiocaps_checkpoints_seed_%d/checkpoint_%06d' % (torch_seed, epoch)
        if not os.path.exists(checkpoint_dir_):
            os.makedirs(checkpoint_dir_)

        path = os.path.join(checkpoint_dir_, "checkpoint")
        torch.save((model.state_dict(), optimizer.state_dict()), path)
        # Evaluate at the best checkpoint
        eval_utils.eval_checkpoint(conf, checkpoint_dir_)
        gt_csv = "test.gt.csv"  # ground truth for Clotho evaluation data
        pred_csv = os.path.join(checkpoint_dir_,
                                "test.output.csv")  # baseline system retrieved output for Clotho evaluation data

        retrieval_metrics(gt_csv, pred_csv, log_wandb, is_ema=False)


def train(config, checkpoint_dir=None):

    training_config = config["training"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_wandb = config["wandb"]

    # if log_wandb:
    #     wandb.config = {
    #         "learning_rate": config["AdamOptimizer"]["args"]["lr"],
    #         "epochs": training_config["algorithm"]["epochs"],
    #         "batch_size": training_config["algorithm"]["batch_size"],
    #         "mixup": config["train_data"]["use_mixup"]
    #     }

    if 'caption_embeddings' in config['train_data']:
        use_sentence_embeddings = True
    else:
        use_sentence_embeddings = False
    print("Using caption embeddings?", use_sentence_embeddings)

    use_scene_passt_embeddings = config['train_data']['use_scene_passt_embeddings']
    print("Using scene PaSST embeddings?", use_scene_passt_embeddings)

    use_auto_caption_embeddings = config['train_data']['use_auto_caption_embeddings']
    print("Using Etienne's auto caption embeddings?", use_auto_caption_embeddings)

    text_datasets, vocabulary = data_utils.load_data(config["train_data"])

    text_loaders = {}
    for split in ["train", "val", "test"]:
        _dataset = text_datasets[split]
        if use_sentence_embeddings:
            if use_auto_caption_embeddings :
                _loader = DataLoader(dataset=_dataset, batch_size=training_config["algorithm"]["batch_size"],
                                     shuffle=True, collate_fn=data_utils.collate_fn_sentence_and_auto_caption_embeddings)
            elif use_scene_passt_embeddings :
                _loader = DataLoader(dataset=_dataset, batch_size=training_config["algorithm"]["batch_size"],
                                     shuffle=True, collate_fn=data_utils.collate_fn_sentence_and_scene_embeddings)
            else:
                _loader = DataLoader(dataset=_dataset, batch_size=training_config["algorithm"]["batch_size"],
                                     shuffle=True, collate_fn=data_utils.collate_fn_sentence_embeddings)
        else:
            _loader = DataLoader(dataset=_dataset, batch_size=training_config["algorithm"]["batch_size"],
                                 shuffle=True, collate_fn=data_utils.collate_fn)

        text_loaders[split] = _loader

    model_config = config[training_config["model"]]
    model = model_utils.get_model(model_config, vocabulary)
    print(model)
    # print(model.audio_encoder.pointwise_conv.weight.data)

    print("Nb of learnable parameters :",
          sum(p.numel() for p in model.parameters() if p.requires_grad))


    alg_config = training_config["algorithm"]

    criterion_config = config[alg_config["criterion"]]
    if config["train_data"]["use_mixup"] and criterion_config["name"] != "MixupTripletRankingLoss":
        print("ERROR: change criterion to MixupTripletRankingLoss in config file")
        return -1

    # sweep
    # criterion_config["args"]["margin"] = wandb.config["margin"]
    # print("train margin:", wandb.config["margin"])

    criterion = getattr(criterion_utils, criterion_config["name"], None)(**criterion_config["args"])

    optimizer_config = config[alg_config["optimizer"]]
    optimizer = getattr(optim, optimizer_config["name"], None)(
        model.parameters(), **optimizer_config["args"]
    )

    lr_scheduler = getattr(optim.lr_scheduler, "ReduceLROnPlateau")(optimizer, **optimizer_config["scheduler_args"])

    if checkpoint_dir is not None:
        model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print("Checkpoint loaded from", checkpoint_dir)

    ratio_embed1 = config['PasstSentModel']['args']["audio_encoder"]['ratio_embed1']

    # sweep
    # ratio_embed1 = wandb.config["ratio_embed1"]
    print("train ratio_embed1:", ratio_embed1)
    audioset_info = config['PasstSentModel']['args']["audio_encoder"]['audioset_info']

    print("nb epochs:", alg_config["epochs"])
    for epoch in range(alg_config["epochs"] + 1):
        # print('epoch %d'%epoch)
        # if epoch > 0:
        # if epoch < 1 :
        #     mixup_lambdas = []
        # else: mixup_lambdas = None

        current_lr_value = optimizer.state_dict()['param_groups'][0]['lr']
        if log_wandb:
            wandb.log({"lr": current_lr_value})
        print('lr = {0}'.format(current_lr_value))

        model_utils.train(model, optimizer, criterion, text_loaders["train"], epoch, use_sentence_embeddings, use_auto_caption_embeddings, use_scene_passt_embeddings, log_wandb)
        # print(model.audio_encoder.pointwise_conv.weight.data)

        epoch_results = {}
        for split in ["train", "val", "test"]:
            epoch_results["{0}_loss".format(split)] = model_utils.eval(model, criterion, text_loaders[split], use_sentence_embeddings, use_auto_caption_embeddings, use_scene_passt_embeddings)
        print('epoch %d --- train %.3f   val %.3f   test %.3f'%(epoch, epoch_results["train_loss"], epoch_results["val_loss"], epoch_results["test_loss"]))


        # Reduce learning rate based on validation loss
        lr_scheduler.step(epoch_results[config["ray_conf"]["stopper_args"]["metric"]])

        if log_wandb:
            wandb.log({"train_loss": epoch_results["train_loss"]})
            wandb.log({"val_loss": epoch_results["val_loss"]})
            wandb.log({"test_loss": epoch_results["test_loss"]})
            # if step[0] > 5000:
            #     wandb.log({"train_ema_loss": epoch_results["train_ema_loss"]})
            #     wandb.log({"val_ema_loss": epoch_results["val_ema_loss"]})
            #     wandb.log({"test_ema_loss": epoch_results["test_ema_loss"]})

            # if epoch < 1:
            #     mixup_lambdas = numpy.array(mixup_lambdas)
            #     print(mixup_lambdas.shape)
            #     print(mixup_lambdas)
            #     # table = wandb.Table(data=mixup_lambdas, columns=["mixup_lambdas"])
            #     wandb.log(
            #         {'mixup_lambdas': wandb.Histogram(mixup_lambdas)})

        # # Save the model to the trial directory: local_dir/exp_name/trial_name/checkpoint_<step>
        # with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
        #     path = os.path.join(checkpoint_dir, "checkpoint")
        #     torch.save((model.state_dict(), optimizer.state_dict()), path)
        # if epoch % 10 == 0:

        checkpoint_dir_ = config['output_path'] + '/checkpoint_seed_%d_%s_%.3f/checkpoint_%06d'%(torch_seed, audioset_info, ratio_embed1, epoch)
        if not os.path.exists(checkpoint_dir_):
            os.makedirs(checkpoint_dir_)

        path = os.path.join(checkpoint_dir_, "checkpoint")
        torch.save((model.state_dict(), optimizer.state_dict()), path)
        # Evaluate at the best checkpoint
        eval_utils.eval_checkpoint(config, checkpoint_dir_)
        gt_csv = "test.gt.csv"  # ground truth for Clotho evaluation data
        pred_csv = os.path.join(checkpoint_dir_,"test.output.csv")  # baseline system retrieved output for Clotho evaluation data

        retrieval_metrics(gt_csv, pred_csv, log_wandb, is_ema=False)

        # if epoch == 20:
        #     # exp. moving average of weights
        #     ema_model = ema.ExponentialMovingAverage(model.named_parameters(), decay=0.995, device=device, use_num_updates=False)
        #
        # if epoch > 20:
        #     ema_model.update(model.named_parameters())
        #     # First save original parameters before replacing with EMA version
        #     ema_model.store(model.named_parameters())
        #     # Copy EMA parameters to model
        #     ema_model.copy_to(model.named_parameters())
        #     for split in ["train", "val", "test"]:
        #         epoch_results["{0}_ema_loss".format(split)] = model_utils.eval(model, criterion, text_loaders[split],
        #                                                                        use_sentence_embeddings,
        #                                                                        use_auto_caption_embeddings)
        #     print('epoch %d ema --- train %.3f   val %.3f   test %.3f' % (
        #         epoch, epoch_results["train_ema_loss"], epoch_results["val_ema_loss"], epoch_results["test_ema_loss"]))
        #
        #     checkpoint_dir_ = config['output_path'] + '/checkpoints/ema_checkpoint_%06d' % epoch
        #     if not os.path.exists(checkpoint_dir_):
        #         os.makedirs(checkpoint_dir_)
        #
        #     path = os.path.join(checkpoint_dir_, "checkpoint")
        #     torch.save((model.state_dict(), optimizer.state_dict()), path)
        #     # Evaluate at the best checkpoint
        #     eval_utils.eval_checkpoint(conf, checkpoint_dir_)
        #     gt_csv = "test.gt.csv"  # ground truth for Clotho evaluation data
        #     pred_csv = os.path.join(checkpoint_dir_,
        #                             "test.output.csv")  # baseline system retrieved output for Clotho evaluation data
        #
        #     retrieval_metrics(gt_csv, pred_csv, log_wandb, is_ema=True)
        #
        #     # Restore original parameters to resume training later
        #     ema_model.restore(model.named_parameters())

        # Send the current statistics back to the Ray cluster
        # tune.report(**epoch_results)


# Main
if __name__ == "__main__":
    # Load configurations

    with open(YAML_FPATH, "rb") as stream:
        conf = yaml.full_load(stream)

    log_wandb = conf["wandb"]

    # with open("sweep.yaml", "rb") as stream:
    #     sweep_conf = yaml.full_load(stream)
    #
    # sweep_id = wandb.sweep(sweep_conf)

    if conf["train_data"]["use_mixup"]:
        use_mixup = True
    else: use_mixup = False

    print('ratio_embed1', conf['PasstSentModel']['args']["audio_encoder"]['ratio_embed1'])
    audioset_info = conf['PasstSentModel']['args']["audio_encoder"]['audioset_info']
    print(audioset_info)

    if log_wandb:
        ratio_embed1 = conf['PasstSentModel']['args']["audio_encoder"]['ratio_embed1']
        margin = conf['TripletRankingLoss']['args']['margin']
        wandb.login(key='cfa500f72c83d0e45e69c8e1ad4a0d1e735bb5b2')
        wandb.init(project="lbar22", entity="topel", name="test_ratio1_passt_mpnet_seed%d_lr%.3f_epochs%d_bs%d_mixup%d_%s_%.3f_margin_%.3f"%(torch_seed,
            conf["AdamOptimizer"]["args"]["lr"],
            conf["training"]["algorithm"]["epochs"],
            conf["training"]["algorithm"]["batch_size"],
            1*use_mixup, audioset_info, ratio_embed1, margin
        ))
        # set up a default config , which might be over-ridden by the sweep.


        # w/ mixup
        # wandb.init(project="lbar22", entity="topel", name="passt_mpnet_lr%.3f_epochs%d_bs%d_mixup%d_beta%.1f_inAffineNo"%(
        #     conf["AdamOptimizer"]["args"]["lr"],
        #     conf["training"]["algorithm"]["epochs"],
        #     conf["training"]["algorithm"]["batch_size"],
        #     1*use_mixup,
        #     conf["train_data"]["beta"]
        #     ))

    # def trial_dirname_creator(trial):
    #     trial_dirname = "{0}_{1}_{2}".format(
    #         conf["training"]["model"], trial.trial_id, time.strftime("%Y-%m-%d_%H-%M-%S")
    #     )
    #     return trial_dirname


    # Run a Ray cluster - local_dir/exp_name/trial_name
    # train(conf, checkpoint_dir='outputs_passt/checkpoints/checkpoint_000030')
    # train_audiocaps(conf)

    # count = 1
    # # number of runs to execute
    # wandb.agent(sweep_id, function=train, count=count)

    train(conf)
    # train_audiocaps(conf)
    # train(conf, checkpoint_dir='outputs_passt/audiocaps_checkpoints/checkpoint_000077')
    # # train(conf, checkpoint_dir='outputs_passt/audiocaps_checkpoints/checkpoint_000046')
    # # train(conf, checkpoint_dir='outputs_passt/audiocaps_checkpoints/checkpoint_000098')

    # # Check the best trial and its best checkpoint
    # best_trial = analysis.get_best_trial(
    #     metric=ray_conf["stopper_args"]["metric"],
    #     mode=ray_conf["stopper_args"]["mode"],
    #     scope="all"
    # )
    # best_checkpoint = analysis.get_best_checkpoint(
    #     trial=best_trial,
    #     metric=ray_conf["stopper_args"]["metric"],
    #     mode=ray_conf["stopper_args"]["mode"]
    # )
    # print("Best trial:", best_trial.trial_id)
    # print("Best checkpoint:", best_checkpoint)
    #
    # # Evaluate at the best checkpoint
    # eval_utils.eval_checkpoint(conf, best_checkpoint)
