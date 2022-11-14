import glob
import os
import pickle

import h5py
import librosa
import numpy as np

import torch
import torchaudio
import torchaudio.functional as torchaudioF

import sys

# Edit this path
sys.path.append("/homelocal/thomas/research/dcase2022/hear21passt")
import hear21passt.base as hear21passt


def log_mel_spectrogram(y,
                        sample_rate=44100,
                        window_length_secs=0.025,
                        hop_length_secs=0.010,
                        num_mels=128,
                        fmin=12.0,
                        fmax=8000,
                        log_offset=0.0):
    """Convert waveform to a log magnitude mel-frequency spectrogram.

    :param y: 1D np.array of waveform data.
    :param sample_rate: The sampling rate of data.
    :param window_length_secs: Duration of each window to analyze.
    :param hop_length_secs: Advance between successive analysis windows.
    :param num_mels: Number of Mel bands.
    :param fmin: Lower bound on the frequencies to be included in the mel spectrum.
    :param fmax: The desired top edge of the highest frequency band.
    :param log_offset: Add this to values when taking log to avoid -Infs.
    :return:
    """
    window_length = int(round(sample_rate * window_length_secs))
    hop_length = int(round(sample_rate * hop_length_secs))
    fft_length = 2 ** int(np.ceil(np.log(window_length) / np.log(2.0)))

    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sample_rate, n_fft=fft_length, hop_length=hop_length,
                                                     win_length=window_length, n_mels=num_mels, fmin=fmin, fmax=fmax)

    return np.log(mel_spectrogram + log_offset)


def extract_log_mel(params):
    output_file = os.path.join(params["dataset_dir"], params["audio_feats"])

    with h5py.File(output_file, "w") as feature_store:
        for split in params["audio_splits"]:

            subset_dir = os.path.join(params["dataset_dir"], split)
            print(subset_dir)

            for fpath in glob.glob("{}/*.wav".format(subset_dir)):
                try:
                    fname = os.path.basename(fpath)
                    fid = global_params["audio_fids"][split][fname]

                    y, sr = librosa.load(fpath, sr=None, mono=True)
                    log_mel = log_mel_spectrogram(y=y, sample_rate=sr, window_length_secs=0.040,
                                                  hop_length_secs=0.020, num_mels=64,
                                                  log_offset=np.spacing(1))

                    feat = np.vstack(log_mel).transpose()  # [Time, Mel]

                    feature_store[str(fid)] = feat
                    print(fid)
                except:
                    print("Error file: {}.".format(fpath))


def raw_resampled_waveform(y,
                           sample_rate=44100,
                           resample_rate=32000):
    # kaiser_best
    # code from https://pytorch.org/tutorials/beginner/audio_resampling_tutorial.html

    resampled_y = torchaudioF.resample(
        y,
        sample_rate,
        resample_rate,
        lowpass_filter_width=64,
        rolloff=0.9475937167399596,
        resampling_method="kaiser_window",
        beta=14.769656459379492,
    )
    return resampled_y


def extract_raw(params):
    output_file = os.path.join(params["dataset_dir"], params["audio_feats"])

    with h5py.File(output_file, "w") as feature_store:
        for split in params["audio_splits"]:

            subset_dir = os.path.join(params["dataset_dir"], split)
            print(subset_dir)

            for fpath in glob.glob("{}/*.wav".format(subset_dir)):
                try:
                    fname = os.path.basename(fpath)
                    fid = global_params["audio_fids"][split][fname]

                    # y, sr = librosa.load(fpath, sr=32000, mono=True)
                    y, sample_rate = torchaudio.load(fpath)
                    y = raw_resampled_waveform(y,
                                               sample_rate=sample_rate,
                                               resample_rate=32000)

                    feature_store[str(fid)] = y
                    print(fid)
                except:
                    print("Error file: {}.".format(fpath))


def extract_passt(params):
    output_file = os.path.join(params["dataset_dir"], params["audio_feats"])

    passt_encoder = hear21passt.load_model(mode="logits")

    print("Nb of learnable parameters :",
          sum(p.numel() for p in passt_encoder.parameters() if p.requires_grad))

    with h5py.File(output_file, "w") as feature_store:
        for split in params["audio_splits"]:

            subset_dir = os.path.join(params["dataset_dir"], split)
            print(subset_dir)

            for fpath in glob.glob("{}/*.wav".format(subset_dir)):
                try:
                    fname = os.path.basename(fpath)
                    fid = global_params["audio_fids"][split][fname]

                    # y, sr = librosa.load(fpath, sr=32000, mono=True)
                    y, sample_rate = torchaudio.load(fpath)
                    y = raw_resampled_waveform(y,
                                               sample_rate=sample_rate,
                                               resample_rate=32000)

                    embed = hear21passt.get_scene_embeddings(y, passt_encoder)

                    feature_store[str(fid)] = torch.squeeze(embed)

                    print(fid)
                except:
                    print("Error file: {}.".format(fpath))


#
# %% Pre-computed stuff
#

RAW_WAV = False
LOGMEL = False
PASST = False  # for dev splits: train/valid/eval
PASST_EVAL = True  # for the task 6b official 2022 eval subset

# Edit the dataset_dir variable
if LOGMEL:
    global_params = {
        "dataset_dir": "/baie/corpus/DCASE2022/Clotho.v2.1",
        "audio_splits": ["development", "validation", "evaluation"],
        "audio_feats": "audio_logmel.hdf5"
    }
elif RAW_WAV:
    global_params = {
        "dataset_dir": "/baie/corpus/DCASE2022/Clotho.v2.1",
        "audio_splits": ["development", "validation", "evaluation"],
        "audio_feats": "audio_raw_waveforms_torch.hdf5"
    }
elif PASST:
    global_params = {
        "dataset_dir": "/baie/corpus/DCASE2022/Clotho.v2.1",
        "audio_splits": ["development", "validation", "evaluation"],
        "audio_feats": "audio_passt_torch.hdf5"
    }
elif PASST_EVAL:
    global_params = {
        "dataset_dir": "/baie/corpus/DCASE2022/evaluation_set_task6b",
        "audio_splits": ["test"],
        "audio_feats": "eval_task6b_audio_passt_torch.hdf5"
    }
else:
    global_params = None
    print("audio_features.py: feature type not defined, use either log-Mel or raw wave signals")

with open(os.path.join(global_params["dataset_dir"], "audio_info.pkl"), "rb") as store:
    global_params["audio_fids"] = pickle.load(store)["audio_fids"]

# Extract log mel features
if LOGMEL:
    extract_log_mel(global_params)
elif PASST or PASST_EVAL:
    extract_passt(global_params)
else:
    extract_raw(global_params)
