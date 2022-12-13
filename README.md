# IRIT-audio-retrieval-system-dcase2022
Language-based audio retrieval system for DCASE 2022 Task 6b

## Part 1 - Data preprocessing

This part is similar to the one documented in the challenge baseline system repo, please give it a visit for more info on these steps:

https://github.com/xieh97/dcase2022-audio-retrieval 

1- Download and extract audio files and captions of the Clotho v2.1 dataset from Zenodo: https://zenodo.org/record/4783391. 

2- Preprocess audio and caption data.

```python scripts/preprocess.py``` 

This step will output four files:

- ```audio_info.pkl```: audio file names and durations.
- ```development_captions.json```: preprocessed captions of Clotho development split.
- ```validation_captions.json```: preprocessed captions of Clotho validation split.
- ```evaluation_captions.json```: preprocessed captions of Clotho evaluation split.
- ```vocab_info.pkl```: vocabulary statistics of Clotho dataset.

I provide another script to do the same on the official evaluation subset:

```python scripts/preprocess_evaluation_set_task6b.py```

Why a separate script? Because unlike the dev/val/eval subsets, the official eval subset provides a single caption instead of five, and of course no ground-truth audio file names with these captions.

## Part 2 - Audio feature extraction: Global PaSST logits (one vector per audio file)

1- Extract the audio scene embeddings (logits) with PaSST 

First, you need to install PaSST, more precisely the HEAR 2021 module, which is a lightweight version useful for feature extraction:

```pip install -e 'git+https://github.com/kkoutini/passt_hear21@0.0.17#egg=hear21passt'``` 

For more details, see https://github.com/kkoutini/passt_hear21

Then, edit the ```scripts/audio_features.py```. Search for "# Edit" in the script. You need to edit the PaSST module path and the ```dataset_dir``` path.

Run the script twice: 

i) once, with ```PASST = True  # for dev splits: train/valid/eval```

ii) once, with ```PASST_EVAL = True  # for the task 6b official 2022 eval subset```

This will output two HDF5 files, containing the logit vectors for all the Clotho v2 splits.

## Part 3 - sentence embedding extraction: with all-mpnet-base-v2

In this part, we extract one sentence embedding per caption. I used MPNet. You need to install the ```transformers``` library from HuggingFace.

1- For the dev/val/eval subsets

```python scripts/sentence_embeddings.py```

This step will output ```sentence_embeddings_mpnet.hdf5```.

2- The same for the evaluation_set_task6b subset

```python scripts/sentence_embeddings_evaluation_set_task6b.py```

## Part 4 - train and test a model

The serious business (problems? ;-)) begins here... 

1- First edit the YAML config file: ```conf.yaml```

2- Run the main script: ```python main.py```

Remarks

- It uses wandb to log the exps, you can activate it in the YAML file with: ```wandb: True```. As provided, it is set to ```False```.

- This script main function is ```train```, but there is also a method to train a model on AudioCaps: ```train_audiocaps```.

## Citation

T. Pellegrini. Language-based audio retrieval with textual embeddings of tag names. In Proc. Workshop DCASE, Nancy, Nov. 2022.
https://ut3-toulouseinp.hal.science/hal-03812737/document

