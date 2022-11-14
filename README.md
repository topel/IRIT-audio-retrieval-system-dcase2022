# IRIT-audio-retrieval-system-dcase2022
Language-based audio retrieval system for DCASE 2022 Task 6b

## Part 1 - Feature extraction

1- Extract the audio scene embeddings (logits) with PaSST 

First, you need to install PaSST, more precisely the HEAR 2021 module, which is a lightweight version useful for feature extraction:

```pip install -e 'git+https://github.com/kkoutini/passt_hear21@0.0.17#egg=hear21passt'``` 

For more details, see https://github.com/kkoutini/passt_hear21

Then, edit the ```scripts/audio_features.py```. Search for "# Edit" in the script. You need to edit the PaSST module path and the ```dataset_dir``` path.

Run the script twice: 

i) once, with ```PASST = True  # for dev splits: train/valid/eval```

ii) once, with ```PASST_EVAL = True  # for the task 6b official 2022 eval subset```

This will output two HDF5 files, containing the logit vectors for all the Clotho v2 splits.

