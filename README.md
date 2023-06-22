# cpjku_dcase23

This repository contains a simple version of the codebase to reproduce the results of the CP-JKU submission 
to the [DCASE23 Task 1 "Low-complexity Acoustic Scene Classification"](https://dcase.community/challenge2023/task-low-complexity-acoustic-scene-classification) challenge.
The implemented model **CP-Mobile** and training procedure scored the top rank in the challenge.

The technical report describing the system can be found [here](https://dcase.community/documents/challenge2023/technical_reports/DCASE2023_Schmid_28_t1.pdf). 
The official ranking of systems submitted to the challenge is available [here](https://dcase.community/challenge2023/task-low-complexity-acoustic-scene-classification-results).

An extension to the technical report (containing an ablation study and further results) is submitted to the [DCASE Workshop](https://dcase.community/workshop2023/) and a link to the paper will be provided soon.

## Setup

Create a conda environment:

```
conda env create -f environment.yml
```

Activate environment:

```
conda activate cpjku_dcase23
```

Download the dataset from [this](https://zenodo.org/record/6337421) location and extract the files.

Adapt path to dataset in the file [datasets/dcase22.py](datasets/dcase22.py) and provide the location of the extracted
"TAU-urban-acoustic-scenes-2022-mobile-development" folder. Put the path in the following variable:

```
dataset_dir = None
```

Run training on the [TAU22 dataset](https://zenodo.org/record/6337421):

```
python run_training.py
```

The configuration can be adapted using the command line, e.g. chaning the probability of the device impulse response augmentation:

```
python run_training.py --dir_prob=0.4
```

The results are automatically logged using Weights & Biases.

## Example experiments

Default command: 

```
python run_training.py
```

Checkout the results on weights and biases.

## Device Impulse Reponses

The device impulse responses in [datasets/dirs](datasets/dirs) are from [MicIRP](http://micirp.blogspot.com/). All files
are shared via Creative Commons license. All credits go to MicIRP & Xaudia.com.


## Teacher Ensemble

We provide the ensembled logits of 3 CP-ResNet [2] models and 3 PaSST [1] transformer models trained on the TAU22 development set train split.
The teacher models are trained using the cropped dataset technique introduced in the technical report. The logits
are automatically downloaded when running the code.

## References

[1] Khaled Koutini, Jan Schlüter, Hamid Eghbal-zadeh, and Gerhard Widmer, “Efficient Training of Audio Transformers with Patchout,” in Interspeech, 2022.

[2] Khaled Koutini, H. Eghbal-zadeh, and G. Widmer, “Receptive field
regularization techniques for audio classification and tagging
with deep convolutional neural networks,” IEEE ACM Trans.
Audio Speech Lang. Process., 2021.