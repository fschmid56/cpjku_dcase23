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

The configuration can be adapted using the command line, e.g. changing the probability of the device impulse response augmentation:

```
python run_training.py --dir_prob=0.4
```

The results are automatically logged using Weights & Biases.

The models can be quantized using Quantization Aware Training (QAT). For this, the trained model from the previous step is loaded by
specifying the Wandb ID and fine-tuned using QAT for 24 epochs. The following command can be used:

```
python run_qat.py --wandb_id=c0a7nzin
```

## Checkpoints

Running the training procedure creates a folder [DCASE23_Task1](DCASE23_Task1). This folder contains subfolder named according
to the ID assigned to the experiment by Weights and Biases. These subfolders contain checkpoints which can be used to load
the trained models (see [run_qat.py](run_qat.py) for an example).

## Example experiments

Default parameters for training on TAU22: 

```
python run_training.py
```

Fine-tuning and quantizing model using Quantization Aware Training (trained model with wandb_id=c0a7nzin already included
in GitHub repo):

```
python run_qat.py --wandb_id=c0a7nzin
```

Checkout the [results](https://wandb.ai/florians/DCASE23_Task1/reports/Test-run-of-CPJKU-Submission-to-DCASE23-Task-1--Vmlldzo0NzEwNjIy?accessToken=vcgldrnpus2r27wr2hir9g0t6l84mat2n9760ab3xf2nbzu9p5850h2g4t8pas63) on Weights & Biases.


## Device Impulse Reponses

The device impulse responses in [datasets/dirs](datasets/dirs) are downloaded from [MicIRP](http://micirp.blogspot.com/). All files
are shared via Creative Commons license. All credits go to MicIRP & Xaudia.com.


## Teacher Ensemble

We provide the ensembled logits of 3 CP-ResNet [2] models and 3 PaSST [1] transformer models trained on the TAU22 development set train split.
The teacher models are trained using the cropped dataset technique introduced in the technical report. The logits
are automatically downloaded when running the code and end up in the [resources](resources) folder.

## Pre-Trained Teacher Models

Based on a request, we also make the pre-trained teacher models available. 
In total 12 pre-trained models are published:
* 2 x PaSST trained with MixStyle and DIR: ```passt_ms_dir_1.pt``` and ```passt_ms_dir_2.pt```
* 2 x PaSST trained with DIR: ```passt_dir_1.pt``` and ```passt_dir_2.pt```
* 2 x PaSST trained with MixStyle: ```passt_ms_1.pt``` and ```passt_ms_2.pt```
* 2 x CP-ResNet trained with MixStyle and DIR: ```cpr_ms_dir_1.pt``` and ```cpr_ms_dir_2.pt```
* 2 x CP-ResNet trained with DIR: ```cpr_dir_1.pt``` and ```cpr_dir_2.pt```
* 2 x CP-ResNet trained with MixStyle: ```cpr_ms_1.pt``` and ```cpr_ms_2.pt```

The file ```run_teacher_validation.py``` is an example of how to use the teacher models for inference.

## References

[1] Khaled Koutini, Jan Schlüter, Hamid Eghbal-zadeh, and Gerhard Widmer, “Efficient Training of Audio Transformers with Patchout,” in Interspeech, 2022.

[2] Khaled Koutini, H. Eghbal-zadeh, and G. Widmer, “Receptive field
regularization techniques for audio classification and tagging
with deep convolutional neural networks,” IEEE ACM Trans.
Audio Speech Lang. Process., 2021.