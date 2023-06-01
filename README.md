# cpjku_dcase23

This repository contains the code for the CP-JKU submission to [DCASE23 Task 1 "Low-complexity Acoustic Scene Classification"](https://dcase.community/challenge2023/task-low-complexity-acoustic-scene-classification).

The technical report describing the system can be found [here](https://dcase.community/documents/challenge2023/technical_reports/DCASE2023_Schmid_28_t1.pdf).
The official ranking of system's submitted to the challenge is available [here](https://dcase.community/challenge2023/task-low-complexity-acoustic-scene-classification-results).

So far, the repository only contains the student network architecture *CP-Mobile*. It will soon be updated with the full
implementation of the ASC system.

## Setup 

Create a conda environment:

```
conda env create -f environment.yml
```

Example of creating an instance of CP-Mobile:

```
python test_cp_mobile.py
```