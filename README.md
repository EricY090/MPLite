# MPLite (Multi-Aspect Pretraining for Mining Clinical Health Records)

# Packages

- python == 3.9.0
- sklearn == 1.4.0
- numpy == 1.26.2
- tensorflow == 2.10
- torch == 2.1.2
- pandas == 1.5.3

# Dataset Preparation
Download MIMIC-III from Physionet (Check: https://physionet.org/content/mimiciii/1.4/) and put the ADMISSION table, LABEVENT table, and DIAGNOSES_ICD table under /data/mimic3/raw/

1. Preprocess MIMIC-III
```bash
python run_preprocessing.py
```
2. Prepared pretrained lab
```bash
python train_lab.py
```

# To run each baselines for comparison:

a. CGL (Collaborative Graph Learning with Auxiliary Text for Temporal Event Prediction in Healthcare)
```bash
python train_CGL.py
```

b. RETAIN (Interpretable Predictive Model in Healthcare using Reverse Time Attention Mechanism)
```bash
python train_RETAIN.py
```

c. Timeline (Interpretable Representation Learning for Healthcare via Capturing Disease Progression through Time)
```bash
python Timeline_preprocess.py
```
```bash
python Timeline.py
```

d. GRU 
```bash
python train_GRU.py
```

