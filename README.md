## WLAC-Joint-Training
This repository contains the code for the paper Rethinking Word-Level Auto-Completion in Computer-Aided Translation, EMNLP 2023

### Requirements
Create the environment using conda:
```bash
conda create -n wlac python=3.6
conda activate wlac
pip install -r requirements.txt
```

### Running the code
#### 1. Prepare the environment variable
> export PROJECT_ROOT=path_to_code/WLAC-Joint-Training

Enter the run script directory. For AIOE-BPE model please refer to `scripts/aioe_bpe`
> cd scripts/aioe

*Please check all the related files and modify the path to the actual path*
#### 2. Preprocess training set
> bash ./preprocess_training_set.sh

#### 3. Preprocess validation set
> bash ./preprocess_valid_set.sh

#### 4. Run training
For AIOE model:
> bash ./train_aioe.sh
For AIOE-Joint model:
> bash ./train_aioe_joint.sh

#### 5. Evaluation
> bash ./eval.sh


## Citation

Please cite as:

``` bibtex
@article{chen2023rethinking,
  title={Rethinking Word-Level Auto-Completion in Computer-Aided Translation},
  author={Chen, Xingyu and Liu, Lemao and Huang, Guoping and Zhang, Zhirui and Yang, Mingming and Shi, Shuming and Wang, Rui},
  journal={arXiv preprint arXiv:2310.14523},
  year={2023}
}
```