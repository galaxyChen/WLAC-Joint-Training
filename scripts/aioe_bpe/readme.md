## Run Pipeline
Please check all the related files and modify the path to the actual path
### 1. Preprocess training set
> bash ./preprocess_training_set.sh

### 2. Preprocess validation set
> bash ./preprocess_valid_set.sh

### 3. Run training
For AIOE model:
> bash ./train_aioe.sh
For AIOE-Joint model:
> bash ./train_aioe_joint.sh

### 4. Evaluation
> bash ./eval.sh