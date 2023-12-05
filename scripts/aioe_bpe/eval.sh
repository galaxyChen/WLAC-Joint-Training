# evaluate a saved checkpoint using fairseq.generate

# 1. create databin for test set
SRC=en
TGT=zh
INPUT_FILE=${PROJECT_ROOT}/sample_dataset/test.bpe.${SRC}-${TGT}.json
VALIDPREF=test

RAW_OUTPUT_DIR=${PROJECT_ROOT}/sample_dataset
python ./utils.py process_validset --input_file ${INPUT_FILE} --output_prefix ${RAW_OUTPUT_DIR}/${VALIDPREF} --src ${SRC} --tgt ${TGT}

# generate databin
RAW_DATA_DIR=${RAW_OUTPUT_DIR}
DEST_DATA_DIR=${PROJECT_ROOT}/sample_databin/${SRC}-${TGT}
mkdir -p ${DEST_DATA_DIR}
SRC_DICT=${PROJECT_ROOT}/sample_databin/${SRC}-${TGT}/dict.${SRC}.txt
TGT_DICT=${PROJECT_ROOT}/sample_databin/${SRC}-${TGT}/dict.${TGT}.txt

fairseq-preprocess --source-lang ${SRC} --target-lang ${TGT} \
--srcdict ${SRC_DICT} \
--tgtdict ${TGT_DICT} \
--testpref ${RAW_DATA_DIR}/${VALIDPREF} \
--destdir ${DEST_DATA_DIR} \
--workers 8

# 2. generate predictions
checkpoint_path=${PROJECT_ROOT}/save/aioe_bpe/en-zh/full/test_bpe_joint/checkpoint_best.pt
SUPPLY_PATH=${PROJECT_ROOT}/sample_dataset
BPE_PATH=${PROJECT_ROOT}/sample_dataset/bpe
# output_path: path to save predictions
output_path=${PROJECT_ROOT}/save/aioe_bpe/en-zh/full/test_bpe_joint

python ${PROJECT_ROOT}/generate.py \
        ${DEST_DATA_DIR} \
        --nbest 5 --beam 5 \
        --supply-path ${SUPPLY_PATH} \
        --bpe-path ${BPE_PATH} \
        --user-dir ${PROJECT_ROOT} \
        --task aioe_bpe \
        --path $checkpoint_path \
        --results-path ${output_path} \
        --batch-size 512 \
        --source-lang ${SRC} \
        --target-lang ${TGT} \
        --gen-subset test \
        --num-workers 10 \
        --suggestion-type full

# 3. compute accuracy
python ./utils.py extract_generation \
    --generate_path ${output_path}/generate-test.txt \
    --input_path ${INPUT_FILE} \
    --output_path ${output_path}/extract_test.txt

python ./utils.py compute_acc \
    --input_path ${output_path}/extract_test.txt

## Line by line generation
# python ./aioe_generation_pipeline.py generate \
#     --checkpoint_path ${checkpoint_path} \
#     --input_path ${input_path} \
#     --output_path ${output_path}

# python ./utils.py compute_acc \
#     --input_path ${output_path}