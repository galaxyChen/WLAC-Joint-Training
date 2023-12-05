# create databin for existing valid set
# use existed dict generated from training set


# split test file with json format into source and target files
SRC=en
TGT=zh
VALIDPREF=valid
# the input file is processed by bpe
INPUT_FILE=${PROJECT_ROOT}/sample_dataset/test.bpe.${SRC}-${TGT}.json
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
--validpref ${RAW_DATA_DIR}/${VALIDPREF} \
--destdir ${DEST_DATA_DIR} \
--workers 8