# The training data is dynamically generated during training so the process here is the same as normal translation task.

SRC=en
TGT=zh
RAW_DATA_DIR=${PROJECT_ROOT}/sample_dataset
DEST_DATA_DIR=${PROJECT_ROOT}/sample_databin/${SRC}-${TGT}
mkdir -p ${DEST_DATA_DIR}

fairseq-preprocess --source-lang ${SRC} --target-lang ${TGT} \
--trainpref ${RAW_DATA_DIR}/train \
--destdir ${DEST_DATA_DIR} \
--nwordssrc 120000 \
--nwordstgt 120000 \
--joined-dictionary \
--workers 16

# add additional <mask> <tip> tokens
echo "<mask> 1" >> ${DEST_DATA_DIR}/dict.${SRC}.txt
echo "<tip> 1" >> ${DEST_DATA_DIR}/dict.${SRC}.txt
echo "<mask> 1" >> ${DEST_DATA_DIR}/dict.${TGT}.txt
echo "<tip> 1" >> ${DEST_DATA_DIR}/dict.${TGT}.txt