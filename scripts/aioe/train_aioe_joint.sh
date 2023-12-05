export PYTHONIOENCODING=utf-8
save_root=${PROJECT_ROOT}/save
dataset_root=${PROJECT_ROOT}/databin

RUN_NAME=test_joint

# full is the combination of all the suggestion types
SUGGESTION_TYPE=full
SRC=en
TGT=zh
SAVE_DIR=${save_root}/aioe/${SRC}-${TGT}/${SUGGESTION_TYPE}
DATA=${PROJECT_ROOT}/sample_databin/${SRC}-${TGT}
# SUPPLY_PATH: the path of .type and .suggestion files (part of the valid set)
SUPPLY_PATH=${PROJECT_ROOT}/sample_dataset

LR=0.0005
MAX_TOKENS=8192
UPDATE_FREQ=1
WARMUP_UPDATES=4000
SEED=42
WC_LOSS_COEF=0.75

CKPT_DIR=${SAVE_DIR}/${RUN_NAME}
mkdir -p ${CKPT_DIR}

# remove the --no-ar-task option and set the WC_LOSS_COEF
python ${PROJECT_ROOT}/train.py \
    ${DATA} \
    --valid-subset valid --supply-path $SUPPLY_PATH \
    --suggestion-type ${SUGGESTION_TYPE} \
    --user-dir ${PROJECT_ROOT} \
    --task aioe \
    --arch aioe_transformer \
    --save-dir "${CKPT_DIR}" \
    --source-lang ${SRC} \
    --target-lang ${TGT} \
    --left-pad-source \
    --optimizer adam \
    --adam-betas "(0.9, 0.98)"\
    --clip-norm 0.0 \
    --lr ${LR} \
    --seed ${SEED} \
    --lr-scheduler inverse_sqrt \
    --warmup-updates ${WARMUP_UPDATES} \
    --warmup-init-lr 1e-07 \
    --dropout 0.1 \
    --weight-decay 0.0 \
    --criterion aioe_loss \
    --label-smoothing 0.1 \
    --max-tokens ${MAX_TOKENS} \
    --update-freq ${UPDATE_FREQ} \
    --max-update 200000 \
    --tensorboard-logdir ${CKPT_DIR}/log \
    --num-workers 10 \
    --save-interval-updates 4000 \
    --keep-interval-updates 1 \
    --no-epoch-checkpoints \
    --keep-best-checkpoints 2 \
    --best-checkpoint-metric accuracy \
    --maximize-best-checkpoint-metric \
    --log-file ${CKPT_DIR}/train.log \
    --report-accuracy \
    --share-all-embeddings \
    --wc-loss-coef ${WC_LOSS_COEF}
