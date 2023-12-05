# evaluate a saved checkpoint

checkpoint_path=${PROJECT_ROOT}/save/aioe/en-zh/full/test/checkpoint_best.pt
# input_path: json format data
input_path=${PROJECT_ROOT}/sample_dataset/test.en-zh.json
# output_path: path to save predictions
output_path=${PROJECT_ROOT}/save/aioe/en-zh/full/test/pred.txt

python ./aioe_generation_pipeline.py generate \
    --checkpoint_path ${checkpoint_path} \
    --input_path ${input_path} \
    --output_path ${output_path} 

python ./utils.py compute_acc \
    --input_path ${output_path}