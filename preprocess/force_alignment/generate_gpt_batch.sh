# Usage: bash preprocess/force_alignment/generate_gpt_batch.sh

python generate_gpt_batch.py --outlier_mapping_dir sentence_map_outlier --split train --save_dir gpt_batch_input
python generate_gpt_batch.py --outlier_mapping_dir sentence_map_outlier --split dev --save_dir gpt_batch_input
python generate_gpt_batch.py --outlier_mapping_dir sentence_map_outlier --split test --save_dir gpt_batch_input
