# Usage: bash preprocess/force_alignment/rearrange_splits.sh

python rearrange_split.py --sentence_map_dir sentence_map \
                --long_sentence_map_dir sentence_map_outlier \
                --gpt_augmentation_map_dir long_sentence_splits \
                --split train \
                --save_dir sentence_map_rearranged \
                --threshold 5 \
                --min_chars 30 \
                --max_chars 150 \
                --num_workers 16

python rearrange_split.py --sentence_map_dir sentence_map \
                --long_sentence_map_dir sentence_map_outlier \
                --gpt_augmentation_map_dir long_sentence_splits \
                --split dev \
                --save_dir sentence_map_rearranged \
                --threshold 5 \
                --min_chars 30 \
                --max_chars 150 \
                --num_workers 16

python rearrange_split.py --sentence_map_dir sentence_map \
                --long_sentence_map_dir sentence_map_outlier \
                --gpt_augmentation_map_dir long_sentence_splits \
                --split test \
                --save_dir sentence_map_rearranged \
                --threshold 5 \
                --min_chars 30 \
                --max_chars 150 \
                --num_workers 16
