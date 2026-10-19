# Usage: bash preprocess/force_alignment/sort_outlier_sentences.sh

python sort_long_sentences.py --input_dir sentence_map --split train --output_dir sentence_map_outlier --threshold 150
python sort_long_sentences.py --input_dir sentence_map --split dev --output_dir sentence_map_outlier --threshold 150
python sort_long_sentences.py --input_dir sentence_map --split test --output_dir sentence_map_outlier --threshold 150
