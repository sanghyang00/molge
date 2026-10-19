# Usage: bash preprocess/force_alignment/split_into_sentences_manual.sh

python split_into_sentences.py --csv_dir preprocessed_csvs --split train --save_dir sentence_map_manual
python split_into_sentences.py --csv_dir preprocessed_csvs --split dev --save_dir sentence_map_manual
python split_into_sentences.py --csv_dir preprocessed_csvs --split test --save_dir sentence_map_manual
