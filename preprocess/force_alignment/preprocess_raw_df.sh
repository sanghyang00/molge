# Usage: DATA_ROOT=/path/to/datasets bash preprocess/force_alignment/preprocess_raw_df.sh
DATA_ROOT=${DATA_ROOT:?Set DATA_ROOT to your dataset directory}

python preprocess_raw_df.py --data_dir ${DATA_ROOT}/omnilingual_asr_corpus_16k --split train --save_dir preprocessed_csvs
python preprocess_raw_df.py --data_dir ${DATA_ROOT}/omnilingual_asr_corpus_16k --split dev --save_dir preprocessed_csvs
python preprocess_raw_df.py --data_dir ${DATA_ROOT}/omnilingual_asr_corpus_16k --split test --save_dir preprocessed_csvs
