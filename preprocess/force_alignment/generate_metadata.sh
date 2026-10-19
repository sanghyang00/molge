# Usage: DATA_ROOT=/path/to/datasets bash preprocess/force_alignment/generate_metadata.sh
DATA_ROOT=${DATA_ROOT:?Set DATA_ROOT to your dataset directory}

python generate_alignment_metadata.py -d ${DATA_ROOT}/omnilingual_asr_corpus_16k/csvs/merged_test.csv -o metadata -c 10
python generate_alignment_metadata.py -d ${DATA_ROOT}/omnilingual_asr_corpus_16k/csvs/merged_dev.csv -o metadata -c 10
python generate_alignment_metadata.py -d ${DATA_ROOT}/omnilingual_asr_corpus_16k/csvs/merged_train.csv -o metadata -c 10
