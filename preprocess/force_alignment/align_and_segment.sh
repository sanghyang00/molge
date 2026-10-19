# Usage: DATA_ROOT=/path/to/datasets bash preprocess/force_alignment/align_and_segment.sh
DATA_ROOT=${DATA_ROOT:?Set DATA_ROOT to your dataset directory}

python align_and_segment.py -d preprocessed_csvs -m sentence_map_manual -t train -o ${DATA_ROOT}/omnilingual_asr_corpus
python align_and_segment.py -d preprocessed_csvs -m sentence_map_manual -t dev -o ${DATA_ROOT}/omnilingual_asr_corpus
python align_and_segment.py -d preprocessed_csvs -m sentence_map_manual -t test -o ${DATA_ROOT}/omnilingual_asr_corpus
