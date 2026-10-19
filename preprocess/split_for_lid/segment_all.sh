# Usage: DATA_ROOT=/path/to/datasets bash preprocess/split_for_lid/segment_all.sh
DATA_ROOT=${DATA_ROOT:?Set DATA_ROOT to your dataset directory}

python segment_audio_for_lid.py --csv_path ${DATA_ROOT}/omnilingual_asr_corpus_16k/csvs/merged_test.csv --output_dir ${DATA_ROOT}/omnilingual_asr_corpus_16k_for_lid --max_duration 15 --num_workers 16
# python segment_audio_for_lid.py --csv_path ${DATA_ROOT}/omnilingual_asr_corpus_16k/csvs/merged_dev.csv --output_dir ${DATA_ROOT}/omnilingual_asr_corpus_16k_for_lid --max_duration 15 --num_workers 16
python segment_audio_for_lid.py --csv_path ${DATA_ROOT}/omnilingual_asr_corpus_16k/csvs/merged_train.csv --output_dir ${DATA_ROOT}/omnilingual_asr_corpus_16k_for_lid --max_duration 15 --num_workers 16
