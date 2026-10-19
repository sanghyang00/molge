# Usage: DATA_ROOT=/path/to/datasets bash preprocess/force_alignment/merge_and_save_segments.sh
DATA_ROOT=${DATA_ROOT:?Set DATA_ROOT to your dataset directory}

python merge_alignments.py --metadata_dir metadata --split train --score_threshold 0.85 --max_duration 30.0 --offset 0.05 --save_dir ${DATA_ROOT}/omnilingual_asr_corpus_16k_segments_30sec --num_workers 16
python merge_alignments.py --metadata_dir metadata --split dev --score_threshold 0.85 --max_duration 30.0 --offset 0.05 --save_dir ${DATA_ROOT}/omnilingual_asr_corpus_16k_segments_30sec --num_workers 16
python merge_alignments.py --metadata_dir metadata --split test --score_threshold 0.85 --max_duration 30.0 --offset 0.05 --save_dir ${DATA_ROOT}/omnilingual_asr_corpus_16k_segments_30sec --num_workers 16
