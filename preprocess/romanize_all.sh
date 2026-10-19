# Romanize FLEURS / CommonVoice / OmniASR splits.
# Usage: DATA_ROOT=/path/to/datasets bash preprocess/romanize_all.sh
DATA_ROOT=${DATA_ROOT:?Set DATA_ROOT to your dataset directory}

python romanize_data.py --data_path ${DATA_ROOT} --dataset omnilingual_asr_corpus_16k_aligned --split train --save_dir csvs
python romanize_data.py --data_path ${DATA_ROOT} --dataset omnilingual_asr_corpus_16k_aligned --split dev --save_dir csvs
python romanize_data.py --data_path ${DATA_ROOT} --dataset omnilingual_asr_corpus_16k_aligned --split test --save_dir csvs

python romanize_data.py --data_path ${DATA_ROOT} --dataset commonvoice22 --split train --save_dir csvs
python romanize_data.py --data_path ${DATA_ROOT} --dataset commonvoice22 --split dev --save_dir csvs
python romanize_data.py --data_path ${DATA_ROOT} --dataset commonvoice22 --split test --save_dir csvs

python romanize_data.py --data_path ${DATA_ROOT} --dataset fleurs --split train --save_dir csvs
python romanize_data.py --data_path ${DATA_ROOT} --dataset fleurs --split dev --save_dir csvs
python romanize_data.py --data_path ${DATA_ROOT} --dataset fleurs --split test --save_dir csvs

