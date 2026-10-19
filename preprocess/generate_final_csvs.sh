# Build final CSV manifests with audio paths.
# Usage: DATA_ROOT=/path/to/datasets bash preprocess/generate_final_csvs.sh
DATA_ROOT=${DATA_ROOT:?Set DATA_ROOT to your dataset directory}

python id_to_filepath.py --data_root "$DATA_ROOT" --input_csv csvs_with_flag/commonvoice22/train.csv --output_csv final_csvs/commonvoice22/train.csv --dataset commonvoice --split train
python id_to_filepath.py --data_root "$DATA_ROOT" --input_csv csvs_with_flag/commonvoice22/dev.csv --output_csv final_csvs/commonvoice22/dev.csv --dataset commonvoice --split dev
python id_to_filepath.py --data_root "$DATA_ROOT" --input_csv csvs_with_flag/commonvoice22/test.csv --output_csv final_csvs/commonvoice22/test.csv --dataset commonvoice --split test

python id_to_filepath.py --data_root "$DATA_ROOT" --input_csv csvs_with_flag/commonvoice22_sampled/train.csv --output_csv final_csvs/commonvoice22_sampled/train.csv --dataset commonvoice --split train
python id_to_filepath.py --data_root "$DATA_ROOT" --input_csv csvs_with_flag/commonvoice22_sampled/dev.csv --output_csv final_csvs/commonvoice22_sampled/dev.csv --dataset commonvoice --split dev
python id_to_filepath.py --data_root "$DATA_ROOT" --input_csv csvs_with_flag/commonvoice22_sampled/test.csv --output_csv final_csvs/commonvoice22_sampled/test.csv --dataset commonvoice --split test

python id_to_filepath.py --data_root "$DATA_ROOT" --input_csv csvs_with_flag/fleurs/train.csv --output_csv final_csvs/fleurs/train.csv --dataset fleurs --split train
python id_to_filepath.py --data_root "$DATA_ROOT" --input_csv csvs_with_flag/fleurs/dev.csv --output_csv final_csvs/fleurs/dev.csv --dataset fleurs --split dev
python id_to_filepath.py --data_root "$DATA_ROOT" --input_csv csvs_with_flag/fleurs/test.csv --output_csv final_csvs/fleurs/test.csv --dataset fleurs --split test

python id_to_filepath.py --data_root "$DATA_ROOT" --input_csv csvs_with_flag/cs-fleurs-v2/train.csv --output_csv final_csvs/cs-fleurs-v2/train.csv --dataset cs-fleurs-v2 --split train
python id_to_filepath.py --data_root "$DATA_ROOT" --input_csv csvs_with_flag/cs-fleurs-v2/dev.csv --output_csv final_csvs/cs-fleurs-v2/dev.csv --dataset cs-fleurs-v2 --split dev
python id_to_filepath.py --data_root "$DATA_ROOT" --input_csv csvs_with_flag/cs-fleurs-v2/test.csv --output_csv final_csvs/cs-fleurs-v2/test.csv --dataset cs-fleurs-v2 --split test
