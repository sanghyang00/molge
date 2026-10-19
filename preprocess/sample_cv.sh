# Usage: bash preprocess/sample_cv.sh

python sample_cv.py --input_file ../csvs_with_flag/commonvoice22/train.csv --save_dir ../csvs_with_flag/commonvoice22_sampled --hours 100
python sample_cv.py --input_file ../csvs_with_flag/commonvoice22/dev.csv --save_dir ../csvs_with_flag/commonvoice22_sampled --hours 10
python sample_cv.py --input_file ../csvs_with_flag/commonvoice22/test.csv --save_dir ../csvs_with_flag/commonvoice22_sampled --hours 10
