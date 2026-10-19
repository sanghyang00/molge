# Usage: bash preprocess/romanize_test.sh

python romanize_data.py --dataset commonvoice22 --split test --save_dir temp
python romanize_data.py --dataset commonvoice22 --split dev --save_dir temp
python romanize_data.py --dataset commonvoice22 --split train --save_dir temp
