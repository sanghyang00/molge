"""Submit / retrieve OpenAI batch jobs for force-alignment GPT postprocessing.

Usage:
  export OPENAI_API_KEY=...
  python preprocess/force_alignment/run_gpt_inference.py --split train
"""

import os, sys, re, time
from argparse import ArgumentParser
from openai import OpenAI

if not os.environ.get('OPENAI_API_KEY'):
    raise SystemExit('Set OPENAI_API_KEY in the environment before running this script.')

def retrieve_and_save(client, batch_id, savepath):
    batch_info = client.batches.retrieve(batch_id)
    result_file_id = batch_info.output_file_id

    if result_file_id is None:
        print(f"No result file for batch {batch_id}")
        sys.exit(1)

    response = client.files.content(result_file_id)
    result_content = response.read().decode("utf-8")

    with open(savepath, "w", encoding="utf-8") as f:
        f.write(result_content)

    print(f"Saved result to {savepath}")

def main():
    
    parser = ArgumentParser()
    parser.add_argument('--jsonl_dir', type=str, default='gpt_batch_input')
    parser.add_argument('--save_dir', type=str, default='gpt_batch_output')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--process_numbers', type=int, nargs='+', default=None, help='List of numbers to process')
    
    args = parser.parse_args()
    
    client = OpenAI()
    
    all_files = os.listdir(os.path.join(args.jsonl_dir, args.split))
    target_files = []

    for f in all_files:
        match = re.search(r"batch_input_(\d+)\.jsonl", f)
        if match:
            file_number = int(match.group(1))
            if args.process_numbers is None or file_number in args.process_numbers:
                target_files.append(f)

    files = sorted(target_files)
    
    print('Collected files:')
    for fname in files:
        print(fname)
    
    savedir = os.path.join(args.save_dir, args.split)
    os.makedirs(savedir, exist_ok=True)
    for fname in files:
        
        fid = fname.split('.')[0]
        savepath = os.path.join(savedir, f"{fid}.jsonl")
        
        batch_input_file = client.files.create(
            file=open(os.path.join(args.jsonl_dir, args.split, fname), "rb"),
            purpose="batch"
        )

        batch_input_file_id = batch_input_file.id
        batch = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        print(f'Metadata of the created batch:\n{batch}')
        
        time.sleep(30)
        
        while True:
            current_status = client.batches.retrieve(batch.id).status
            print(f'Batch ID: {batch.id}, Current Status: {current_status}')
            
            if current_status == "failed":
                print(f"Error: Batch ID {batch.id} failed. Exiting the program.")
                sys.exit(1)
            
            elif current_status == "expired":
                print(f"Batch ID {batch.id} has expired. Exiting the program.")
                sys.exit(1) 
            
            elif current_status == "completed":
                print(f"Batch ID {batch.id} completed successfully.")
                retrieve_and_save(client, batch.id, savepath)
                print(f"Batch ID {batch.id} saved successfully in {savepath}.")
                break 
            
            elif current_status == "in_progress":
                print(f"Batch ID {batch.id} is in progress.")
            
            time.sleep(600)
    
if __name__=='__main__':
    main()
