import os, json
import argparse

def main():
    
    parser = argparse.ArgumentParser(description="Merge multiple JSONL response files into a single JSON mapping file.")
    parser.add_argument('--data_dir', type=str, default='gpt_batch_output', 
                        help='Base directory containing split subdirectories (e.g., train, dev, test).')
    parser.add_argument('--split', type=str, default='train', 
                        help='The specific split (subdirectory) to process (e.g., train, dev, test).')
    parser.add_argument('--save_dir', type=str, default='long_sentence_splits', 
                        help='Directory where the final merged JSON file will be saved.')
    args = parser.parse_args()
    
    # ----------------------------------------------------
    # ----------------------------------------------------
    
    input_dir = os.path.join(args.data_dir, args.split) 
    
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return
        
    os.makedirs(args.save_dir, exist_ok=True)
    
    sentence_mapping = {}
    
    print(f"🔍 Reading and merging JSONL files from: {input_dir}")
    
    # ----------------------------------------------------
    # ----------------------------------------------------
    
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.jsonl'):
            file_path = os.path.join(input_dir, file_name)
            print(f"  -> Processing {file_name}...")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip():
                            continue
                            
                        gpt_response = json.loads(line.strip())
                        
                        fid = gpt_response.get('custom_id') 
                        
                        try:
                            content = gpt_response.get('response', {}).get('body', {}).get('choices', [{}])[0].get('message', {}).get('content')
                        except (KeyError, IndexError, TypeError):
                            print(f"Warning: Failed to parse content for custom_id: {fid} in {file_name}. Skipping.")
                            continue

                        if fid and content:
                            sentence_mapping[fid] = content
                        
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")
                
    print(f"\n✅ Total {len(sentence_mapping)} mappings collected.")
    
    # ----------------------------------------------------
    # ----------------------------------------------------
    
    save_path = os.path.join(args.save_dir, f'{args.split}.json')
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(sentence_mapping, f, ensure_ascii=False, indent=4)
                    
    print(f"🎉 Successfully saved merged JSON to: {save_path}")
                    
if __name__ == '__main__':
    main()
