#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script processes a single CSV file, samples it by language
to a specified maximum duration, and saves the new "balanced" CSV
to an output directory.
"""

import os, warnings, argparse
import pandas as pd

warnings.filterwarnings('ignore')

def cut_dataset_by_duration(df, hours_per_lang):
    """
    Cuts the DataFrame by language to a specified total duration.

    Args:
        df (pd.DataFrame): The source DataFrame, containing
                         'matrix language' and 'duration' columns.
        hours_per_lang (float or int): Target total hours per language.

    Returns:
        pd.DataFrame: A new DataFrame with the sampled data.
    """
    
    threshold_seconds = hours_per_lang * 3600
    
    def _cut_group(group):
        """
        Helper function for groupby.apply:
        Shuffles each language group, calculates cumsum, and cuts.
        """
        shuffled_group = group.sample(frac=1, random_state=42) # Set random_state for reproducibility
        
        cumulative_durations = shuffled_group['duration'].cumsum()
        
        cut_group = shuffled_group[cumulative_durations <= threshold_seconds]
        
        return cut_group

    print(f"Target time: {hours_per_lang} hours per language ({threshold_seconds:,.0f} seconds)")
    
    cut_df = (
        df.groupby('matrix language', group_keys=False)
          .apply(_cut_group)
          .reset_index(drop=True)
    )
    
    return cut_df

def main():
    """
    Main function to parse arguments and process a single file.
    """
    parser = argparse.ArgumentParser(description="Balance a dataset by cutting each language to a max duration.")
    
    parser.add_argument('--input_file', type=str, required=True,
                        help="Path to the single input CSV file.")
                        
    parser.add_argument('--save_dir', type=str, required=True,
                        help="Directory to save the new balanced CSV file.")
    parser.add_argument('--hours', type=float, required=True,
                        help="Target maximum hours per language (e.g., 100).")
    
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Input file: {args.input_file}")
    print(f"Output directory: {args.save_dir}")
    print(f"Target hours per language: {args.hours}")

    input_path = args.input_file
    output_filename = os.path.basename(input_path) # Get the filename from the path
    output_path = os.path.join(args.save_dir, output_filename)
    
    try:
        # Load data
        print(f"\nLoading {input_path}...")
        data = pd.read_csv(input_path)
        
        # Check for required columns
        if 'matrix language' not in data.columns or 'duration' not in data.columns:
            print(f"Error: Skipping {input_path}. It lacks 'matrix language' or 'duration' column.")
            return

        # Apply sampling function
        print(f"Processing {output_filename}...")
        sampled_data = cut_dataset_by_duration(data, args.hours)
        
        # Save data
        sampled_data.to_csv(output_path, index=False)
        print(f"✅ Successfully saved balanced file to {output_path} ({len(sampled_data)} rows)")
    
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
    except Exception as e:
        print(f"An error occurred while processing {input_path}: {e}")
    
    print("\nProcessing finished.")

# Standard main execution block
if __name__ == "__main__":
    main()
