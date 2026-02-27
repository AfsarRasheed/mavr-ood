import os
import shutil
import argparse
from tqdm import tqdm

def reconstruct_subset(original_dir, id_file, output_dir):
    """
    Reconstructs the Challenging Subset from the original RoadAnomaly dataset
    based on a provided ID list file.
    
    It copies both the original image (from 'original' folder) 
    and its corresponding .png label (from 'labels' folder).
    """
    
    # 1. Define paths for the original dataset
    # -----------------------------------------------------------------
    # Based on user confirmation:
    # Source images are in 'original'
    # Source labels are in 'labels'
    original_images_dir = os.path.join(original_dir, 'original') 
    original_labels_dir = os.path.join(original_dir, 'labels')     
    # -----------------------------------------------------------------
    
    # 2. Define paths for the new subset to be created
    #    (We mirror the original structure)
    output_images_dir = os.path.join(output_dir, 'original')
    output_labels_dir = os.path.join(output_dir, 'labels')
    
    # 3. Create the output directories if they don't exist
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    # 4. Read the ID list file
    try:
        with open(id_file, 'r') as f:
            # Read the exact image filenames (e.g., 'animals03.jpg')
            image_filenames = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: ID file not found at '{id_file}'")
        return
        
    if not image_filenames:
        print(f"Error: The ID file '{id_file}' is empty.")
        return

    print(f"Starting to copy {len(image_filenames)} image/label pairs...")
    
    copied_count = 0
    missing_files = []

    # 5. Iterate through the list and copy files
    for image_filename in tqdm(image_filenames, desc="Reconstructing subset"):
        
        # 5a. Get the base name (e.g., 'animals03.jpg' -> 'animals03')
        basename = os.path.splitext(image_filename)[0]
        
        # 5b. Define the corresponding label filename (e.g., 'animals03.png')
        label_filename = basename + '.png'
        
        # 5c. Define full source and destination paths
        src_image_path = os.path.join(original_images_dir, image_filename)
        src_label_path = os.path.join(original_labels_dir, label_filename)
        
        dst_image_path = os.path.join(output_images_dir, image_filename)
        dst_label_path = os.path.join(output_labels_dir, label_filename)
        
        # 5d. Check if both files exist and then copy
        image_exists = os.path.exists(src_image_path)
        label_exists = os.path.exists(src_label_path)
        
        if image_exists and label_exists:
            shutil.copy(src_image_path, dst_image_path)
            shutil.copy(src_label_path, dst_label_path)
            copied_count += 1
        else:
            # Log files that were not found
            if not image_exists:
                missing_files.append(f"[Image not found] {src_image_path}")
            if not label_exists:
                missing_files.append(f"[Label not found] {src_label_path}")

    print(f"\nCopy complete: Successfully copied {copied_count} out of {len(image_filenames)} pairs.")
    
    if missing_files:
        print("\nWarning: The following files could not be found in the original directory:")
        for missing in missing_files:
            print(f"  - {missing}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruct the RoadAnomaly Challenging Subset from the original dataset.")
    
    parser.add_argument('--original_dir', type=str, required=True,
                        help="Path to the root of the original RoadAnomaly dataset (must contain 'original' and 'labels' sub-folders).")
                        
    parser.add_argument('--id_file', type=str, required=True,
                        help="Path to the .txt file containing the image filenames (with extensions) for the challenging subset.")
                        
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Path to the new directory where the challenging subset will be saved.")
                        
    args = parser.parse_args()
    
    reconstruct_subset(args.original_dir, args.id_file, args.output_dir)
