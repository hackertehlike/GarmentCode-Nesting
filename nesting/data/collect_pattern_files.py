import os
import shutil
import argparse

def collect_files(source_dir, dest_dir, file_extensions=['.json', '.yaml', '.yml']):
    """
    Collect files with specified extensions from source_dir and copy them to dest_dir
    while preserving the garment folder structure.
    
    Args:
        source_dir (str): Source directory (T7 mount path)
        dest_dir (str): Destination directory
        file_extensions (list): List of file extensions to collect
    """
    # Get list of garment folders (rand_XXX folders)
    data_dir = os.path.join(source_dir, "data")
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found at {data_dir}")
        return
    
    garment_folders = [f for f in os.listdir(data_dir) 
                     if os.path.isdir(os.path.join(data_dir, f)) and f.startswith("rand_")]
    
    print(f"Found {len(garment_folders)} garment folders")
    
    # Process each garment folder
    for garment_folder in garment_folders:
        garment_path = os.path.join(data_dir, garment_folder)
        garment_dest = os.path.join(dest_dir, garment_folder)

        # Walk through the garment folder and collect matching files first
        matches = []
        for root, _, files in os.walk(garment_path):
            for filename in files:
                if any(filename.lower().endswith(ext) for ext in file_extensions):
                    matches.append((root, filename))

        if not matches:
            print(f"Skipping {garment_folder}: no matching files found")
            continue

        # Create destination folder for this garment only if matches exist
        os.makedirs(garment_dest, exist_ok=True)

        # Copy the collected files
        for root, filename in matches:
            source_path = os.path.join(root, filename)

            # Create relative path to preserve structure
            rel_path = os.path.relpath(root, garment_path)
            if rel_path == ".":
                dest_path = os.path.join(garment_dest, filename)
            else:
                subfolder = os.path.join(garment_dest, rel_path)
                os.makedirs(subfolder, exist_ok=True)
                dest_path = os.path.join(subfolder, filename)

            try:
                shutil.copy2(source_path, dest_path)
                print(f"Copied {source_path} → {dest_path}")
            except shutil.SameFileError:
                print(f"Skipped (same file exists): {dest_path}")
            except Exception as e:
                print(f"Error copying {source_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Collect JSON and YAML files from T7 volume')
    parser.add_argument('--source', type=str, required=True, 
                        help='Source directory (T7 mount path)')
    parser.add_argument('--dest', type=str, default='pattern_files',
                        help='Destination directory (default: pattern_files)')
    parser.add_argument('--extensions', type=str, default='.json,.yaml,.yml',
                        help='Comma-separated list of file extensions to collect (default: .json,.yaml,.yml)')
    
    args = parser.parse_args()
    
    extensions = [ext.strip() for ext in args.extensions.split(',')]
    collect_files(args.source, args.dest, extensions)

if __name__ == "__main__":
    main()
