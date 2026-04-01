import glob
import os
import zipfile

import kaggle
from CleanCodeRelease.helpers.file_system import cleanup_redundant_nested_folders

# --- Configuration (same as before) ---
DATASET_SLUG = 'jessicali9530/stanford-cars-dataset'  # Or your dataset
DOWNLOAD_ROOT_DIR = './datasets'
TARGET_DATA_DIR = os.path.join(DOWNLOAD_ROOT_DIR, 'stanford_cars')
# CHECK_FOLDER_NAME is what you expect after *initial extraction* to decide if download is needed.
# The cleanup happens *after* this decision and initial extraction.
# For jessicali9530/stanford-cars-dataset, 'car_data' is a good top-level item after extraction.
CHECK_FOLDER_NAME = 'car_data'


# (The cleanup_redundant_nested_folders function from above should be defined here or imported)

def download_and_unzip_kaggle_dataset(dataset_slug, download_path, expected_check_item):
    """
    Downloads a Kaggle dataset, manually unzips, and then cleans up redundant nested folders.
    """
    full_check_path = os.path.join(download_path, expected_check_item)

    if os.path.exists(full_check_path):
        print(
            f"Dataset's expected item '{expected_check_item}' already found in '{download_path}'. Skipping download and extraction.")
        # Even if already downloaded, you might want to run cleanup if it hasn't run before
        # or if the definition of "clean" has changed. For simplicity, we only run cleanup
        # if we also download. You can separate this if needed.
        # cleanup_redundant_nested_folders(download_path) # Optional: run cleanup even if not re-downloading
        return
    else:
        print(
            f"Dataset's expected item '{expected_check_item}' not found. Proceeding with download and processing.")

    os.makedirs(download_path, exist_ok=True)

    print(f"Initializing Kaggle API...")
    try:
        api = kaggle.KaggleApi()
        api.authenticate()
        print(
            f"Downloading dataset '{dataset_slug}' to '{download_path}' (zip files only, no auto-unzip):")

        api.dataset_download_files(dataset_slug,
                                   path=download_path,
                                   unzip=False,
                                   quiet=False)
        print("\nDownload complete. Proceeding with manual unzipping.")

        zip_files_found = glob.glob(os.path.join(download_path, "*.zip"))

        if not zip_files_found:
            print(f"No .zip files found in '{download_path}' after download.")
            # Check if the expected item is NOW present (e.g. dataset wasn't zipped)
            if not os.path.exists(full_check_path):
                print(
                    f"And expected item '{expected_check_item}' is also NOT present. Download might have failed or dataset structure is unexpected.")
                return  # Critical failure if no zips and no expected data
            else:
                print(
                    f"However, expected item '{expected_check_item}' is present. Assuming data is okay (not zipped or already extracted).")
                # Proceed to cleanup if data is present but wasn't from a zip we processed now
                cleanup_redundant_nested_folders(download_path)
                return

        for zip_filepath in zip_files_found:
            zip_filename = os.path.basename(zip_filepath)
            print(f"Processing '{zip_filename}'...")
            try:
                with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
                    print(
                        f"  Extracting '{zip_filename}' directly into '{download_path}'...")
                    zip_ref.extractall(download_path)
                print(f"  Successfully extracted '{zip_filename}'.")
                try:
                    os.remove(zip_filepath)
                    print(f"  Removed '{zip_filename}'.")
                except OSError as e:
                    print(f"  Error removing '{zip_filename}': {e}")
            except zipfile.BadZipFile:
                print(
                    f"  Error: '{zip_filename}' is not a valid zip file or is corrupted.")
            except Exception as e:
                print(
                    f"  An error occurred while processing '{zip_filename}': {e}")

        # --- Call the cleanup function AFTER all zips are processed ---
        # Only cleanup if we actually got data
        if os.path.exists(full_check_path) or zip_files_found:
            cleanup_redundant_nested_folders(download_path)
        else:
            print(
                f"\nSkipping cleanup as initial expected data or zip files were not found/processed.")

        # Final verification based on the initial check item
        if os.path.exists(full_check_path):
            print(
                f"\nSuccessfully processed. Initial check item '{expected_check_item}' found in '{download_path}'.")
        else:
            # This check might be tricky if the cleanup renames the expected_check_item itself.
            # The check should ideally be for a more fundamental piece of data.
            print(
                f"\nNote: Processing finished. The initial check item '{expected_check_item}' might have been affected by cleanup if it was part of a redundant hierarchy.")
            print(f"Please verify the contents of '{download_path}'.")

    except Exception as e:
        print(f"An error occurred during the Kaggle API operation: {e}")
        # ... (other error messages from before)


if __name__ == "__main__":
    if not os.path.exists(DOWNLOAD_ROOT_DIR):
        os.makedirs(DOWNLOAD_ROOT_DIR)

    download_and_unzip_kaggle_dataset(
        DATASET_SLUG, TARGET_DATA_DIR, CHECK_FOLDER_NAME)

    if os.path.exists(TARGET_DATA_DIR):
        print(f"\nFinal contents of '{TARGET_DATA_DIR}' after all processing:")
        for root, dirs, files in os.walk(TARGET_DATA_DIR):
            level = root.replace(TARGET_DATA_DIR, '').count(os.sep)
            indent = ' ' * 4 * (level)
            print(f'{indent}{os.path.basename(root)}/')
            sub_indent = ' ' * 4 * (level + 1)
            for f in files:
                print(f'{sub_indent}{f}')
