import ast  # For safely evaluating Python literals
import json
import shutil  # For creating backups
from pathlib import Path


def convert_pseudo_json_file(filepath, backup=True):
    """
    Attempts to convert a single file that might be using Python-style
    single quotes (or other Python literals) into a valid JSON format.

    Args:
        filepath (Path): Path object for the file.
        backup (bool): If True, creates a backup of the original file.

    Returns:
        tuple: (bool: success, str: message)
    """
    try:
        # Read with UTF-8, common for JSON
        original_content = filepath.read_text(encoding='utf-8')
    except Exception as e:
        return False, f"Error reading file '{filepath.name}': {e}"

    # 1. Try to load as valid JSON first. If it works, no changes needed.
    try:
        json.loads(original_content)  # Just try parsing, don't store the result
        return True, f"File '{filepath.name}' is already valid JSON. Skipped."
    except json.JSONDecodeError:
        # This is expected if the file uses single quotes or other non-JSON Python literals.
        # We'll proceed to try ast.literal_eval.
        pass
    except Exception as e:
        # Catch other unexpected errors during the initial JSON load attempt
        return False, f"Unexpected error trying to parse '{filepath.name}' as JSON initially: {e}"

    # 2. If not valid JSON, try to parse as a Python literal and convert.
    try:
        # ast.literal_eval can safely evaluate a string containing a Python literal
        # (dict, list, tuple, string, number, boolean, None)
        python_object = ast.literal_eval(original_content)

        # Convert the Python object to a valid JSON string
        # indent=4 for pretty printing, ensure_ascii=False for better unicode handling
        valid_json_content = json.dumps(python_object, indent=4, ensure_ascii=False)

        # Backup the original file if requested
        if backup:
            backup_filepath = filepath.with_suffix(filepath.suffix + '.bak')
            try:
                shutil.copy2(filepath, backup_filepath)  # copy2 preserves metadata
                # print(f"  Backed up '{filepath.name}' to '{backup_filepath.name}'") # Verbose
            except Exception as e:
                return False, f"Error creating backup for '{filepath.name}': {e}. Conversion aborted for this file."

        # Write the corrected JSON content back to the original file
        filepath.write_text(valid_json_content, encoding='utf-8')
        return True, f"Successfully converted and overwrote '{filepath.name}'."

    except ValueError as ve:  # ast.literal_eval can raise ValueError for malformed strings
        return False, f"File '{filepath.name}' is not valid JSON and could not be parsed as a Python literal (ValueError): {ve}. Skipped."
    except SyntaxError as se:  # ast.literal_eval can raise SyntaxError
        return False, f"File '{filepath.name}' is not valid JSON and could not be parsed as a Python literal (SyntaxError): {se}. Skipped."
    except Exception as e:
        # Catch any other errors during conversion
        return False, f"An unexpected error occurred while converting '{filepath.name}': {e}. Skipped."


def batch_convert_json_files(directory_path_str, create_backups=True):
    """
    Iterates over .json files in the specified directory and attempts to
    convert them from Python-literal style (single quotes) to valid JSON.
    """
    directory_path = Path(directory_path_str)
    if not directory_path.is_dir():
        print(f"Error: Directory '{directory_path_str}' not found or is not a directory.")
        return

    print(f"\nStarting conversion process in directory: {directory_path.resolve()}")
    print("-" * 30)
    print("IMPORTANT:")
    print("1. This script will attempt to OVERWRITE .json files if conversion is successful.")
    if create_backups:
        print("2. Backups of original files WILL be created with a '.bak' extension in the same directory.")
    else:
        print("2. WARNING: Backups are DISABLED. Files will be overwritten directly if conversion succeeds.")
    print("3. It is STRONGLY recommended to have a separate, manual backup of your entire directory before proceeding.")

    proceed = input("Do you understand the risks and want to proceed? (yes/no): ").strip().lower()
    if proceed != 'yes':
        print("Operation cancelled by the user.")
        return

    processed_files_count = 0
    successful_conversions_count = 0
    already_valid_count = 0
    failed_files_list = []

    # Using .glob('*.json') to find all files ending with .json
    json_files_in_dir = list(directory_path.glob('*.json'))

    if not json_files_in_dir:
        print(f"No .json files found in '{directory_path_str}'.")
        return

    print(f"\nFound {len(json_files_in_dir)} .json files to process...")

    for filepath in json_files_in_dir:
        if filepath.is_file():  # Ensure it's a file
            processed_files_count += 1
            print(f"\nProcessing '{filepath.name}'...")
            success, message = convert_pseudo_json_file(filepath, backup=create_backups)
            print(f"  Status: {message}")
            if success:
                if "Successfully converted" in message:
                    successful_conversions_count += 1
                elif "already valid JSON" in message:
                    already_valid_count += 1
            else:
                failed_files_list.append(filepath.name)

    print("\n--- Conversion Summary ---")
    print(f"Total .json files found: {len(json_files_in_dir)}")
    print(f"Files processed: {processed_files_count}")
    print(f"Files successfully converted and overwritten: {successful_conversions_count}")
    print(f"Files already in valid JSON format (skipped): {already_valid_count}")
    print(f"Files that failed conversion or other errors: {len(failed_files_list)}")
    if failed_files_list:
        print("\nFiles that failed or had errors:")
        for f_name in failed_files_list:
            print(f"  - {f_name}")
    print("--- End of Summary ---")


if __name__ == '__main__':
    target_directory = input("Enter the path to the directory containing your JSON files: ").strip()

    backup_choice_str = input(
        "Create backups (.bak) of original files before converting? (yes/no, default: yes): ").strip().lower()
    # Default to yes if user presses Enter or types something other than 'no'
    should_create_backups = backup_choice_str != 'no'

    batch_convert_json_files(target_directory, create_backups=should_create_backups)
