import glob
import os
import random
import shutil  # For moving and removing directories
from pathlib import Path


def read_filenames(file_path):
    """
    Reads a file and extracts filenames into a list.

    Args:
      file_path: The path to the input text file.

    Returns:
      A list of filenames.
    """
    filenames = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                filenames.append(parts[1])
    return filenames


def cleanup_redundant_nested_folders(base_path):
    """
    Cleans up folder structures like base_path/FolderA/FolderB/...
    where FolderA only contains FolderB and no other files.
    In such cases, FolderB is "promoted" to replace FolderA (i.e., FolderB
    is moved up and renamed to FolderA's original name).
    This process is repeated until no such structures are found within base_path.

    Args:
        base_path (str): The root directory within which to perform the cleanup.
    """
    print(f"\nStarting redundant folder cleanup within '{base_path}'...")
    while True:
        made_change_in_pass = False
        # We use os.walk with topdown=False to process deeper directories first.
        # This helps simplify logic when a promotion might enable another promotion
        # at a higher level in the same pass, though the while loop handles full re-scans.
        for dir_path, sub_dir_names, file_names in os.walk(base_path, topdown=False):
            # We're interested in 'dir_path' (this is our potential FolderA)
            # if it's a sub-directory of base_path (not base_path itself)
            # and if it contains exactly one sub-directory and no files.
            if dir_path == base_path:
                continue  # Don't try to modify the base_path itself with this logic

            if len(sub_dir_names) == 1 and not file_names:
                # This 'dir_path' is FolderA.
                # 'sub_dir_names[0]' is the name of FolderB.
                folder_A_path = dir_path
                folder_B_name = sub_dir_names[0]
                folder_B_path = os.path.join(folder_A_path, folder_B_name)

                # Get details of FolderA for renaming FolderB later
                parent_of_A_path = os.path.dirname(folder_A_path)
                original_A_name = os.path.basename(folder_A_path)

                # Temporary path for FolderB at the same level as FolderA.
                # Suffix to avoid collision if FolderB's name already exists at A's level,
                # or if FolderB's name is the same as FolderA's name.
                temp_promoted_B_name = folder_B_name + "__promoting_temp"
                temp_storage_path_for_B = os.path.join(parent_of_A_path, temp_promoted_B_name)

                # Safety check for temporary path
                if os.path.exists(temp_storage_path_for_B):
                    print(
                        f"  Skipping promotion for '{folder_A_path}': Temporary path '{temp_storage_path_for_B}' already exists. Manual cleanup might be needed.")
                    continue

                print(
                    f"  Action: Folder '{folder_A_path}' contains only sub-folder '{folder_B_name}'. Promoting '{folder_B_name}'.")

                try:
                    # 1. Move FolderB out from FolderA to the temporary location (sibling to FolderA)
                    # e.g., move base_path/A/B  ->  base_path/B_temp
                    shutil.move(folder_B_path, temp_storage_path_for_B)
                    print(f"    Moved '{folder_B_path}' to temporary location '{temp_storage_path_for_B}'")

                    # 2. Delete the now-empty FolderA
                    # e.g., delete base_path/A
                    shutil.rmtree(folder_A_path)
                    print(f"    Removed original empty folder '{folder_A_path}'")

                    # 3. Rename the moved FolderB (from its temp path) to FolderA's original name and location
                    # e.g., rename base_path/B_temp  ->  base_path/A (which is now the content of old B)
                    final_path_for_promoted_B = os.path.join(parent_of_A_path, original_A_name)
                    shutil.move(temp_storage_path_for_B, final_path_for_promoted_B)
                    print(f"    Renamed temporary '{temp_storage_path_for_B}' to '{final_path_for_promoted_B}'")

                    made_change_in_pass = True
                    # Since the directory structure has changed, break from os.walk
                    # and restart the 'while True' loop to get a fresh view of the directories.
                    break
                except Exception as e:
                    print(f"    ERROR during promotion of '{folder_B_name}' from '{folder_A_path}': {e}")
                    print(
                        f"    The directory structure might be in an inconsistent state. Manual inspection of '{base_path}' is advised.")
                    # Attempting to restore automatically is complex and risky here.
                    # If temp_storage_path_for_B still exists, it's an orphaned part of the operation.
                    if os.path.exists(temp_storage_path_for_B):
                        print(f"    A temporary folder '{temp_storage_path_for_B}' might still exist.")
                    # We will try to continue the while loop to see if other changes can be made, or exit if this was the only one.

            if made_change_in_pass:  # If inner loop broke due to a change
                break

        if not made_change_in_pass:
            # No changes were made in a full pass of os.walk, so the structure is stable.
            print("Redundant folder cleanup: No further changes detected in this pass. Structure is stable.")
            break
        else:
            # If changes were made, the 'while True' loop will continue, re-running os.walk.
            print("Redundant folder cleanup: Changes were made. Re-scanning directory structure...")

    print("Finished redundant folder cleanup process.")


def find_file_in_hierarchy(start_dir, filename):
    """
    Searches for a file in a given directory and its parent directories
    until the file is found or the root directory is reached.

    Args:
        start_dir (str): The absolute or relative path to the directory
                         where the search should begin.
        filename (str): The name of the file to search for.

    Returns:
        str: The absolute path to the directory containing the found file.

    Raises:
        ValueError: If `start_dir` is not a valid directory, or if
                    `filename` is empty or invalid.
        FileNotFoundError: If the file is not found in the `start_dir`
                           or any of its parent directories up to the root.
    """
    # Validate the filename
    if not filename:
        raise ValueError("Filename cannot be empty.")
    if os.path.sep in filename or (os.altsep and os.altsep in filename):
        raise ValueError(f"Filename '{filename}' should not contain path separators. Provide only the file's name.")

    # Validate and normalize the starting directory
    if not os.path.isdir(start_dir):
        raise ValueError(f"The starting path '{start_dir}' is not a valid directory or does not exist.")

    current_dir = os.path.abspath(start_dir)  # Start with the absolute path

    # Loop indefinitely until the file is found or the root is unequivocally reached
    while True:
        potential_file_path = os.path.join(current_dir, filename)

        # Check if the file exists at the potential path
        if os.path.isfile(potential_file_path):
            return Path(current_dir)  # File found, return its directory's absolute path

        # Get the parent directory of the current directory
        parent_dir = os.path.dirname(current_dir)

        # Check if we have reached the root of the filesystem
        # This happens when the parent directory is the same as the current directory
        # (e.g., dirname('/') is '/', dirname('C:\\') is 'C:\\')
        if parent_dir == current_dir:
            # If we are at the root and haven't found the file, it doesn't exist in the hierarchy
            original_start_abs_path = os.path.abspath(start_dir)  # For a clear error message
            raise FileNotFoundError(
                f"File '{filename}' not found in '{original_start_abs_path}' "
                f"or any of its parent directories up to the root."
            )

        # Move up to the parent directory for the next iteration
        current_dir = parent_dir


def get_path_components(path_string: str) -> list[str]:
    """
    Extracts the components of a path string into a list.
    For absolute paths, the anchor (root) is excluded.
    For example, "/home/user/file.txt" becomes ["home", "user", "file.txt"].
    A relative path like "project/data" becomes ["project", "data"].

    Args:
        path_string: The path string to process.

    Returns:
        A list of strings, where each string is a component of the path.
    """
    p = Path(path_string)
    path_parts_tuple = p.parts

    if p.is_absolute():
        # For an absolute path, path_parts_tuple[0] is the anchor (e.g., '/', 'C:\\').
        # We return the list of components after this anchor.
        return list(path_parts_tuple[1:])
    else:
        # For a relative path, all parts are considered components.
        return list(path_parts_tuple)


def read_file(file_path, mode="img_path"):
    """
    Reads file containing image paths or bounding box coordinates.

    Args:
        file_path (str): Path to the file.

    Returns:
        list: List of tuples containing file content.
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()

            if mode == "img_path":
                data.append((int(parts[0]), parts[1]))

            elif mode == "bbox":
                data.append((int(parts[0]), *map(int, map(float, parts[1:]))))

            else:
                raise ValueError("Invalid mode. Must be either 'img_path' or 'bbox'.")

    return data


def get_random_img_paths(folder: str, n: int, seed: int = 42) -> list[str]:
    random.seed(seed)
    paths = glob.glob(os.path.join(folder, "*", "*.jpg"))
    return random.sample(paths, n)


def get_folder_count(folder_path):
    if os.path.exists(folder_path):
        return len(os.listdir(folder_path))
    else:
        return 0


def gen_sample_paths(num_classes: int = 4,
                     samples_per_class: int = 2,
                     class_idx: int = None,
                     class_indices: list = None,
                     return_img_selection: bool = True):
    base_folder = Path.home() / "tmp/Datasets/CUB200/CUB_200_2011" / "images"
    folder_list = os.listdir(base_folder)

    if '.DS_Store' in folder_list:
        folder_list.remove('.DS_Store')

    sample_paths = []

    if class_idx is not None:
        class_folder = sorted(folder_list, key=lambda x: int(x.split(".")[0]))[class_idx - 1]

        # Pick the first 4 images
        # for the selected class (class_idx)
        max_num = len(os.listdir(base_folder / class_folder))
        if samples_per_class == -1 or samples_per_class > max_num:
            samples_per_class = max_num

        samples = os.listdir(base_folder / class_folder)
        sample_paths_raw = random.sample(samples, samples_per_class)

        sample_paths = [base_folder / class_folder / path for path in sample_paths_raw]

        if samples_per_class == -1 and num_classes == 1:
            samples_per_class = len(sample_paths)

        img_selection = random.sample(range(len(sample_paths)), min(samples_per_class, 4))

    else:
        if class_indices is None:
            if num_classes == -1:
                class_indices = list(range(1, 201))
            else:
                class_indices = random.sample(range(1, 200), num_classes)

        for idx in class_indices:
            class_folder = sorted(folder_list, key=lambda x: int(x.split(".")[0]))[idx - 1]
            filenames = os.listdir(base_folder / class_folder)

            if samples_per_class == -1:
                sample_paths.extend([str(base_folder / class_folder / file) for file in filenames])

            else:
                sample_paths.extend([str(base_folder / class_folder / file) for file in random.sample(filenames,
                                                                                                      samples_per_class)])

        if return_img_selection:
            if samples_per_class == -1:
                samples_per_class = 4
            img_selection = random.sample(range(len(sample_paths)), min(samples_per_class, 4))

    if return_img_selection:
        return sample_paths, img_selection

    return sample_paths


def extract_output_dir(img_path: str | Path, folder: str | Path) -> Path:
    if isinstance(img_path, str):
        img_path = Path(img_path)

    if isinstance(folder, str):
        folder = Path(folder)

    # Extract folder containing the image and image name
    folder_name = os.path.basename(os.path.dirname(img_path))
    img_name = os.path.basename(img_path)

    # Remove the file extension from the image name
    image_name_without_ext = os.path.splitext(img_name)[0]

    # Concatenate folder name and image name without extension and return it
    return folder / folder_name / image_name_without_ext


if __name__ == "__main__":
    pth = "/home/robert/tmp/dinov2/CUB2011/normal/hp_test/829188_28/ft"
    print(find_file_in_hierarchy(pth, "Trained_DenseModel.pth"))
