def _find_all_paths_recursive(current_element, target_key, current_path, found_paths_list):
    """
    Recursively searches for all occurrences of the target_key in a nested structure
    (dict or list) and appends their paths to found_paths_list.

    Args:
        current_element: The current dictionary or list being searched.
        target_key: The key to search for.
        current_path: A list of keys/indices representing the path so far from the root.
        found_paths_list: A list to accumulate all found paths.
    """
    if isinstance(current_element, dict):
        for key, value in current_element.items():
            # Path to the current key being checked
            path_to_this_key = current_path + [key]
            if key == target_key:
                found_paths_list.append(path_to_this_key)  # Add path to list

            # If the value is a collection, recurse into it
            if isinstance(value, (dict, list)):
                _find_all_paths_recursive(value, target_key, path_to_this_key, found_paths_list)
    elif isinstance(current_element, list):
        for index, item in enumerate(current_element):
            # Path to the current item being checked (using its index)
            path_to_this_item = current_path + [index]
            # If the item in the list is a collection, recurse into it
            if isinstance(item, (dict, list)):
                _find_all_paths_recursive(item, target_key, path_to_this_item, found_paths_list)


def find_and_update_key_value(data_dict, target_key,
                              new_value=None, set_value_flag=False,
                              discriminator_key=None, debug=False):
    """
    Finds occurrences of a key in a nested dictionary. If duplicates are found,
    a discriminator_key can be used to select the specific instance to act upon
    by checking if the discriminator_key is part of the path to the target_key.
    Optionally updates its value.

    Args:
        data_dict (dict): The dictionary to search within.
        target_key (any): The key to find.
        new_value (any, optional): The new value if set_value_flag is True. Defaults to None.
        set_value_flag (bool, optional): If True, update the key's value. Defaults to False.
        discriminator_key (any, optional): A key that, if present in the path to an
                                           instance of target_key, will select that instance.
        debug (bool, optional): If True, print detailed debugging information. Defaults to False.

    Returns:
        tuple: (selected_path, value_or_old_value)
               - selected_path (list or None): Path to the (potentially discriminated) target_key.
                                             None if not found or discriminator fails.
               - value_or_old_value (any or None): Old value if updated, current value if found, else None.
    """
    if debug:
        print(
            f"[DEBUG] Searching for target_key='{target_key}' with discriminator_key='{discriminator_key}' (discriminator checks if it's IN the path)")

    all_paths = []
    _find_all_paths_recursive(data_dict, target_key, [], all_paths)

    if debug:
        print(f"[DEBUG] Found {len(all_paths)} raw path(s) for '{target_key}': {all_paths}")

    if not all_paths:
        if debug: print(f"[DEBUG] Target key '{target_key}' not found anywhere.")
        return None, None

    selected_path = None

    if len(all_paths) == 1:
        selected_path = all_paths[0]
        if debug: print(f"[DEBUG] Single instance of '{target_key}' found. Selected path: {selected_path}")
    else:  # Multiple paths found for target_key
        if debug: print(f"[DEBUG] Multiple instances ({len(all_paths)}) of '{target_key}' found.")
        if discriminator_key:
            if debug: print(
                f"[DEBUG] Attempting to discriminate: if '{discriminator_key}' is IN a path to '{target_key}'.")
            for path_candidate in all_paths:
                if debug: print(f"[DEBUG]  Checking candidate path: {path_candidate}")
                # Check if discriminator_key is one of the elements in the path_candidate
                if discriminator_key in path_candidate:
                    selected_path = path_candidate
                    if debug: print(
                        f"[DEBUG]   Discriminator '{discriminator_key}' found IN path. Selected path: {selected_path}")
                    break  # Use the first path that contains the discriminator
                else:
                    if debug: print(
                        f"[DEBUG]   Discriminator '{discriminator_key}' NOT found in path {path_candidate}.")

            if not selected_path:
                warning_msg = f"Warning: Found {len(all_paths)} instances of '{target_key}', but none of their paths contained '{discriminator_key}'."
                print(warning_msg)
                if debug: print(f"[DEBUG] {warning_msg}")
                return None, None
        else:
            warning_msg = f"Warning: Found {len(all_paths)} instances of '{target_key}'. No discriminator_key provided. Using the first instance found at: {all_paths[0]}"
            print(warning_msg)
            if debug: print(f"[DEBUG] {warning_msg}")
            selected_path = all_paths[0]

    if not selected_path:
        if debug: print(f"[DEBUG] No suitable path was selected for '{target_key}'.")
        return None, None

    if debug: print(f"[DEBUG] Final selected path for '{target_key}': {selected_path}")

    current_element_ref = data_dict
    # Navigate to the parent element
    for key_or_index_in_path in selected_path[:-1]:
        current_element_ref = current_element_ref[key_or_index_in_path]

    final_accessor = selected_path[-1]  # This is the target key or index in the parent

    if debug:
        print(
            f"[DEBUG] Parent element of '{target_key}' (at path {selected_path[:-1]}): type={type(current_element_ref)}")
        print(f"[DEBUG] Final accessor for '{target_key}': '{final_accessor}' (type: {type(final_accessor)})")

    # Ensure the final accessor is valid for the current_element_ref
    if isinstance(current_element_ref, dict) and final_accessor not in current_element_ref:
        if debug: print(f"[DEBUG] Error: Final accessor '{final_accessor}' not in parent dict.")
        return None, None
    if isinstance(current_element_ref, list) and \
            (not isinstance(final_accessor, int) or final_accessor < 0 or final_accessor >= len(current_element_ref)):
        if debug: print(
            f"[DEBUG] Error: Final accessor '{final_accessor}' is an invalid index for parent list of size {len(current_element_ref) if isinstance(current_element_ref, list) else 'N/A'}.")
        return None, None

    current_value = current_element_ref[final_accessor]
    if debug: print(f"[DEBUG] Current value of '{target_key}' at {selected_path}: {current_value}")

    if set_value_flag:
        old_value = current_value
        current_element_ref[final_accessor] = new_value  # This is the direct modification
        if debug: print(
            f"[DEBUG] Updated value of '{target_key}' at {selected_path} from '{old_value}' to '{new_value}'")
        return selected_path, old_value
    else:
        return selected_path, current_value


def update_dict(dict1, dict2):
    """
    Updates dict1 with key-value pairs from dict2,
    only if the key from dict2 does not already exist in dict1.

    Args:
      dict1: The dictionary to be updated (and potentially have items copied into).
      dict2: The dictionary whose items will be considered for copying.

    Returns:
      The updated dict1.
    """
    for key, value in dict2.items():
        if key not in dict1:
            dict1[key] = value
    return dict1
