import shutil
import re
from pathlib import Path


def move_folders_with_shift(base_dir: Path, target_dir: Path, shift: int, dry_run: bool = True) -> None:
    """
    Move folders from base_dir to target_dir, shifting the run number by a given amount.

    Folders are expected to follow the pattern: <big_number>_<run_number>
    The run_number will be increased by 'shift' in the target directory.

    Parameters
    ----------
    base_dir : Path
        Source directory containing folders with pattern XXXXXX_Y
    target_dir : Path
        Target directory where folders will be moved
    shift : int
        Amount to add to the run number (the number after '_')
    dry_run : bool
        If True, only print what would be done without actually moving files

    Examples
    --------
    >>> move_folders_with_shift(Path('/source'), Path('/target'), shift=10, dry_run=False)
    # Moves /source/1804013_0 to /target/1804013_10
    # Moves /source/1804013_1 to /target/1804013_11
    """
    base_dir = Path(base_dir)
    target_dir = Path(target_dir)

    if not base_dir.exists():
        raise ValueError(f"Base directory does not exist: {base_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(r'^(\d+)_(\d+)$')

    folders_to_move = []

    for item in base_dir.iterdir():
        if not item.is_dir():
            continue

        match = pattern.match(item.name)
        if not match:
            print(f"Skipping '{item.name}' - doesn't match pattern")
            continue

        prefix = match.group(1)
        run_number = int(match.group(2))
        new_run_number = run_number + shift
        new_name = f"{prefix}_{new_run_number}"

        target_path = target_dir / new_name

        if target_path.exists():
            print(f"Warning: Target already exists, skipping: {target_path}")
            continue

        folders_to_move.append((item, target_path, run_number, new_run_number))

    if not folders_to_move:
        print("No folders found matching the pattern.")
        return

    print(f"\n{'='*70}")
    print(f"Moving {len(folders_to_move)} folder(s) with shift={shift}")
    print(f"From: {base_dir}")
    print(f"To:   {target_dir}")
    print(f"{'='*70}\n")

    for source, target, old_num, new_num in folders_to_move:
        if dry_run:
            print(
                f"[DRY RUN] Would move: {source.name} -> {target.name} (run {old_num} -> {new_num})")
        else:
            print(
                f"Moving: {source.name} -> {target.name} (run {old_num} -> {new_num})")
            shutil.move(str(source), str(target))

    if dry_run:
        print(
            f"\n[DRY RUN] No files were actually moved. Run with dry_run=False to perform the move.")
    else:
        print(f"\nSuccessfully moved {len(folders_to_move)} folder(s).")


if __name__ == "__main__":
    base_dir = Path(
        "/home/zimmerro/tmp/dinov2/CUB2011/Masterarbeit_Experiments/MAS7-grounding-l1_fv-rpl_2")
    target_dir = Path(
        "/home/zimmerro/tmp/dinov2/CUB2011/Masterarbeit_Experiments/MAS7-grounding-l1_fv-rpl")
    shift = 54

    move_folders_with_shift(base_dir, target_dir, shift, dry_run=False)
