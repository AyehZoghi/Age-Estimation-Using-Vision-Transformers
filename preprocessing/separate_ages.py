#Organize UTKFace images into age folders.

import argparse
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def get_age_from_filename(file_path: Path) -> str | None:
    age = file_path.name.split("_")[0]
    if age.isdigit():
        return age
    return None


def separate_ages(source_dir: Path, copy_files: bool = False) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source folder does not exist: {source_dir}")

    moved_count = 0
    skipped_count = 0

    for file_path in sorted(source_dir.iterdir()):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
            skipped_count += 1
            continue

        age = get_age_from_filename(file_path)
        if age is None:
            skipped_count += 1
            continue

        age_dir = source_dir / age
        age_dir.mkdir(parents=True, exist_ok=True)
        target_path = age_dir / file_path.name

        if target_path.exists():
            skipped_count += 1
            continue

        if copy_files:
            shutil.copy2(file_path, target_path)
        else:
            shutil.move(str(file_path), str(target_path))

        moved_count += 1

    action = "Copied" if copy_files else "Moved"
    print(f"{action} images: {moved_count}")
    print(f"Skipped files: {skipped_count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Organize UTKFace images into age folders.")
    parser.add_argument("--source_dir", type=Path, default=Path("UTKFace_aligned"))
    parser.add_argument("--copy", action="store_true", help="Copy files instead of moving them.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    separate_ages(source_dir=args.source_dir, copy_files=args.copy)
