#Remove known bad images from age folders.

import argparse
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def remove_bad_images(base_dir: Path, bad_dir_name: str = "bad_photos", dry_run: bool = False) -> None:
    if not base_dir.exists():
        raise FileNotFoundError(f"Base folder does not exist: {base_dir}")

    bad_photos_dir = base_dir / bad_dir_name
    if not bad_photos_dir.exists():
        raise FileNotFoundError(f"Bad photos folder does not exist: {bad_photos_dir}")

    # rglob allows the bad_photos folder to contain category subfolders
    bad_filenames = {
        file_path.name
        for file_path in bad_photos_dir.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
    }

    removed_count = 0

    for age_dir in sorted(base_dir.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 10**9):
        if not age_dir.is_dir() or not age_dir.name.isdigit():
            continue

        for file_path in age_dir.iterdir():
            if not file_path.is_file():
                continue

            if file_path.name in bad_filenames:
                print(f"Removing: {file_path}")
                if not dry_run:
                    file_path.unlink()
                removed_count += 1

    print(f"Bad image filenames listed: {len(bad_filenames)}")
    print(f"Matched and removed images: {removed_count}")
    if dry_run:
        print("Dry run only. No files were deleted.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove known bad images from age folders.")
    parser.add_argument("--base_dir", type=Path, default=Path("UTKFace_aligned"))
    parser.add_argument("--bad_dir_name", type=str, default="bad_photos")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    remove_bad_images(
        base_dir=args.base_dir,
        bad_dir_name=args.bad_dir_name,
        dry_run=args.dry_run,
    )
