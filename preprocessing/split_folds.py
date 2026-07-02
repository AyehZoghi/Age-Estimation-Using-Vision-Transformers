#Create fixed K-fold train/test folders from age-organized UTKFace images.

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def get_age_dirs(source_dir: Path, min_age: int, max_age: int) -> list[Path]:
    age_dirs = []

    for age_dir in source_dir.iterdir():
        if not age_dir.is_dir() or not age_dir.name.isdigit():
            continue

        age = int(age_dir.name)
        if min_age <= age <= max_age:
            age_dirs.append(age_dir)

    return sorted(age_dirs, key=lambda path: int(path.name))


def create_kfolds(
    source_dir: Path,
    folds_dir: Path,
    k: int = 5,
    seed: int = 42,
    min_age: int = 1,
    max_age: int = 100,
) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source folder does not exist: {source_dir}")

    if k < 2:
        raise ValueError("k must be at least 2.")

    rng = random.Random(seed)
    age_dirs = get_age_dirs(source_dir, min_age=min_age, max_age=max_age)

    if not age_dirs:
        raise ValueError("No age folders were found in the selected age range.")

    if folds_dir.exists():
        shutil.rmtree(folds_dir)
    folds_dir.mkdir(parents=True, exist_ok=True)

    age_to_folds: dict[str, list[list[Path]]] = {}

    for age_dir in age_dirs:
        images = [
            file_path
            for file_path in age_dir.iterdir()
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        images = sorted(images)
        rng.shuffle(images)

        split_images = [images[i::k] for i in range(k)]
        age_to_folds[age_dir.name] = split_images

    for fold_index in range(k):
        print(f"Creating fold {fold_index}")

        fold_dir = folds_dir / f"fold_{fold_index}"
        train_dir = fold_dir / "train"
        test_dir = fold_dir / "test"
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        for age, split_images in age_to_folds.items():
            train_age_dir = train_dir / age
            test_age_dir = test_dir / age
            train_age_dir.mkdir(parents=True, exist_ok=True)
            test_age_dir.mkdir(parents=True, exist_ok=True)

            test_images = split_images[fold_index]
            train_images = [
                image
                for i, fold_images in enumerate(split_images)
                if i != fold_index
                for image in fold_images
            ]

            for image_path in train_images:
                shutil.copy2(image_path, train_age_dir / image_path.name)

            for image_path in test_images:
                shutil.copy2(image_path, test_age_dir / image_path.name)

    print(f"Created {k} folds in: {folds_dir}")
    print(f"Source images were read from: {source_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create fixed K-fold train/test folders.")
    parser.add_argument("--source_dir", type=Path, default=Path("UTKFace_aligned"))
    parser.add_argument("--folds_dir", type=Path, default=Path("UTKFace_folds"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_age", type=int, default=1)
    parser.add_argument("--max_age", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_kfolds(
        source_dir=args.source_dir,
        folds_dir=args.folds_dir,
        k=args.k,
        seed=args.seed,
        min_age=args.min_age,
        max_age=args.max_age,
    )
