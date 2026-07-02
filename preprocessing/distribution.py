#Save and plot the age distribution of an age-organized UTKFace folder.

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def count_age_distribution(base_dir: Path) -> dict[int, int]:
    if not base_dir.exists():
        raise FileNotFoundError(f"Base folder does not exist: {base_dir}")

    age_distribution: dict[int, int] = defaultdict(int)

    for age_dir in base_dir.iterdir():
        if not age_dir.is_dir() or not age_dir.name.isdigit():
            continue

        age = int(age_dir.name)
        count = sum(
            1
            for file_path in age_dir.iterdir()
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
        )
        age_distribution[age] = count

    return dict(sorted(age_distribution.items()))


def save_distribution_csv(age_distribution: dict[int, int], csv_path: Path) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Age", "Total Number"])
        for age, count in age_distribution.items():
            writer.writerow([age, count])


def save_distribution_plot(age_distribution: dict[int, int], plot_path: Path, show_plot: bool = False) -> None:
    ages = list(age_distribution.keys())
    counts = list(age_distribution.values())

    plt.figure(figsize=(16, 8))
    plt.bar(ages, counts)
    plt.xlabel("Age")
    plt.ylabel("Number of Images")
    plt.title("Age Distribution in UTKFace Dataset")
    plt.xticks(ages, rotation=90)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)

    if show_plot:
        plt.show()

    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save and plot the age distribution.")
    parser.add_argument("--base_dir", type=Path, default=Path("UTKFace_aligned"))
    parser.add_argument("--csv_path", type=Path, default=Path("data.csv"))
    parser.add_argument("--plot_path", type=Path, default=Path("age_distribution.png"))
    parser.add_argument("--show_plot", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    distribution = count_age_distribution(args.base_dir)

    for age, count in distribution.items():
        print(f"Age {age}: {count} images")

    save_distribution_csv(distribution, args.csv_path)
    save_distribution_plot(distribution, args.plot_path, args.show_plot)

    print(f"Saved CSV: {args.csv_path}")
    print(f"Saved plot: {args.plot_path}")
