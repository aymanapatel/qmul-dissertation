from pathlib import Path
import shutil

# Replace with the folder
ROOT = Path("/Users/aymanpatel/Desktop/Uni/Dissertation/2_Data/browser-use/outputs/dataset_v3.0/axe-core")

REQUIRED_PATTERNS = [
    "*.html",
    "*.png",
    "page*home.json",
    "*visual.json",
]

def has_required_files(folder: Path) -> bool:
    return all(any(folder.glob(pattern)) for pattern in REQUIRED_PATTERNS)

def main() -> None:
    kept = 0
    removed = 0

    for site_folder in ROOT.iterdir():
        if not site_folder.is_dir():
            continue

        if has_required_files(site_folder):
            kept += 1
            continue

        shutil.rmtree(site_folder)
        removed += 1
        print(f"removed: {site_folder.name}")

    print(f"kept: {kept}")
    print(f"removed: {removed}")

if __name__ == "__main__":
    main()