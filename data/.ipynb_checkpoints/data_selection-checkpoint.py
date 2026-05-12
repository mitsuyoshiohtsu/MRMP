from pathlib import Path
import shutil

# === SETTINGS ===
LABEL_FILES = [
    "clean_noisy_labels_train.txt",
    "clean_noisy_labels_val.txt",
    "clean_noisy_labels_test.txt",
]
SOURCE_DIR = Path("../../clothing1M")
OUTPUT_DIR = Path("../../mini_clothing1M")

# === Step 1: Collect target paths from label files ===
# Key format in label files: images/0/00/xxxxxxx.jpg
target_paths = set()
for label_file in LABEL_FILES:
    path = OUTPUT_DIR / label_file
    if not path.exists():
        print(f"  [!] Label file not found, skipping: {path}")
        continue
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                target_paths.add(parts[0])  # e.g. images/3/01/78484370,182039301.jpg

print(f"[✓] Targeting {len(target_paths)} unique file paths.")

# === Step 2: Copy matching files ===
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

total_copied  = 0
total_missing = 0
for rel_path in sorted(target_paths):
    src = SOURCE_DIR / rel_path
    dst = OUTPUT_DIR / rel_path
    if not src.exists():
        print(f"  [!] Not found, skipping: {src}")
        total_missing += 1
        continue
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    total_copied += 1

print(f"[✓] Done. {total_copied} images copied to: {OUTPUT_DIR}")
if total_missing:
    print(f"  [!] {total_missing} files were missing from source.")
