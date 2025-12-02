#!/usr/bin/env python3
import os
import shutil

SAVE_FILES = [
    "checkpoint.pt",
    "roster.json",
    "win_rate.png",
    "results.csv"
]

AGENTS_DIR = "agents"

def safe_delete_file(path):
    if os.path.exists(path):
        print(f"Deleting file: {path}")
        os.remove(path)
    else:
        print(f"[skip] File does not exist: {path}")

def safe_delete_dir(path):
    if os.path.exists(path):
        print(f"Deleting directory and all contents: {path}")
        shutil.rmtree(path)
    else:
        print(f"[skip] Directory does not exist: {path}")

def main():
    print("WARNING: This will delete training save files:")
    confirm = input("'[Y/N]' to proceed: ").strip()

    if confirm != "Y":
        print("Aborted. No files were deleted.")
        return

    # delete checkpoint + roster
    for f in SAVE_FILES:
        safe_delete_file(f)

    # delete agents directory
    safe_delete_dir(AGENTS_DIR)

    print("\nCleanup complete.")

if __name__ == "__main__":
    main()
