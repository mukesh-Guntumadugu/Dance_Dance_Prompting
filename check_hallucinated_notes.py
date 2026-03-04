import csv
from pathlib import Path
from collections import Counter

BASE_DIR = Path("src/musicForBeatmap/Fraxtil's Arrow Arrangements")

def check_files():
    all_chars = Counter()
    files_checked = 0
    
    # Check all sorted CSV files for gemini
    for csv_path in BASE_DIR.rglob("*gemini*.csv"):
        if "sorted" not in csv_path.name:
            continue
        files_checked += 1
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    notes = row.get("notes", "")
                    for char in notes:
                        all_chars[char] += 1
        except Exception as e:
            pass
            
    print(f"\nFiles checked: {files_checked}")
    print("\nCharacter Counts in 'notes' column:")
    for char, count in all_chars.most_common():
        print(f"  {repr(char)}: {count}")

        
if __name__ == "__main__":
    check_files()
