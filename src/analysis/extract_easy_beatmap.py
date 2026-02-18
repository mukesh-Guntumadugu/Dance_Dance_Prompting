import re
import os

def extract_easy_beatmap(ssc_path, output_path):
    """
    Extracts the 'Easy' difficulty beatmap from an .ssc file.
    """
    with open(ssc_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find the 'Easy' chart block
    # We look for #DIFFICULTY:Easy; and then capture the #NOTES: section until the next ;
    
    # This regex looks for the start of a note data section, checks if it contains Difficulty:Easy, 
    # and then captures the Notes section.
    # Note: .ssc files structure can be tricky. We'll iterate through #NOTEDATA sections.
    
    note_datas = content.split('#NOTEDATA:;')
    
    target_notes = None
    
    for section in note_datas:
        if 'DIFFICULTY:Easy;' in section or 'DIFFICULTY:Easy\n' in section:
            # Found the Easy section
            # Now find #NOTES:
            match = re.search(r'#NOTES:\s*([^;]+);', section, re.DOTALL)
            if match:
                target_notes = match.group(1).strip()
                break
    
    if target_notes:
        # Post-process: remove metadata lines if any, keep only note data lines (4 chars + separators)
        cleaned_lines = []
        for line in target_notes.split('\n'):
            line = line.strip()
            if not line: continue
            if line.startswith('//'): continue # Comments
            
            # Keep measures separators or note lines (4 chars usually)
            if ',' in line:
                cleaned_lines.append(',')
            elif len(line) == 4 and all(c in '01234M' for c in line):
                 cleaned_lines.append(line)
            # Sometimes lines have comments at the end
            elif len(line) >= 4 and all(c in '01234M' for c in line[:4]):
                cleaned_lines.append(line[:4])

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(cleaned_lines))
        print(f"Successfully extracted Easy beatmap to {output_path}")
        print(f"Extracted {len(cleaned_lines)} lines (including separators)")
    else:
        print("Could not find Easy difficulty chart in the provided .ssc file.")

if __name__ == "__main__":
    SSC_FILE = "src/musicForBeatmap/MechaTribe Assault/Mecha-Tribe Assault.ssc"
    OUTPUT_FILE = "MechaTribe_original_easy.ssc.text"
    
    # Use absolute paths if needed, or relative to CWD
    base_dir = os.getcwd()
    ssc_full = os.path.join(base_dir, SSC_FILE)
    out_full = os.path.join(base_dir, OUTPUT_FILE)
    
    extract_easy_beatmap(ssc_full, out_full)
