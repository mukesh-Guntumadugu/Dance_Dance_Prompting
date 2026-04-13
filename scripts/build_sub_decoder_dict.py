import os
import sys
import sqlite3
import json
import traceback

# Add pattern_finding_approach to path to use its functions
sys.path.append(os.path.join(os.path.dirname(__file__), '../pattern_finding_approach'))

from pattern_finding import parse_ssc_sm, clean_and_split_measures, upscale_measure

def build_sub_decoder_dict(db_path, output_json_path):
    print(f"Connecting to database at {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all measure cluster assignments that are valid (not outlier -1)
    # We group by file_path and difficulty to avoid re-reading files unnecessarily
    cursor.execute("""
        SELECT file_path, difficulty, measure_idx, cluster_id
        FROM measure_cluster_assignments
        WHERE cluster_id != -1
        ORDER BY file_path, difficulty, measure_idx
    """)
    rows = cursor.fetchall()

    print(f"Found {len(rows)} valid measure assignments in the database.")

    # Structure to hold patterns: "cluster_ID": set(pattern_strings)
    cluster_dict = {}

    current_file = None
    current_difficulty = None
    current_chart_measures = None

    # Build a lookup table of actual local files to handle remote/local DB path mismatch
    print("Building local file index...")
    music_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'musicForBeatmap')
    local_file_lookup = {}
    for root, _, files in os.walk(music_dir):
        for f in files:
            if f.endswith('.ssc') or f.endswith('.sm'):
                local_file_lookup[f] = os.path.join(root, f)

    processed_count = 0

    for file_path, difficulty, measure_idx, cluster_id in rows:
        cluster_key = f"<|cluster_{cluster_id}|>"
        
        if cluster_key not in cluster_dict:
            cluster_dict[cluster_key] = set()

        # Load the file and difficulty if we haven't already
        if file_path != current_file or difficulty != current_difficulty:
            try:
                base_name = os.path.basename(file_path)
                if base_name in local_file_lookup:
                    full_file_path = local_file_lookup[base_name]
                else:
                    # Silently skip missing local files
                    current_chart_measures = None
                    continue

                charts, metadata = parse_ssc_sm(full_file_path)
                
                # Find the specific chart
                chart_notes_str = None
                if charts:
                    for chart in charts:
                        if chart['difficulty'] == difficulty:
                            chart_notes_str = chart['notes_string']
                            break

                if not chart_notes_str:
                    print(f"Could not find difficulty {difficulty} in {file_path}")
                    current_chart_measures = None
                else:
                    current_chart_measures = clean_and_split_measures(chart_notes_str)
                    if not current_chart_measures:
                        print(f"Clean and split returned empty for {file_path}")

            except Exception as e:
                print(f"Error parsing {file_path}: {e}")
                current_chart_measures = None

            current_file = file_path
            current_difficulty = difficulty

        if not current_chart_measures:
            continue

        if measure_idx >= len(current_chart_measures):
            # Sometimes parsing inconsistencies handle trailing measures differently
            print(f"Index out of bounds for {file_path} {difficulty}: {measure_idx} >= {len(current_chart_measures)}")
            continue

        raw_lines = current_chart_measures[measure_idx]
        try:
            # Upscale measure strictly to 192 rows using existing logic
            upscaled = upscale_measure(raw_lines, target_rows=192)
            
            # Convert list of lists to a single string delimiter format
            pattern_str = "\n".join("".join(row) for row in upscaled)
            
            # Deduplicate via set
            cluster_dict[cluster_key].add(pattern_str)
            
            processed_count += 1
            if processed_count % 5000 == 0:
                print(f"Processed {processed_count} measures...")

        except Exception as e:
            # In case the upscale measure errors out
            print(f"Error upscaling measure: {e}")

    print("Finished extracting patterns. Converting unique sets to lists...")
    
    # Convert sets to lists for JSON serialization
    serializable_dict = {k: list(v) for k, v in cluster_dict.items()}

    print(f"Saving dictionary to {output_json_path}...")
    with open(output_json_path, 'w') as f:
        json.dump(serializable_dict, f, indent=2)

    print(f"Success! Processed {processed_count} total measures into {len(serializable_dict.keys())} discrete clusters.")
    
    # Also save just the special tokens to a text file for easy copy/paste training integration
    tokens = sorted(list(serializable_dict.keys()))
    tokens_path = output_json_path.replace('.json', '_tokens.txt')
    with open(tokens_path, 'w') as f:
        for t in tokens:
            f.write(t + "\n")
    print(f"Saved token list separately to {tokens_path}")

if __name__ == "__main__":
    db_path = os.path.join(os.path.dirname(__file__), '../pattern_finding_approach/processed_files.db')
    output_json = os.path.join(os.path.dirname(__file__), 'cluster_to_patterns.json')
    
    build_sub_decoder_dict(db_path, output_json)
