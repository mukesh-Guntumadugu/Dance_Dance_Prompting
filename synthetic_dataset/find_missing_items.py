import pandas as pd

# Load the full expected sets from complete files
try:
    df_full = pd.read_csv("Flamingo_full_song_wav_rmse.csv")
    full_song_expected = set(zip(df_full['song_name'], df_full['window_start'], df_full['window_end']))
except Exception as e:
    full_song_expected = set()

try:
    df_chunk = pd.read_csv("Flamingo_stateless_chunk_mp3_rmse.csv")
    chunk_expected = set(zip(df_chunk['song_name'], df_chunk['window_start'], df_chunk['window_end']))
except Exception as e:
    chunk_expected = set()

def find_missing(target_file, expected_set):
    try:
        df = pd.read_csv(target_file)
        actual = set(zip(df['song_name'], df['window_start'], df['window_end']))
        missing = expected_set - actual
        # Group missing by song
        missing_songs = {}
        for song, start, end in missing:
            if song not in missing_songs:
                missing_songs[song] = []
            missing_songs[song].append((start, end))
        return missing_songs
    except Exception as e:
        return {}

missing_dr_full = find_missing("DeepResonance_full_song_wav_rmse.csv", full_song_expected)
missing_dr_chunk = find_missing("DeepResonance_stateless_chunk_mp3_rmse.csv", chunk_expected)

print("--- DeepResonance_full_song_wav_rmse.csv ---")
if missing_dr_full:
    print(f"Missing completely or partially for {len(missing_dr_full)} songs.")
    completely_missing = []
    partially_missing = []
    for song, windows in missing_dr_full.items():
        if len(windows) > 20: 
            completely_missing.append(song)
        else:
            partially_missing.append(f"{song} ({len(windows)} windows missing)")
    print("Completely missing songs:")
    print(", ".join(sorted(completely_missing)))
    if partially_missing:
        print("Partially missing songs:")
        print(", ".join(sorted(partially_missing)))
else:
    print("None")

print("\n--- DeepResonance_stateless_chunk_mp3_rmse.csv ---")
if missing_dr_chunk:
    print(f"Missing completely or partially for {len(missing_dr_chunk)} songs.")
    completely_missing = []
    partially_missing = []
    for song, windows in missing_dr_chunk.items():
        if len(windows) > 8: 
            completely_missing.append(song)
        else:
            partially_missing.append(f"{song} ({len(windows)} windows missing)")
    print("Completely missing songs:")
    print(", ".join(sorted(completely_missing)))
    if partially_missing:
        print("Partially missing songs:")
        print(", ".join(sorted(partially_missing)))
else:
    print("None")
