#!/bin/bash

OUT_DIR="synthetic_data"
mkdir -p "$OUT_DIR"

CSV_FILE="synthetic_dataset_summary.csv"
echo "song_name,duration,category,bpm_sequence" > "$CSV_FILE"

echo "Building V2 Synthetic Dataset (50 Songs)..."

generate_songs() {
    local start_id=$1
    local end_id=$2
    local interval=$3
    local description=$4
    local folder_name=$5

    echo "--- Generating Songs $start_id to $end_id ($description) ---"
    
    # Create the sub-folder for this category
    CATEGORY_DIR="$OUT_DIR/$folder_name"
    mkdir -p "$CATEGORY_DIR"
    
    for i in $(seq $start_id $end_id); do
        SONG_NAME="song_$i"
        SONG_DIR="$CATEGORY_DIR/$SONG_NAME"
        mkdir -p "$SONG_DIR"
        
        python3 generate_synthetic_beats.py \
            --song_name "$SONG_NAME" \
            --output_dir "$SONG_DIR" \
            --interval $interval \
            --csv_summary "$CSV_FILE" \
            --category "$description"
            
        sleep 0.1
    done
}

# 1. 10 songs constant BPM
generate_songs 1 10 0 "Constant BPM" "constant_bpm"

# 2. 10 songs changing every 60s
generate_songs 11 20 60 "Changing every 60s" "changing_every_60s"

# 3. 10 songs changing every 40s
generate_songs 21 30 40 "Changing every 40s" "changing_every_40s"

# 4. 10 songs changing every 50s
generate_songs 31 40 50 "Changing every 50s" "changing_every_50s"

# 5. 10 songs changing randomly (>5s intervals)
generate_songs 41 50 -1 "Changing randomly" "changing_randomly"

echo "All 50 songs generated successfully!"
