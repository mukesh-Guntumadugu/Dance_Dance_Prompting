import sqlite3
import os
import torch
import librosa
import numpy as np
import tempfile
from peft import PeftModel
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

MODEL_ID = "/data/mg546924/models/Qwen2-Audio-7B-Instruct"
PEFT_DIR = "/data/mg546924/models/qwen2-audio-hierarchical-director"
DB_PATH = "/data/mg546924/llm_beatmap_generator/pattern_finding_approach/processed_files.db"
SONG_PATH = "/data/mg546924/llm_beatmap_generator/src/musicForBeatmap/Fraxtil's Arrow Arrangements/Bad Ketchup/Bad Ketchup.ogg"

def get_actual_clusters(db_path, song_like, diff, measures_per_chunk=4):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        SELECT file_path FROM audio_features WHERE file_path LIKE ? LIMIT 1
    """, (f"%{song_like}%",))
    res = c.fetchone()
    if not res: return []
    file_path = res[0]
    
    c.execute("""
        SELECT af.start_time, af.end_time, mca.cluster_id
        FROM audio_features af
        JOIN measure_cluster_assignments mca ON af.file_path=mca.file_path AND af.difficulty=mca.difficulty AND af.measure_idx=mca.measure_idx AND af.run_id=mca.run_id AND af.chop_length=mca.chop_length
        WHERE af.file_path=? AND af.difficulty=? AND mca.cluster_id != -1 AND af.chop_length=1
        ORDER BY af.measure_idx ASC
    """, (file_path, diff))
    measures = c.fetchall()
    conn.close()
    
    chunks = []
    for i in range(0, len(measures), measures_per_chunk):
        chunk = measures[i : i + measures_per_chunk]
        if len(chunk) < measures_per_chunk: continue
        win_start = chunk[0][0]
        win_end = chunk[-1][1]
        clusters = [m[2] for m in chunk]
        chunks.append({"win_start": win_start, "win_end": win_end, "clusters": clusters})
    return chunks

TOKENS_TXT = "/data/mg546924/llm_beatmap_generator/scripts/cluster_to_patterns_tokens.txt"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
print(f"Adding tokens from {TOKENS_TXT}...", flush=True)
with open(TOKENS_TXT, "r") as f:
    cluster_tokens = [line.strip() for line in f if line.strip()]
    cluster_tokens = cluster_tokens[:-1] # Drop last token to match training vocab size
processor.tokenizer.add_tokens(cluster_tokens, special_tokens=True)

print("Loading model...")
model = Qwen2AudioForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
model.resize_token_embeddings(len(processor.tokenizer))
model = PeftModel.from_pretrained(model, PEFT_DIR)
model.cuda().eval()

y, sr = librosa.load(SONG_PATH, sr=16000)

for diff in ["Beginner", "Easy", "Medium", "Hard", "Challenge"]:
    print(f"\nEvaluating Difficulty: {diff}")
    chunks = get_actual_clusters(DB_PATH, "Bad Ketchup", diff)
    print(f"Found {len(chunks)} chunks.")
    if len(chunks) == 0: continue
    
    correct_chunks = 0
    total_chunks = 0
    
    print(f"{'Chunk Time':<15} | {'Actual Clusters':<30} | {'Predicted Clusters':<30} | {'Exact?'}")
    print("-" * 100)

    for chunk in chunks:
        target_c = chunk['clusters'][-1]
        actual_str = "<|cluster_empty|>" if target_c == -1 else f"<|cluster_{target_c}|>"
        
        prev_clusters = chunk['clusters'][:-1]
        prev_clusters_str = " ".join([("<|cluster_empty|>" if c == -1 else f"<|cluster_{c}|>") for c in prev_clusters])

        start = int(chunk['win_start'] * 16000)
        end = int(chunk['win_end'] * 16000)
        audio_segment = y[start:end]
        
        onset_frames = librosa.onset.onset_detect(y=audio_segment, sr=16000)
        onset_times = librosa.frames_to_time(onset_frames, sr=16000)
        onsets_str = ", ".join([f"{t:.2f}s" for t in onset_times]) if len(onset_times) > 0 else "None"
        
        prompt = (
            "You are a rhythm game beatmap pattern generator. "
            f"Listen to this 4-measure audio segment. "
            f"The difficulty is {diff}. "
            f"Note onsets relative to the start of this chunk are: {onsets_str}. "
            f"The previous 3 measure clusters are: {prev_clusters_str}. "
            "Predict the rhythmic pattern cluster token for the FINAL measure."
        )
        
        text = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\nAudio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        inputs = processor(text=text, audio=[audio_segment], sampling_rate=16000, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=40)
        
        response = processor.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)
        
        import re
        pred_clusters = re.findall(r'<\|cluster_(\d+)\|>', response)
        pred_str = f"<|cluster_{pred_clusters[0]}|>" if pred_clusters else response.strip()
        if '<|cluster_empty|>' in response: pred_str = '<|cluster_empty|>'
        
        correct = (actual_str == pred_str)
        if correct:
            correct_chunks += 1
        
        total_chunks += 1
        
        print(f"[{chunk['win_start']:04.1f}s-{chunk['win_end']:04.1f}s] | {actual_str:<30} | {pred_str:<30} | {correct}")
    
    print(f"\nFinal Exact Chunk Matches for {diff}: {correct_chunks}/{total_chunks}\n")
