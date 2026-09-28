import os, sys, gc, sqlite3, tempfile
import torch, librosa, re
import soundfile as sf

os.environ["LD_LIBRARY_PATH"] = f"/data/mg546924/conda_envs/deepresonance_env/lib/python3.10/site-packages/nvidia/cusparse/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

PROJ = "/data/mg546924/llm_beatmap_generator"
DB_PATH = os.path.join(PROJ, "pattern_finding_approach", "processed_files.db")
SONG_PATH = os.path.join(PROJ, "src", "musicForBeatmap", "Fraxtil's Arrow Arrangements", "Bad Ketchup", "Bad Ketchup.ogg")
CKPT = os.path.join(PROJ, "DeepResonance", "ckpt")
PEFT_DIR = "/data/mg546924/models/deepresonance-hierarchical-director/checkpoint_final.pt"
TOKENS_TXT = "/data/mg546924/llm_beatmap_generator/scripts/cluster_to_patterns_tokens.txt"

sys.path.insert(0, os.path.join(PROJ, "DeepResonance", "code"))
os.chdir(os.path.join(PROJ, "DeepResonance", "code"))
from inference_deepresonance import DeepResonancePredict

def get_actual_clusters(db_path, song_like, diff, measures_per_chunk=4):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT file_path FROM audio_features WHERE file_path LIKE ? LIMIT 1", (f"%{song_like}%",))
    res = c.fetchone()
    if not res: return []
    file_path = res[0]
    
    c.execute("""
        SELECT af.start_time, af.end_time, mca.cluster_id
        FROM audio_features af
        JOIN measure_cluster_assignments mca 
          ON af.file_path=mca.file_path AND af.difficulty=mca.difficulty 
          AND af.measure_idx=mca.measure_idx AND af.run_id=mca.run_id AND af.chop_length=mca.chop_length
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

args = {
    "stage": 2, "mode": "test", "dataset": "musiccaps",
    "project_path": os.path.join(PROJ, "DeepResonance", "code"),
    "llm_path": os.path.join(CKPT, "pretrained_ckpt", "vicuna-7b-v1.1"),
    "imagebind_path": os.path.join(CKPT, "pretrained_ckpt", "imagebind_ckpt", "huge"),
    "imagebind_version": "huge",
    "max_length": 512, "max_output_length": 512,
    "num_clip_tokens": 77, "gen_emb_dim": 768,
    "preencoding_dropout": 0.1, "num_preencoding_layers": 1,
    "lora_r": 32, "lora_alpha": 32, "lora_dropout": 0.1,
    "freeze_lm": False, "freeze_input_proj": False, "freeze_output_proj": False,
    "prompt": "", "prellmfusion": False, "prellmfusion_dropout": 0.1,
    "num_prellmfusion_layers": 4, "imagebind_embs_seq": False, "topp": 1.0, "temp": 0.001,
    "ckpt_path": os.path.join(CKPT, "DeepResonance_data_models", "ckpt", "deepresonance_beta_delta_ckpt", "delta_ckpt", "deepresonance", "7b_tiva_v0"),
}

print("Loading DeepResonance model...", flush=True)
model = DeepResonancePredict(args)

print(f"Adding tokens from {TOKENS_TXT}...", flush=True)
with open(TOKENS_TXT, "r") as f:
    cluster_tokens = [line.strip() for line in f if line.strip()]
model.model.llama_tokenizer.add_special_tokens({"additional_special_tokens": cluster_tokens})
model.model.llama_model.resize_token_embeddings(len(model.model.llama_tokenizer))

print("Loading fine-tuned checkpoint...", flush=True)
delta_ckpt = torch.load(PEFT_DIR, map_location="cpu")

embed_key = 'llama_model.base_model.model.model.embed_tokens.weight'
lm_head_key = 'llama_model.base_model.model.lm_head.weight'
model_sd = model.model.state_dict()

for key in [embed_key, lm_head_key]:
    if key in delta_ckpt:
        ckpt_w = delta_ckpt[key]
        model_w = model_sd[key]
        if ckpt_w.shape[0] < model_w.shape[0]:
            print(f"Padding {key} from {ckpt_w.shape[0]} to {model_w.shape[0]}", flush=True)
            pad_size = model_w.shape[0] - ckpt_w.shape[0]
            padded_w = torch.cat([ckpt_w, model_w[-pad_size:].to(ckpt_w.device)], dim=0)
            delta_ckpt[key] = padded_w
        elif ckpt_w.shape[0] > model_w.shape[0]:
            print(f"Slicing {key} from {ckpt_w.shape[0]} to {model_w.shape[0]}", flush=True)
            delta_ckpt[key] = ckpt_w[:model_w.shape[0]]

model.model.load_state_dict(delta_ckpt, strict=False)
model.model.cuda().bfloat16().eval()

print("Loading audio...", flush=True)
y, sr = librosa.load(SONG_PATH, sr=16000)

# Apply monkey-patch ONCE globally
orig_decode = model.model.llama_tokenizer.batch_decode
def custom_decode(*args, **kwargs):
    kwargs["skip_special_tokens"] = False
    return orig_decode(*args, **kwargs)
model.model.llama_tokenizer.batch_decode = custom_decode

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
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio_segment, 16000)
            tmp_path = tmp.name
            
        prompt = (
            "You are a rhythm game beatmap pattern generator. "
            f"Listen to this 4-measure audio segment. "
            f"The difficulty is {diff}. "
            f"Note onsets relative to the start of this chunk are: {onsets_str}. "
            f"The previous 3 measure clusters are: {prev_clusters_str}. "
            "Predict the rhythmic pattern cluster token for the FINAL measure."
        )
        
        inputs = {
            "inputs": ["<Audio>"],
            "instructions": [prompt],
            "mm_names": [["audio"]],
            "mm_paths": [[os.path.basename(tmp_path)]],
            "mm_root_path": os.path.dirname(tmp_path),
            "outputs": [""],
        }
        
        resp = model.predict(inputs, max_tgt_len=200, top_p=1.0, temperature=0.001, stops_id=[[835]])
        if isinstance(resp, list): resp = resp[0]
        
        pred_clusters = re.findall(r'<\|cluster_(\d+)\|>', resp or "")
        pred_str = f"<|cluster_{pred_clusters[0]}|>" if pred_clusters else (resp or "").strip()
        if '<|cluster_empty|>' in (resp or ""): pred_str = '<|cluster_empty|>'
        
        correct = (actual_str == pred_str)
        if correct: correct_chunks += 1
        total_chunks += 1
        
        print(f"[{chunk['win_start']:04.1f}s-{chunk['win_end']:04.1f}s] | {actual_str:<30} | {pred_str:<30} | {correct}", flush=True)
        os.remove(tmp_path)
    
    print(f"\nFinal Exact Chunk Matches for {diff}: {correct_chunks}/{total_chunks}\n")
