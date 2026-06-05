#!/usr/bin/env python3
import os
import sys
import json
import csv
import argparse
try:
    import torch
except ImportError:
    pass
import numpy as np
import soundfile as sf
import traceback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def main():
    parser = argparse.ArgumentParser(description="Evaluate models on the Matrix Transition dataset")
    parser.add_argument("--model", type=str, required=True, choices=["Qwen", "Flamingo", "MuMu", "Librosa", "DeepResonance"])
    parser.add_argument("--matrix_dir", type=str, default="matrix_dataset")
    parser.add_argument("--base_bpm", type=int, required=True, help="Base BPM to process (array job ID)")
    args = parser.parse_args()

    print(f"Evaluating {args.model} on Matrix Dataset for Base BPM {args.base_bpm}...")

    # Load model
    if args.model == "Qwen":
        from src.qwen_interface import setup_qwen, generate_beatmap_with_qwen
        setup_qwen()
        def get_qwen_bpm(path):
            resp = generate_beatmap_with_qwen(path, "Estimate BPM")
            nums = __import__('re').findall(r"\d+\.?\d*", str(resp))
            return float(nums[0]) if nums else 0.0
        infer_func = get_qwen_bpm
    elif args.model == "Flamingo":
        os.environ["HF_HOME"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Music-Flamingo", "checkpoints"))
        from src.music_flamingo_interface import setup_music_flamingo, generate_beatmap_with_flamingo
        setup_music_flamingo()
        def get_flamingo_bpm(path):
            resp = generate_beatmap_with_flamingo(path, "Estimate BPM")
            nums = __import__('re').findall(r"\d+\.?\d*", str(resp))
            return float(nums[0]) if nums else 0.0
        infer_func = get_flamingo_bpm
    elif args.model == "MuMu":
        from src.mumu_interface import setup_mumu, generate_beatmap_with_mumu
        setup_mumu()
        def get_mumu_bpm(path):
            resp = generate_beatmap_with_mumu(path, "Estimate BPM")
            nums = __import__('re').findall(r"\d+\.?\d*", str(resp))
            return float(nums[0]) if nums else 0.0
        infer_func = get_mumu_bpm
    elif args.model == "Librosa":
        import librosa
        def get_librosa_bpm(audio_path):
            y, sr = librosa.load(audio_path, sr=16000)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            return float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
        infer_func = get_librosa_bpm
    elif args.model == "DeepResonance":
        proj = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        ckpt = os.path.join(proj, "DeepResonance", "ckpt")
        sys.path.insert(0, os.path.join(proj, "DeepResonance", "code"))
        os.chdir(os.path.join(proj, "DeepResonance", "code"))
        from inference_deepresonance import DeepResonancePredict
        model_args = {
            "stage": 2, "mode": "test", "dataset": "musiccaps", "project_path": os.path.join(proj, "DeepResonance", "code"),
            "llm_path": os.path.join(ckpt, "pretrained_ckpt", "vicuna_ckpt", "7b_v0"), 
            "imagebind_path": os.path.join(ckpt, "pretrained_ckpt", "imagebind_ckpt", "huge"),
            "imagebind_version": "huge", "max_length": 512, "max_output_length": 512, "num_clip_tokens": 77, "gen_emb_dim": 768,
            "preencoding_dropout": 0.1, "num_preencoding_layers": 1, "lora_r": 32, "lora_alpha": 32, "lora_dropout": 0.1,
            "freeze_lm": False, "freeze_input_proj": False, "freeze_output_proj": False, "prompt": "", "prellmfusion": True,
            "prellmfusion_dropout": 0.1, "num_prellmfusion_layers": 1, "imagebind_embs_seq": True, "topp": 1.0, "temp": 0.1,
            "ckpt_path": os.path.join(ckpt, "DeepResonance_data_models", "ckpt", "deepresonance_beta_delta_ckpt", "delta_ckpt", "deepresonance", "7b_tiva_v0"),
        }
        dr_model = DeepResonancePredict(model_args)
        def get_dr_bpm(path):
            inputs = {
                "inputs": ["Estimate BPM"], "instructions": ["Estimate BPM"], "mm_names": [["audio"]],
                "mm_paths": [[os.path.basename(path)]], "mm_root_path": os.path.dirname(path), "outputs": [""]
            }
            resp = dr_model.predict(inputs, max_tgt_len=512, top_p=1.0, temperature=0.1, stops_id=[[835]])
            if isinstance(resp, list): resp = resp[0]
            nums = __import__('re').findall(r"\d+\.?\d*", str(resp))
            return float(nums[0]) if nums else 0.0
        infer_func = get_dr_bpm
    else:
        raise ValueError("Invalid model")

    song_name = f"base_bpm_{args.base_bpm}"
    audio_path = os.path.join(args.matrix_dir, f"{song_name}.wav")
    json_path = os.path.join(args.matrix_dir, f"{song_name}_groundtruth.json")

    if not os.path.exists(audio_path) or not os.path.exists(json_path):
        print(f"Error: Could not find audio or JSON for {song_name}")
        sys.exit(1)

    print(f"Loading {audio_path}...")
    try:
        y, sr = sf.read(audio_path)
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        sys.exit(1)

    with open(json_path, 'r') as f:
        ground_truth = json.load(f)

    out_csv = os.path.join(args.matrix_dir, f"{args.model}_{song_name}_rmse.csv")
    
    # We write continuously so we don't lose data on crash
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["song_name", "window_start", "window_end", "actual_bpm", "pred_bpm"])

        segments = ground_truth.get("segments", [])
        total_segments = len(segments)
        print(f"Processing {total_segments} segments...")
        
        for i, segment in enumerate(segments):
            start_s = segment["start"]
            end_s = segment["end"]
            actual_bpm = segment["bpm"]
            
            start_idx = int(start_s * sr)
            end_idx = int(end_s * sr)
            chunk = y[start_idx:end_idx]
            
            # Save chunk temporarily to disk since inference functions expect a file path
            tmp_path = f"/tmp/matrix_{args.model}_{song_name}_chunk.wav"
            sf.write(tmp_path, chunk, sr)
            
            try:
                pred_bpm = infer_func(tmp_path)
            except Exception as e:
                print(f"Error inferring segment {i}: {e}")
                pred_bpm = None
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                
                # Global PyTorch Memory Leak Protection
                import gc
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
            
            print(f"Segment {i}/{total_segments} ({start_s}s - {end_s}s) | Target: {actual_bpm} | Pred: {pred_bpm}")
            writer.writerow([song_name, start_s, end_s, actual_bpm, pred_bpm])
            f.flush()

    print(f"Finished evaluating {song_name}")

if __name__ == "__main__":
    main()
