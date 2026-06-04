#!/bin/bash
echo "Evaluating Librosa on WAV sweep..."
python3 evaluate_sweep.py --model Librosa --mode full_song --ext wav

echo "Evaluating Librosa on MP3 sweep..."
python3 evaluate_sweep.py --model Librosa --mode full_song --ext mp3

echo "Evaluating Librosa on OGG sweep..."
python3 evaluate_sweep.py --model Librosa --mode full_song --ext ogg

echo "Finished all Librosa sweep evaluations!"
