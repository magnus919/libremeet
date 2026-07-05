#!/usr/bin/env python3
"""
WhisperX Speaker Diarization Sidecar for LibreMeet.

Runs WhisperX with VAD, forced alignment, and pyannote speaker diarization
on a meeting audio file. Outputs JSON array of segments with speaker labels.

Usage:
    python3 whisperx_diarize.py --audio <path> --output <path> [--hf-token <token>] [--model large-v3]

Output format (JSON):
    [
        {"speaker_id": "SPEAKER_00", "start_sec": 0.5, "end_sec": 3.2, "text": "Hello everyone"},
        {"speaker_id": "SPEAKER_01", "start_sec": 3.5, "end_sec": 6.1, "text": "Hi there"}
    ]

Progress is reported on stdout with "PROGRESS:" prefix.
Errors go to stderr.
"""

import argparse
import json
import sys
import os
import traceback


def main():
    parser = argparse.ArgumentParser(
        description="WhisperX speaker diarization sidecar for LibreMeet"
    )
    parser.add_argument(
        "--audio", required=True, help="Path to the meeting audio file"
    )
    parser.add_argument(
        "--output", required=True, help="Path to write the diarization JSON output"
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace token for pyannote model access (optional but recommended)",
    )
    parser.add_argument(
        "--model",
        default="large-v3",
        help="WhisperX model name (default: large-v3)",
    )
    args = parser.parse_args()

    audio_path = args.audio
    output_path = args.output
    hf_token = args.hf_token
    model_name = args.model

    # Validate input
    if not os.path.exists(audio_path):
        print(f"ERROR: Audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    print("PROGRESS:Initializing WhisperX...", flush=True)

    try:
        import whisperx
        import torch
    except ImportError as e:
        print(
            f"ERROR: WhisperX is not installed. Install with: pip install whisperx",
            file=sys.stderr,
        )
        print(f"ERROR: Import details: {e}", file=sys.stderr)
        sys.exit(1)

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    print(f"PROGRESS:Using device: {device}, compute: {compute_type}", flush=True)

    try:
        # Step 1: Load model and transcribe with VAD
        print("PROGRESS:Loading WhisperX model...", flush=True)
        model = whisperx.load_model(
            model_name, device, compute_type=compute_type, language="en"
        )

        print("PROGRESS:Loading audio file...", flush=True)
        audio = whisperx.load_audio(audio_path)

        print("PROGRESS:Transcribing with voice activity detection...", flush=True)
        result = model.transcribe(audio, batch_size=16)

        detected_language = result.get("language", "en")
        print(
            f"PROGRESS:Detected language: {detected_language}, {len(result['segments'])} segments",
            flush=True,
        )

        if not result["segments"]:
            print("WARNING: No speech segments detected in audio", file=sys.stderr)
            # Write empty output
            with open(output_path, "w") as f:
                json.dump([], f, indent=2)
            print("PROGRESS:Done. 0 segments, 0 speakers.", flush=True)
            sys.exit(0)

        # Step 2: Align
        print("PROGRESS:Loading alignment model...", flush=True)
        model_a, metadata = whisperx.load_align_model(
            language_code=detected_language, device=device
        )

        print("PROGRESS:Aligning transcription with audio...", flush=True)
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )

        # Step 3: Diarize
        print("PROGRESS:Running speaker diarization...", flush=True)

        try:
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=hf_token, device=device
            )
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)
        except Exception as diar_err:
            error_msg = str(diar_err)
            if "pyannote" in error_msg.lower() or "401" in error_msg or "gated" in error_msg.lower():
                print(
                    "ERROR: pyannote model requires a HuggingFace token. "
                    "Please provide --hf-token with a valid HuggingFace access token "
                    "that has accepted the pyannote model terms.",
                    file=sys.stderr,
                )
                print(f"ERROR: Details: {diar_err}", file=sys.stderr)
                sys.exit(1)
            else:
                print(
                    f"WARNING: Speaker diarization failed: {diar_err}. "
                    "Falling back to single-speaker output.",
                    file=sys.stderr,
                )
                # Fallback: assign all segments to "SPEAKER_00"
                for seg in result["segments"]:
                    seg["speaker"] = "SPEAKER_00"

        # Step 4: Build output
        print("PROGRESS:Building output...", flush=True)
        output = []
        for seg in result.get("segments", []):
            speaker = seg.get("speaker", "SPEAKER_00")
            output.append(
                {
                    "speaker_id": speaker,
                    "start_sec": round(seg.get("start", 0), 2),
                    "end_sec": round(seg.get("end", 0), 2),
                    "text": (seg.get("text", "") or "").strip(),
                }
            )

        # Sort by start time
        output.sort(key=lambda s: s["start_sec"])

        # Write output
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        unique_speakers = len(set(s["speaker_id"] for s in output))
        print(
            f"PROGRESS:Done. {len(output)} segments, {unique_speakers} speakers.",
            flush=True,
        )

    except Exception as e:
        print(f"ERROR: Diarization failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
