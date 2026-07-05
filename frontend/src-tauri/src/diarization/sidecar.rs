// One-shot sidecar process management for WhisperX diarization.
// Unlike llama-helper (persistent process with JSON-lines protocol),
// this spawns a Python script once per diarization request, monitors
// progress, and reads the output JSON file.

use std::path::{Path, PathBuf};
use std::process::Stdio;

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

/// A single diarization segment from WhisperX output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiarizationSegment {
    pub speaker_id: String,
    pub start_sec: f64,
    pub end_sec: f64,
    pub text: String,
}

/// Resolve the path to the WhisperX sidecar Python script.
///
/// Priority order:
/// 1. `WHISPERX_SIDECAR_PATH` environment variable
/// 2. `<exe_dir>/../libremeet-sidecars/whisperx_diarize.py` (bundled)
/// 3. Workspace root `libremeet-sidecars/whisperx_diarize.py` (dev)
fn resolve_sidecar_script() -> Result<PathBuf> {
    // 1. Environment variable (dev override)
    if let Ok(env_path) = std::env::var("WHISPERX_SIDECAR_PATH") {
        let path = PathBuf::from(&env_path);
        if path.exists() {
            log::info!(
                "Using WhisperX sidecar from WHISPERX_SIDECAR_PATH: {}",
                path.display()
            );
            return Ok(path);
        }
        log::warn!(
            "WHISPERX_SIDECAR_PATH set but file not found: {}",
            path.display()
        );
    }

    // 2. Relative to executable (bundled app)
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            // In a Tauri bundle, resources are next to the binary
            let bundled = exe_dir
                .parent() // up from MacOS/Contents to app root
                .map(|p| {
                    p.join("Resources")
                        .join("libremeet-sidecars")
                        .join("whisperx_diarize.py")
                });
            if let Some(ref bundled_path) = bundled {
                if bundled_path.exists() {
                    log::info!("Using bundled sidecar: {}", bundled_path.display());
                    return Ok(bundled_path.clone());
                }
            }

            // Direct next-to-exe (flat layout)
            let flat = exe_dir
                .join("libremeet-sidecars")
                .join("whisperx_diarize.py");
            if flat.exists() {
                log::info!("Using flat sidecar: {}", flat.display());
                return Ok(flat);
            }
        }
    }

    // 3. Workspace root (dev mode)
    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        let workspace_root = PathBuf::from(&manifest_dir)
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from(&manifest_dir));
        let dev_path = workspace_root.join("libremeet-sidecars").join("whisperx_diarize.py");
        if dev_path.exists() {
            log::info!("Using dev sidecar: {}", dev_path.display());
            return Ok(dev_path);
        }
    }

    Err(anyhow!(
        "WhisperX sidecar script not found. Expected at libremeet-sidecars/whisperx_diarize.py \
         or set WHISPERX_SIDECAR_PATH env var."
    ))
}

/// Resolve the Python interpreter to use.
///
/// Tries `python3`, then `python` (Windows-friendly).
fn resolve_python() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        "python"
    }
    #[cfg(not(target_os = "windows"))]
    {
        "python3"
    }
}

/// Run WhisperX diarization on an audio file.
///
/// Spawns a Python sidecar process, monitors progress via stdout lines
/// prefixed with "PROGRESS:", and returns the parsed diarization segments.
pub async fn run_whisperx(
    audio_path: &Path,
    output_path: &Path,
    hf_token: Option<&str>,
    progress_callback: &(dyn Fn(&str) + Send + Sync),
) -> Result<Vec<DiarizationSegment>> {
    let script_path = resolve_sidecar_script()?;
    let python = resolve_python();

    if !audio_path.exists() {
        return Err(anyhow!(
            "Audio file not found: {}",
            audio_path.display()
        ));
    }

    log::info!("Running WhisperX diarization on: {}", audio_path.display());
    log::info!("Sidecar script: {}", script_path.display());
    log::info!("Output path: {}", output_path.display());

    let mut cmd = Command::new(python);
    cmd.arg(&script_path)
        .arg("--audio")
        .arg(audio_path)
        .arg("--output")
        .arg(output_path);

    if let Some(token) = hf_token {
        if !token.is_empty() {
            cmd.arg("--hf-token").arg(token);
        }
    }

    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x08000000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    let mut child = cmd
        .spawn()
        .with_context(|| format!("Failed to spawn Python sidecar: {} {}", python, script_path.display()))?;

    // Read stdout for progress messages
    let stdout = child.stdout.take().ok_or_else(|| anyhow!("Failed to capture stdout"))?;
    let stderr = child.stderr.take().ok_or_else(|| anyhow!("Failed to capture stderr"))?;

    // Spawn stderr reader to capture errors in background
    let stderr_handle = tokio::spawn(async move {
        let mut reader = BufReader::new(stderr).lines();
        let mut errors = Vec::new();
        while let Ok(Some(line)) = reader.next_line().await {
            let trimmed = line.trim().to_string();
            if !trimmed.is_empty() {
                log::error!("[whisperx] {}", trimmed);
                errors.push(trimmed);
            }
        }
        errors
    });

    // Read stdout line by line, forward progress
    let mut stdout_reader = BufReader::new(stdout).lines();
    while let Ok(Some(line)) = stdout_reader.next_line().await {
        let trimmed = line.trim();
        if let Some(progress) = trimmed.strip_prefix("PROGRESS:") {
            progress_callback(progress.trim());
            log::info!("[whisperx] {}", progress.trim());
        } else if !trimmed.is_empty() {
            log::debug!("[whisperx stdout] {}", trimmed);
        }
    }

    // Wait for process to complete
    let status = child.wait().await.context("Failed to wait for sidecar process")?;
    let stderr_output = stderr_handle.await.unwrap_or_default();

    if !status.success() {
        let error_detail = if stderr_output.is_empty() {
            format!("exit code {}", status.code().unwrap_or(-1))
        } else {
            stderr_output.join("; ")
        };
        return Err(anyhow!("WhisperX diarization failed: {}", error_detail));
    }

    // Read and parse the output JSON
    let output_json = tokio::fs::read_to_string(output_path)
        .await
        .with_context(|| format!("Failed to read diarization output: {}", output_path.display()))?;

    let segments: Vec<DiarizationSegment> = serde_json::from_str(&output_json)
        .with_context(|| "Failed to parse diarization output JSON")?;

    log::info!(
        "Diarization complete: {} segments, {} unique speakers",
        segments.len(),
        segments
            .iter()
            .map(|s| &s.speaker_id)
            .collect::<std::collections::HashSet<_>>()
            .len()
    );

    Ok(segments)
}

/// Match diarization segments to transcript segments by temporal overlap.
///
/// For each transcript segment, find the diarization segment with the most
/// temporal overlap and assign its speaker.
pub fn assign_speakers_by_overlap(
    diarization: &[DiarizationSegment],
    transcript_ranges: &[(String, f64, f64)], // (id, start_sec, end_sec)
) -> Vec<(String, String)> {
    // Returns Vec<(transcript_id, speaker_id)>
    transcript_ranges
        .iter()
        .map(|(id, t_start, t_end)| {
            let mut best_speaker = "SPEAKER_UNKNOWN".to_string();
            let mut best_overlap: f64 = 0.0;

            for diar_seg in diarization {
                // Calculate overlap between [t_start, t_end] and [d_start, d_end]
                let overlap_start = t_start.max(diar_seg.start_sec);
                let overlap_end = t_end.min(diar_seg.end_sec);
                let overlap = (overlap_end - overlap_start).max(0.0);

                if overlap > best_overlap {
                    best_overlap = overlap;
                    best_speaker = diar_seg.speaker_id.clone();
                }
            }

            (id.clone(), best_speaker)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assign_speakers_by_overlap() {
        let diarization = vec![
            DiarizationSegment {
                speaker_id: "SPEAKER_00".into(),
                start_sec: 0.0,
                end_sec: 5.0,
                text: "Hello".into(),
            },
            DiarizationSegment {
                speaker_id: "SPEAKER_01".into(),
                start_sec: 5.0,
                end_sec: 10.0,
                text: "Hi".into(),
            },
        ];

        let transcript_ranges = vec![
            ("t1".into(), 1.0, 3.0),  // fully inside SPEAKER_00
            ("t2".into(), 6.0, 8.0),  // fully inside SPEAKER_01
            ("t3".into(), 4.0, 6.0),  // overlaps both, 1s with 00, 1s with 01 -> first wins
        ];

        let result = assign_speakers_by_overlap(&diarization, &transcript_ranges);
        assert_eq!(result[0].1, "SPEAKER_00");
        assert_eq!(result[1].1, "SPEAKER_01");
        // t3: overlap with 00 (4..5=1s) and 01 (5..6=1s) — both equal, first (SPEAKER_00) wins
        assert_eq!(result[2].1, "SPEAKER_00");
    }
}
