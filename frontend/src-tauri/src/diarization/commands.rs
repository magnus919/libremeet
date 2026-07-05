// Tauri commands for speaker diarization via WhisperX sidecar.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Runtime};
use tempfile::tempdir;

use crate::database::repositories::meeting::MeetingsRepository;
use crate::state::AppState;

use super::sidecar;

/// Result returned to the frontend after diarization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiarizationResult {
    pub status: String,
    pub speaker_count: usize,
    pub chunk_count: usize,
    pub error: Option<String>,
}

/// Progress event emitted during diarization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiarizationProgress {
    pub status: String, // "processing" | "completed" | "error"
    pub message: Option<String>,
    pub speaker_count: Option<usize>,
    pub chunk_count: Option<usize>,
}

/// Audio file extensions to search for in a meeting folder.
const AUDIO_CANDIDATES: &[&str] = &[
    "audio.mp4",
    "audio.m4a",
    "audio.wav",
    "audio.mp3",
    "audio.flac",
    "audio.ogg",
    "recording.mp4",
    "audio.mkv",
    "audio.webm",
    "audio.wma",
];

/// Diarize a meeting's audio recording using WhisperX.
///
/// 1. Loads meeting metadata to find the audio file
/// 2. Spawns the WhisperX Python sidecar
/// 3. Assigns speaker IDs to transcript segments by time overlap
/// 4. Stores speaker name mappings in meeting metadata
#[tauri::command]
pub async fn api_diarize_meeting<R: Runtime>(
    app: AppHandle<R>,
    meeting_id: String,
    hf_token: Option<String>,
    state: tauri::State<'_, AppState>,
) -> Result<DiarizationResult, String> {
    log::info!("api_diarize_meeting called for meeting_id: {}", meeting_id);

    let pool = state.db_manager.pool();

    // 1. Load meeting metadata to get folder_path
    let meeting = MeetingsRepository::get_meeting_metadata(pool, &meeting_id)
        .await
        .map_err(|e| format!("Failed to load meeting: {}", e))?
        .ok_or_else(|| format!("Meeting not found: {}", meeting_id))?;

    let folder_path = match meeting.folder_path.filter(|p| !p.trim().is_empty()) {
        Some(p) => PathBuf::from(p),
        None => return Err("No recording folder found for this meeting. The meeting may not have been recorded.".into()),
    };

    // 2. Find the audio file in the meeting folder
    let audio_path = find_audio_in_folder(&folder_path)
        .map_err(|e| format!("No audio file found: {}", e))?;

    log::info!("Found audio file for diarization: {}", audio_path.display());

    // 3. Create temp dir for output
    let temp_dir = tempdir().map_err(|e| format!("Failed to create temp dir: {}", e))?;
    let output_path = temp_dir.path().join("diarization_output.json");

    // 4. Emit processing started
    emit_progress(&app, "processing", "Starting speaker identification...", None, None);

    // 5. Run WhisperX sidecar
    let app_for_progress = app.clone();
    let progress_callback = Box::new(move |msg: &str| {
        emit_progress(&app_for_progress, "processing", msg, None, None);
    });

    let hf_token_ref: Option<&str> = hf_token.as_deref().filter(|t| !t.is_empty());

    let diarization_segments = match sidecar::run_whisperx(
        &audio_path,
        &output_path,
        hf_token_ref,
        &*progress_callback,
    )
    .await
    {
        Ok(segments) => segments,
        Err(e) => {
            let msg = format!("Diarization failed: {}", e);
            emit_progress(&app, "error", &msg, None, None);
            return Err(msg);
        }
    };

    if diarization_segments.is_empty() {
        emit_progress(&app, "error", "No speech segments detected in the recording.", None, None);
        return Err("No speech segments detected".into());
    }

    // 6. Get transcript segments with their time ranges for matching
    let transcript_count = {
        let (transcripts, _total) = MeetingsRepository::get_meeting_transcripts_paginated(
            pool,
            &meeting_id,
            i64::MAX, // Get all segments
            0,
        )
        .await
        .map_err(|e| format!("Failed to load transcripts: {}", e))?;

        // Build (id, start_sec, end_sec) tuples for overlap matching
        let transcript_ranges: Vec<(String, f64, f64)> = transcripts
            .iter()
            .filter_map(|t| {
                let start = t.audio_start_time.unwrap_or(0.0);
                let end = t.audio_end_time.unwrap_or(start + t.duration.unwrap_or(2.0));
                Some((t.id.clone(), start, end))
            })
            .collect();

        // 7. Assign speakers by temporal overlap
        emit_progress(&app, "processing", "Assigning speakers to transcript segments...", None, None);

        let assignments = sidecar::assign_speakers_by_overlap(&diarization_segments, &transcript_ranges);

        // 8. Update each transcript segment in the database
        let mut assigned_count = 0usize;
        for (transcript_id, speaker_id) in &assignments {
            match MeetingsRepository::update_transcript_speaker(pool, transcript_id, speaker_id).await {
                Ok(true) => assigned_count += 1,
                Ok(false) => log::warn!("Transcript segment {} not found for speaker update", transcript_id),
                Err(e) => log::error!("Failed to update speaker for {}: {}", transcript_id, e),
            }
        }

        assignments.len()
    };

    // 9. Build speaker name mapping and store in meeting metadata
    let unique_speakers: Vec<String> = {
        let mut speakers: Vec<String> = diarization_segments
            .iter()
            .map(|s| s.speaker_id.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        speakers.sort();
        speakers
    };

    let mut speaker_map: HashMap<String, String> = HashMap::new();
    for (i, speaker_id) in unique_speakers.iter().enumerate() {
        speaker_map.insert(speaker_id.clone(), format!("Speaker {}", i + 1));
    }

    let metadata = serde_json::json!({
        "speakers": speaker_map,
    });

    MeetingsRepository::save_meeting_metadata(pool, &meeting_id, &metadata.to_string())
        .await
        .map_err(|e| format!("Failed to save meeting metadata: {}", e))?;

    // 10. Emit completion
    let speaker_count = unique_speakers.len();
    emit_progress(
        &app,
        "completed",
        &format!("Identified {} speakers across {} segments", speaker_count, transcript_count),
        Some(speaker_count),
        Some(transcript_count),
    );

    Ok(DiarizationResult {
        status: "completed".into(),
        speaker_count,
        chunk_count: transcript_count,
        error: None,
    })
}

/// Find an audio file in the given folder.
fn find_audio_in_folder(folder: &std::path::Path) -> Result<PathBuf> {
    if !folder.exists() {
        return Err(anyhow!("Folder does not exist: {}", folder.display()));
    }

    for name in AUDIO_CANDIDATES {
        let path = folder.join(name);
        if path.exists() {
            return Ok(path);
        }
    }

    // Fallback: scan for any audio file
    if let Ok(entries) = std::fs::read_dir(folder) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(ext) = path.extension() {
                let ext = ext.to_string_lossy().to_lowercase();
                if matches!(
                    ext.as_str(),
                    "mp4" | "m4a" | "wav" | "mp3" | "flac" | "ogg" | "webm" | "wma" | "mkv"
                ) {
                    return Ok(path);
                }
            }
        }
    }

    Err(anyhow!("No audio file found in: {}", folder.display()))
}

/// Emit a diarization progress event to the frontend.
fn emit_progress<R: Runtime>(
    app: &AppHandle<R>,
    status: &str,
    message: &str,
    speaker_count: Option<usize>,
    chunk_count: Option<usize>,
) {
    let payload = DiarizationProgress {
        status: status.to_string(),
        message: Some(message.to_string()),
        speaker_count,
        chunk_count,
    };

    if let Err(e) = app.emit("diarization-progress", payload) {
        log::error!("Failed to emit diarization progress: {}", e);
    }
}

/// Save custom speaker names for a meeting.
///
/// The `speakers` map is keyed by speaker_id (e.g., "SPEAKER_00") with
/// the user-provided display name (e.g., "Alice").
#[tauri::command]
pub async fn api_save_speaker_names<R: Runtime>(
    _app: AppHandle<R>,
    meeting_id: String,
    speakers: HashMap<String, String>,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    log::info!("api_save_speaker_names for meeting_id: {}", meeting_id);

    let pool = state.db_manager.pool();
    let metadata = serde_json::json!({
        "speakers": speakers,
    });

    MeetingsRepository::save_meeting_metadata(pool, &meeting_id, &metadata.to_string())
        .await
        .map_err(|e| format!("Failed to save speaker names: {}", e))
}
