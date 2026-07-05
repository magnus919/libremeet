-- Migration: Add speaker diarization support
-- Adds speaker_id to transcripts for per-segment speaker labels
-- Adds metadata to meetings for storing speaker name mappings and other metadata

ALTER TABLE transcripts ADD COLUMN speaker_id TEXT;
ALTER TABLE meetings ADD COLUMN metadata TEXT;
