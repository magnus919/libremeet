# LibreMeet

> **Note:** This fork exists to illustrate what vibe coding can do when applied to open-core software, supporting the article series ["Vibe Coding Open Core Out of its Lockbox"](https://magnus919.com/series/vibe-coding-open-core-out-of-its-lockbox/). I do not seriously intend to pursue a fully functional fork, and this repository is archived. If you're motivated to pick it up, I'd love to see what you build. — Magnus

A fork of [Meetily](https://meetily.ai) — an open-core AI meeting assistant — that exists because the MIT license lets you take good software and make it better without locking features behind a paywall.

This repo is the subject of the three-part series ["Vibe Coding Open Core Out of its Lockbox"](https://magnus919.com/series/vibe-coding-open-core-out-of-its-lockbox/), which documents the fork from start to finish:

- **Part I: Use the Source** — Reading the MIT codebase, mapping what's gated, building a roadmap
- **Part II: May the Fork Be With You** — The rebrand, stripping telemetry, the first feature
- **Part III: The Source Awakens** — Speaker diarization, the hard problems, and what vibe coding actually proves

## What's Different

Compared to the upstream Community Edition, this fork:

- Has its own name (rebranded across 69 files, 223 changes)
- Strips all PostHog analytics infrastructure (928 lines removed, 24 command registrations deleted)
- Ships a visual template editor UI for custom summary templates (670-line React dialog with backend save/delete)
- Includes speaker diarization via WhisperX sidecar (13 files, 488 insertions — a feature Meetily Pro marks as "Coming Soon")
- Uses more accurate open transcription models (NVIDIA Canary Qwen 2.5B at 5.63% WER, IBM Granite Speech 3.3 8B at 5.85% WER — both open licensed)

Everything here is MIT-licensed (or compatible). No features hidden behind a paywall. The code is what it is.

## Building

```bash
brew install rust cmake
git clone https://github.com/magnus919/libremeet.git
cd libremeet/frontend
cargo build --release
```

See the article series for the full story, including the build failures and why each missing dependency revealed itself.

## Why

Open-core is a business model where a company releases a limited version of their software under an open source license and keeps the advanced features behind a paywall. The paywalled features in Meetily's Pro tier — better transcription models, custom summary templates, advanced exports — use models and frameworks that are themselves open source. This fork simply connects the pieces that shouldn't have been separated.

## License

MIT. The same license as the upstream — with the original copyright notice preserved, as required.
