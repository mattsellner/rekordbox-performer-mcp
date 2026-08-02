# Rekordbox Performer

Safety-first live Rekordbox control over a dedicated virtual MIDI port.

> [!WARNING]
> This is an experimental prototype for supervised performance. Keep a
> physical controller available and test mappings with channel faders down.

## Signal path

`MCP client -> verified transition card -> quantized scheduler -> loopMIDI -> Rekordbox MIDI Learn`

The DDJ-FLX4 remains connected as the Hardware Unlock device and manual control
surface. This server does not modify the Rekordbox database or process audio.

## Performance-intelligence workflow

The original elapsed-time scheduler remains available only for dry-run,
low-level testing. Live musical handoffs must use the guarded workflow:

1. Seed candidate metadata with `preflight_playlist`.
2. Call rekordbox-mcp `get_track_analysis` or `resolve_track_analysis`, then
   pass its result to `ingest_rekordbox_analysis`.
3. Ingest native `PVDI` vocal spans and `PWV7` low-band spans when available;
   use `upsert_track_profile` only for additional verified landmarks.
4. Require `audit_track_profiles` and `preview_set_plan` to pass before track one.
5. Launch the first deck with `launch_staged_track` or
   `launch_verified_hot_cue`, then call `refresh_transition_state`. Do not
   promote an outbound MIDI command into an observation.
6. Express the transition in bars and beats with a `TransitionCard`.
7. Call `preview_transition_card`, arm control, then call
   `perform_transition_card` with a unique `execution_id` for that one live
   pass. Reuse the same ID only when retrying a lost MCP response.
8. Grade the practice pass with `record_rehearsal`.

For measured practice passes, install `.[rehearsal]`, call
`start_rehearsal_capture`, perform the mix, and call
`stop_rehearsal_capture`. `analyze_rehearsal` reports level, clipping, onset
regularity, and errors against supplied expected downbeats. Vocal clashes still
require listening and are recorded in the rehearsal review.

The card validator rejects:

- unverified phrase, vocal, bass, or incoming-load plans;
- unprepared tracks;
- stale or low-confidence deck clocks;
- missing or disabled incoming Beat Sync/Quantize state;
- blind Sync or Quantize toggle events;
- starts that do not land on a Rekordbox-analyzed phrase boundary;
- manual clocks or unverified vision clocks;
- an already-playing incoming deck;
- an incoming launch that is not exactly bar 0 beat 1 from a verified
  mix-in/phrase-start Hot Cue for a blend or bass swap;
- an incompatible Camelot move outside same-key, relative-major/minor, or
  adjacent-wheel relationships unless harmonic risk is explicitly accepted;
- events that are not ordered musically;
- a critical handoff without both bass moves on the same downbeat;
- an outgoing deck that is not faded to zero and stopped.

Simple `phrase_cut`, `echo_exit`, and `breakdown_handoff` cards may launch a
stopped deck with `play_pause` only when the incoming profile has a
high-confidence file-start phrase landmark. Drop-anchored blends and bass swaps
still require a verified Hot Cue.

The scheduler reports average and maximum event lateness for diagnostics.
Events sharing one musical timestamp dispatch concurrently, so two-deck Hot
Cue launches and bass swaps are not skewed by MIDI note-hold time.
Committed cards require an execution ID. Retrying an identical card with the
same ID returns the original job instead of scheduling it again; using that ID
for different events fails closed.
MIDI dispatch alone is never reported as a successful live handoff:
`perform_transition_card` verifies afterward that the incoming deck is moving
and the outgoing deck is stopped. A failed or unavailable observation marks the
job failed.

## Performance hardening

Codex may launch more than one stdio MCP client. Read-only clients can coexist,
but an OS-backed lease allows only one process to own the live MIDI output.
`control_status` reports the current lease owner. The lease is released on
disconnect or process exit.

Performance profiles and rehearsal reviews use a shared SQLite database in WAL
mode. Legacy `track-profiles.json` and `rehearsals.jsonl` data is imported once,
so concurrent MCP clients cannot overwrite one another's preparation.

`rekordbox_ui_status` is served by one shared demand-bounded observer. It scans
only during an explicit observation window and publishes timestamped snapshots
for every MCP client, including observation age and observer PID. It restarts
after idle, allows slow Rekordbox UI Automation scans to finish, and never
returns a stale snapshot as current. A screenshot-derived
snapshot is still planning evidence, not an authoritative musical clock.

Each observation now samples the UI Automation tree and window geometry once,
then parses both decks from that immutable sample. Transport checks reuse an
already captured initial deck snapshot when available. This avoids repeated
hundreds-of-control geometry walks during staging, cue verification, launch,
and post-transition verification while preserving title, artist, elapsed-time,
Sync, and Quantize evidence.

Track readiness uses three tiers:

- `A`: fully automation-ready;
- `B`: analyzed and suitable for a verified simple cut/reset, but not a full
  automated overlap;
- `C`: manual-only until preparation gaps are resolved.

`scheduler_metrics` reports aggregate p95/p99 event lateness, maximum lateness,
and the number of events that missed the 10 ms deadline.

## Native Rekordbox launch timing

The mapping includes Quantize and MIX POINT LINK controls. When verified Hot
Cues or memory cues mark MIX OUT and MIX IN, prefer Rekordbox's native MIX POINT
LINK for starting the incoming deck. This avoids treating an external monotonic
timer as Rekordbox's musical clock. Mixer and effect moves can still run through
the performer after the native launch.

Enable MIX POINT LINK in Rekordbox Preferences and re-import the current mapping
file after upgrading from the prototype.

## Safe Beat Sync and Quantize

The MIDI assignments for Beat Sync and Quantize are toggles, not idempotent
"on" commands. Never put `sync` or `quantize` directly in a transition card.
Instead:

1. submit a fresh high-confidence observation containing `sync_enabled` and
   `quantize_enabled`;
2. call `ensure_deck_modes`;
3. visually or natively re-observe the indicators;
4. preview the transition card.

`ensure_deck_modes` sends nothing when the requested state is already active.
If it must toggle a mode, the performer refuses to treat the predicted result
as verified until a new observation arrives.

## Phrase-aware compiler

Track profiles now retain Rekordbox's exact `PSSI` boundaries. The compiler
selects the next analyzed phrase satisfying `minimum_lead_bars`, or an explicit
`start_phrase_index`/`start_phrase_label`. It does not round starts to a fixed
8-, 16-, or 32-bar modulus. This preserves pickup phrases and structures whose
boundaries begin on beats such as bar 17 beat 3.

Live commits through the legacy wall-clock `perform_transition` tool are
disabled. Use `perform_transition_card`; low-level transition previews remain
available for MIDI bench testing.

## FLX4-style browser and load control

The mapping exposes the browser encoder and deck load buttons over the virtual
MIDI port:

- `browse_tracks(steps)` moves the browser selection up or down;
- `load_selected_track(deck)` loads the focused row to deck 1 or deck 2;
- `browser_back` and `browser_forward` close or open browser folders through
  the low-level `trigger_control` tool.

Call `browse_tracks` before `load_selected_track`. A highlighted search result
does not necessarily own keyboard/controller focus until the encoder moves the
browser selection. After loading, verify that the expected title is resident
and stopped before arming any cue or transition event. Rekordbox's Load Lock is
preserved and should reject loads into a playing deck.

For exact-title workflows, `select_track_exact` searches the visible browser
and refuses ambiguous rows. `load_track_exact` verifies that the target deck is
stopped, loads the selected row, and checks title and artist afterward. Unicode
apostrophe and quote variants are normalized without weakening artist checks.

Prefer `stage_track` for live work. It accepts a stable catalog `track_id`,
title, artist, source, and optional duplicate-row index. It restores browser
focus, first closes the channel fader and sends CUE, verifies the deck is
stopped, attempts the mapped FLX4 Load action, and falls back to a deterministic
drag-to-deck load if MIDI focus fails. If Rekordbox auto-starts the newly loaded
track, the server immediately mutes and stops that deck and rejects the staging
attempt; stage it again before verification. Browser row selection and drag
fallbacks use the row's non-editable selector gutter rather than the title cell,
preventing accidental inline title editing.

Successful staging also learns a session-local route for the stable
`track_id`: the query, duplicate-row index, and verified MIDI-or-drag load
method. Restaging that track later in the same set tries the proven query first,
skips a known-failing MIDI load when appropriate, and reuses coordinates from
the current browser snapshot. Preflight each planned track once before playback
so retired-deck reloads take this optimized path. The cache is intentionally
discarded when Codex/Rekordbox Performer restarts.

## Telemetry boundary

MIDI Learn is a control surface, not a complete deck-state API.
`observe_deck_state` accepts observations from a native, MIDI, vision, or manual
adapter and rejects stale state at compile time. Because this server currently
has outbound MIDI only, high-confidence `source="midi"` observations are
rejected: a command timestamp is not feedback. Manual observations are never
accepted as authoritative live clocks. Vision observations require verified
adapter confidence; an ad-hoc screenshot estimate cannot authorize a card.

Use `verify_hot_cue` before committing a Hot Cue card. Verification is
session-scoped and proves the loaded title, recalled position, and advancing
transport while the deck is muted. Use `launch_staged_track` for a verified
file-start launch and `refresh_transition_state` immediately before previewing
and committing the card. Raw `trigger_control` responses explicitly report
`effect_verified=false`.

After a transition completes, the verifier promotes the incoming deck's exact
scheduled Hot Cue dispatch time into the next authoritative live clock. This
keeps a multi-song set's timing runway continuous instead of depending on a
slow post-transition refresh. Any BPM difference greater than 0.05 BPM requires
Beat Sync in the card. Drop-oriented transition families (`long_blend`,
`bass_swap`, and `double_drop`) also require their critical bass-swap beat to
land on a high-confidence or verified incoming `drop` landmark; a merely
convenient phrase boundary is not sufficient.

`rank_transition_candidates` normalizes Camelot and conventional key labels,
filters incompatible moves by default, and ranks the remainder by harmonic
relationship and BPM proximity. Pass `outgoing_track_id` to add Rekordbox vocal
analysis as a soft score penalty. Vocal overlap never excludes a candidate or
blocks a card; phrase, bass, transport, and harmonic checks remain enforced.

## Continuous-set performance workflow

The reliability upgrade adds a restart-safe three-track horizon and measured
handoffs:

1. Call `prepare_set_session` with the ordered track IDs, then
   `start_set_session`. `set_session_status` always exposes current, next, and
   following tracks, even after the MCP process restarts.
2. Use `prepare_track_cues` offline to find a phrase-safe cue 16 bars (or 8 bars
   when necessary) before each verified drop. This produces a plan only; set
   the cue in Rekordbox and prove it with `verify_hot_cue`.
3. Use `plan_vocal_handoff` or the vocal-aware candidate ranker to choose a
   vocal owner without treating overlap as a hard failure.
4. Call `recommend_transition_fx` for Echo, Reverb, Spiral, or Vinyl Brake
   recommendations. Every recipe includes wet/dry and a mandatory off/reset
   tail. Rekordbox exposes Select Next/Back and Beat Up/Down through MIDI Learn,
   not fixed-effect selectors, so observe the current effect before cycling to
   the recommendation. Map those controls from
   `mapping/rekordbox-midi-learn.csv` before using them live.
5. Stage and schedule with `stage_and_schedule_transition_card`. The scheduler
   now reserves the existing user authorization for the complete job, so a
   long transition cannot lose control halfway through merely because the
   original arming timer expires.
6. After completion, call `transition_quality_report`, then
   `settle_set_transition`. The queue advances only after postconditions and QA
   pass; failures preserve the current/next pair for recovery.

`refresh_transition_state` now reconciles both decks from Rekordbox's displayed
elapsed time and the analyzed beat grid. This corrects absolute bar position
after a long-running track instead of copying a stale launch-relative clock.
The UI clock is whole-second precision, so exact phase is reported as telemetry;
Beat Sync, Quantize, and verified grid-aligned launch remain the phase authority.

For the current Rekordbox 7 setup, MIX POINT LINK is the preferred precise
launch mechanism. A future native or vision adapter can provide continuous deck
observations without changing the transition-card contract.

## Requirements

- Windows with Rekordbox 7 in Performance mode
- Python 3.12 or newer
- A virtual MIDI port such as loopMIDI
- A Rekordbox-compatible controller for Hardware Unlock when required

## Install

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[rehearsal]"
```

## Setup

1. Install and start loopMIDI.
2. Create a port named `Codex Rekordbox Performer` and enable loopMIDI autostart.
3. Open Rekordbox 7 in Performance mode with the FLX4 connected.
4. Open Preferences -> Controller -> MIDI and select the virtual port.
5. Import `mapping/Codex-Rekordbox-Performer.midi.csv`. The fallback
   `mapping/rekordbox-midi-learn.csv` can be used for manual MIDI Learn.
6. Keep channel faders down and decks stopped during mapping.
7. Register `rekordbox-performer` as a stdio MCP server in your MCP client.
8. Call `list_midi_outputs`, `connect_midi`, and `control_status`.
9. Enable MIX POINT LINK in Preferences when available.
10. Preflight profiles and a complete set plan.
11. Before track one, cycle every planned incoming track through its assigned
    deck and verify its Hot Cue for the current session; then restage the first
    two tracks. This keeps expensive cue audits out of the live retirement
    window.
12. Preview a transition card, arm control, then perform it.

The dedicated mapping does not replace the DDJ-FLX4 factory mapping.

## Safety model

- Channel faders own level handoffs. The crossfader is left untouched so it can
  remain disabled in Rekordbox.
- Rekordbox Track Separation is supported through visually verified Vocal,
  Instrumental, and Drums toggles. Import the current MIDI mapping before using
  stem-dependent transition cards.
- Prefer a phrase-aligned loop or verified vocal-stem mute when a useful long
  blend would otherwise create vocal interference. Restore every loop and stem
  state before the retired deck is reloaded.
- Live messages are rejected until control is armed.
- Arming expires automatically after at most 30 minutes.
- Transition plans default to dry-run preview.
- Low-level scheduled events run against a local monotonic clock.
- Musical cards quantize their start from a fresh beat/bar observation.
- Stale state and incomplete preparation fail closed.
- Emergency stop cancels pending automation and disarms without moving controls.
- The physical FLX4 remains the manual override.

Do not move a physical FLX4 continuous control while automation is moving the
same software control; mismatched hardware positions can cause value jumps.
