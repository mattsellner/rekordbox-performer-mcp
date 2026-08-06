# RekordBot

An autonomous Windows DJ and safety-first Rekordbox control engine operating
over a dedicated virtual MIDI port.

> [!WARNING]
> This is an experimental prototype for supervised performance. Keep a
> physical controller available and test mappings with channel faders down.

## Signal path

`DJ/TransitionKing plan -> autonomous set runner -> verified transition cards -> quantized scheduler -> loopMIDI -> Rekordbox MIDI Learn`

The DDJ-FLX4 remains connected as the Hardware Unlock device and manual control
surface. This server does not modify the Rekordbox database or process audio.

## RekordBot for Windows

RekordBot moves the DJ and TransitionKing decisions into the local
Rekordbox runtime. Codex is not consulted between tracks and can be closed
without interrupting the set. RekordBot:

- selects a complete route from Tier-A analyzed tracks before playback;
- scores long blends, compact bass swaps, filter exits, breakdown handoffs,
  loop bridges, phrase cuts, and proficiency-gated stem blends from verified
  phrase, vocal, waveform-energy, key, and BPM evidence;
- pairs the incoming chorus/drop with an outgoing energy release instead of
  choosing a bass-swap bar from elapsed overlap length alone;
- starts the opener at its own native BPM before enabling Beat Sync;
- keeps the outgoing channel full while the incoming channel establishes,
  swaps bass only on a verified incoming bass phrase, and retires the outgoing
  deck afterward;
- preloads the following track while a gradual tempo ramp is running;
- protects 64/32/16-bar staging, reserve, and rescue deadlines locally; and
- exposes a modern expanded setup panel plus a compact, click-through live
  overlay that stays visible without blocking Rekordbox controls or captures;
  and provides system-tray controls for Hold Current, Stop After Current, and
  Emergency Stop.

The live runtime is deterministic and local. It does not call an LLM, Codex,
OpenAI, or another cloud AI while a set is running. Rekordbox analysis and the
local technique scorer provide the planning evidence; the live clock, MIDI
events, safety checks, and recovery path remain ordinary repeatable code.

Only one process may own the virtual MIDI port. Disconnect the Codex Performer
before starting a live standalone set. Rekordbox must remain open in Performance
mode with the current MIDI mapping active and the FLX4 selected as its audio
device.

Run from source:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[app]"
.\.venv\Scripts\rekordbot.exe
```

Build the Windows executable:

```powershell
powershell -ExecutionPolicy Bypass -File .\build-app.ps1
```

The executable is created at `dist\RekordBot\RekordBot.exe`. Closing the
expanded window sends it to the system tray, so the local engine continues
running. During a set, RekordBot automatically uses its non-activating compact
overlay. Use **Open** from the tray to expand steering controls temporarily;
queueing a direction returns to the compact view. The app refuses to exit from
the tray while a set is active; use a normal or emergency stop first.

The existing `Codex Rekordbox Performer` virtual MIDI port, MIDI mapping file,
and `%LOCALAPPDATA%\rekordbox-performer` profile store intentionally retain
their compatibility names. Upgrading to RekordBot therefore preserves the
current Rekordbox mapping and analyzed-track library.

### RekordBot workflow

The Rekordbox browser highlight is not a stable track identity. To choose an
opener, use either of these explicit paths:

1. Load the intended opener onto stopped Deck 1 in Rekordbox, then click
   **Use loaded Deck 1**. The app observes the deck title and resolves it to one
   Tier-A prepared profile.
2. Choose an exact title/artist from the **Opening track** prepared-library
   dropdown. Typing part of a title or artist is accepted only when it resolves
   to one unambiguous ready profile.

Then set **Set length** and optionally **Finish near BPM**, choose a
**Destination track**, or select a direction such as **More downtempo**,
**More energetic**, **Deeper / darker**, **More vocal**, or
**More instrumental**. Click **Plan, preflight, and start set**. The app connects
MIDI at that point, verifies all tracks and transitions, and only then starts
the opener.

During playback, the destination, vibe, and **Arrive over next transitions**
controls remain available. Click **Queue steering** to redirect the set over
two to six transitions. The transition already armed is never replaced; the
new route begins with its incoming track, which preserves phrase timing and
the rolling two-track safety lead.

## Performance-intelligence workflow

The original elapsed-time scheduler remains available only for dry-run,
low-level testing. Live musical handoffs must use the guarded workflow:

1. Seed candidate metadata with `preflight_playlist`.
2. Call rekordbox-mcp `get_track_analysis` or `resolve_track_analysis`, then
   pass its result to `ingest_rekordbox_analysis`.
3. Ingest native `PVDI` vocal spans plus the complete per-bar `PWV7` low-band
   curve; use `upsert_track_profile` only for additional verified landmarks.
   Bass swaps fail closed unless the incoming downbeat and following phrase
   have strong, sustained low-end relative to that track. Use
   `analyze_incoming_bass_phrase` to inspect the evidence. A deliberate
   breakdown handoff must set `intentional_energy_drop` explicitly.
4. Require `audit_track_profiles` and `preflight_autonomous_set` to pass before
   track one. Preflight validates every primary and fallback branch, warms each
   exact browser/load route, and proves only the Hot Cues a card actually uses.
5. Express the first handoff in bars and beats with a `TransitionCard`, including
   an explicit analyzed `start_phrase_index` and any verified incoming Hot Cue.
6. Arm control, then call `start_autonomous_set`. It stages Track 1 and starts it
   only after the opening handoff is accepted by the local scheduler. Never
   manually launch Track 1 before that atomic call.
7. The local runner owns all later stage/load/verify/schedule/QA/advance work.
   It does not wait for a model or MCP round trip between songs. Use
   `autonomous_set_status` to monitor it and `stop_autonomous_set` to stop it.
8. `stage_and_schedule_transition_card` remains the atomic primitive for a
   supervised one-off rolling handoff. Use a unique `execution_id`; reuse it
   only when retrying a lost MCP response.
9. Grade the practice pass with `record_rehearsal`.

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
- a drop-oriented handoff without both bass moves on the same downbeat;
- an unaccepted tempo stretch above 4 percent;
- a loop action whose creation and release were not both verified/planned;
- an outgoing deck that is not faded to zero and stopped.

Simple `phrase_cut`, `echo_exit`, and `breakdown_handoff` cards may launch a
stopped deck with `play_pause` only when the incoming profile has a
high-confidence file-start phrase landmark. This is the cue-agnostic fallback:
the runner can keep a set moving even when a track has no prepared Hot Cue.
Drop-anchored blends and bass swaps still require a verified Hot Cue. When a
new automation-only cue is useful, reserve pads G and H; pads A, B, and C are
treated as the DJ's existing cues.

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

The configured MCP command is a stable stdio supervisor. After the one
initial Codex restart that activates this version, call `restart_performer`
between sets to load changed Performer code without closing Codex or Rekordbox.
The tool refuses to run while a transition job is active, disarms and
disconnects MIDI, replaces only the Performer child process, replays the MCP
handshake, and asks the client to refresh its tool list. Reconnect MIDI and arm
control again before performing; session-only Hot Cue proofs must be repeated.

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

The mapping also includes dedicated one- and two-beat jump controls used only
while the incoming channel fader is at zero. After launch, the Sync guard reads
the red downbeat markers in Rekordbox's stacked waveforms. A clean integer
one- or two-beat bar error is corrected on the muted incoming deck and observed
again. The detector follows the waveform rows across supported FX-panel heights
and requires two agreeing screenshots before trusting or correcting the result;
an unavailable, unstable, or still-misaligned grid cancels the job before the
first audible fader rise.

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
discarded when Codex/RekordBot Performer restarts.

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
transport while the deck is muted. `launch_staged_track` remains available for
single-track playback and diagnostics, but a live set opening must use
`launch_and_schedule_opening_transition`; this prevents audible playback from
continuing without an accepted transition job. Raw `trigger_control` responses
explicitly report `effect_verified=false`.

After a transition completes, the verifier promotes the incoming deck's exact
scheduled Hot Cue dispatch time into the next authoritative live clock. This
keeps a multi-song set's timing runway continuous instead of depending on a
slow post-transition refresh. Any BPM difference greater than 0.05 BPM requires
Beat Sync in the card. Any transition that transfers both decks' low EQ on its
critical downbeat is treated as a real bass handoff regardless of the family
label. It must launch a verified phrase-start Hot Cue exactly 8 or 16 bars
before a high-confidence/verified incoming `drop` or `bass_in` landmark, and
that landmark must itself be a Rekordbox phrase downbeat. A file start, an
arbitrary elapsed overlap, or a convenient phrase label cannot certify the
bass swap. `cue_preparation_plan` reserves G/H and works backward from those
verified bass-phrase landmarks.

`rank_transition_candidates` normalizes Camelot and conventional key labels,
filters incompatible moves by default, and ranks the remainder by harmonic
relationship and BPM proximity. Pass `outgoing_track_id` to add Rekordbox vocal
analysis as a soft score penalty. Vocal overlap never excludes a candidate or
blocks a card; phrase, bass, transport, and harmonic checks remain enforced.

## Continuous-set performance workflow

The reliability upgrade moves lifecycle ownership into Performer:

1. DJ and TransitionKing produce an `AutonomousSetPlan`: one opening track and
   a directed set of transition options. Every option contains the exact load
   identity, a verified transition card, a priority, and an optional post-mix
   `TempoPlan`. Multiple options from one outgoing track are ordered fallbacks.
   With the default `tempo_strategy="auto"`, Performer materializes a target
   for every path depth, interpolating from the opening BPM to the final native
   BPM (or `tempo_target_bpm`) over the whole set.
2. Call `preflight_autonomous_set`. It validates the complete reachable path,
   preparation tier, phrase/drop evidence, harmonic safety, tempo stretch,
   load identity, and cue requirements. It then warms deterministic load
   routes and restores the opening pair to stopped decks.
3. Call `start_autonomous_set` once. Performer persists the plan/state, reserves
   control for the estimated set duration, starts the opening track atomically,
   and adopts the first scheduler job.
4. After each verified handoff, the runner immediately selects, stages, and
   schedules the next option locally. It advances only after transport,
   fader/EQ, effects/stems cleanup, and pre-audible bar-alignment QA pass.
5. Before every staging retry, the runner measures remaining bars. At sixteen
   bars or less it creates a 4-, 8-, or 16-beat loop on the next phrase
   downbeat and verifies actual transport repetition. The accepted transition
   card releases that loop on bar 0 beat 1. This prevents an exhausted outgoing
   track from reaching silence while loading or verification recovers.
6. A failed primary option is retried only within its explicit budget; then the
   runner selects the next prepared fallback. A transient stale/missing deck
   observation does not consume that musical retry budget: the runner keeps the
   route, monitors the deadline, and engages the rescue loop if necessary.
   State and failures are visible in `autonomous_set_status` and survive a
   Performer process restart. Because scheduler jobs themselves are
   process-local, do not restart Performer during an active handoff.
7. After each verified handoff, the resolved `TempoPlan` ramps only the new,
   verified Master deck, normally over 32 bars. `tempo_strategy="manual"`
   requires an explicit ramp when the set spans a material BPM change;
   `tempo_strategy="hold"` is the deliberate fixed-tempo opt-out. Candidate
   selection and card validation reject more than a 4 percent stretch by
   default. Use an intermediate BPM bridge or explicitly accept the risk; do
   not leave a 130 BPM track parked at 121 BPM.

For a manually supervised one-off handoff, the older
`stage_and_schedule_transition_card` -> `transition_quality_report` ->
`settle_set_transition` path remains supported.

Use `plan_vocal_handoff` or the vocal-aware candidate ranker to choose a vocal
owner without treating overlap as a hard failure. `recommend_transition_fx`
provides Echo, Reverb, Spiral, or Vinyl Brake recipes with mandatory off/reset
tails. Rekordbox exposes Select Next/Back and Beat Up/Down through MIDI Learn,
so observe the current effect before cycling to a recommendation.

`refresh_transition_state` now reconciles both decks from Rekordbox's displayed
elapsed time and the analyzed beat grid. This corrects absolute bar position
after a long-running track instead of copying a stale launch-relative clock.
The elapsed-time UI clock is whole-second precision, so it is transport evidence
rather than phase authority. Exact beat-in-bar acceptance comes from the visible
downbeat-marker alignment; Beat Sync alone is insufficient because it can lock
beats while beat 1 on one deck is aligned with beat 3 on the other.

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
7. Register `rekordbot` as a stdio MCP server in your MCP client.
8. Call `list_midi_outputs`, `connect_midi`, and `control_status`.
9. Enable MIX POINT LINK in Preferences when available.
10. Build an `AutonomousSetPlan` and call `preflight_autonomous_set` while both
    decks are stopped. This warms loads and verifies required Hot Cues.
11. Confirm the FLX4 is the audio device and PC MASTER OUT is off.
12. Arm control and call `start_autonomous_set`.

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
- Normal manual arming expires automatically after at most 30 minutes. An
  explicitly started autonomous set extends that authorization only for its
  estimated duration (maximum four hours) and releases it on completion,
  failure, or stop.
- Transition plans default to dry-run preview.
- Low-level scheduled events run against a local monotonic clock.
- Musical cards quantize their start from a fresh beat/bar observation.
- Stale state and incomplete preparation fail closed.
- Emergency stop cancels pending automation and disarms without moving controls.
- The physical FLX4 remains the manual override.

Do not move a physical FLX4 continuous control while automation is moving the
same software control; mismatched hardware positions can cause value jumps.
