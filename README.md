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
5. Feed fresh observations for both decks through `observe_deck_state`,
   including Beat Sync and Quantize indicator state.
6. Express the transition in bars and beats with a `TransitionCard`.
7. Call `preview_transition_card`, arm control, then call
   `perform_transition_card`.
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
  mix-in/phrase-start Hot Cue;
- events that are not ordered musically;
- a critical handoff without both bass moves on the same downbeat;
- an outgoing deck that is not faded to zero and stopped.

The scheduler reports average and maximum event lateness for diagnostics.
Events sharing one musical timestamp dispatch concurrently, so two-deck Hot
Cue launches and bass swaps are not skewed by MIDI note-hold time.

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

## Telemetry boundary

MIDI Learn is a control surface, not a complete deck-state API.
`observe_deck_state` accepts observations from a native, MIDI, vision, or manual
adapter and rejects stale state at compile time. Manual observations are never
accepted as authoritative live clocks. Vision observations require verified
adapter confidence; an ad-hoc screenshot estimate cannot authorize a card.

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
11. Preview a transition card, arm control, then perform it.

The dedicated mapping does not replace the DDJ-FLX4 factory mapping.

## Safety model

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
