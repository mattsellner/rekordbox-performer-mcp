# Rekordbox Performer

Safety-first live Rekordbox control over a dedicated virtual MIDI port.

> [!WARNING]
> This is an experimental prototype for supervised performance. Keep a
> physical controller available and test mappings with channel faders down.

## Signal path

`MCP client → transition scheduler → loopMIDI → Rekordbox MIDI Learn`

The DDJ-FLX4 remains connected as the Hardware Unlock device and manual control
surface. This server does not modify the Rekordbox database or process audio.

## Requirements

- Windows with Rekordbox 7 in Performance mode
- Python 3.12 or newer
- A virtual MIDI port such as loopMIDI
- A Rekordbox-compatible controller for Hardware Unlock when required

## Install

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

## Setup

1. Install and start loopMIDI.
2. Create a port named `Codex Rekordbox Performer` and enable loopMIDI autostart.
3. Open Rekordbox 7 in Performance mode with the FLX4 connected.
4. Open Preferences → Controller → MIDI and select the virtual port.
5. Import `mapping/Codex-Rekordbox-Performer.midi.csv`. The fallback
   `mapping/rekordbox-midi-learn.csv` can be used for manual MIDI Learn.
6. Keep channel faders down and decks stopped during mapping.
7. Register `rekordbox-performer` as a stdio MCP server in your MCP client.
8. Call `list_midi_outputs`, `connect_midi`, and `control_status`.
9. Preview a transition, arm control, then perform it.

The dedicated mapping does not replace the DDJ-FLX4 factory mapping.

## Safety model

- Live messages are rejected until control is armed.
- Arming expires automatically after at most 30 minutes.
- Transition plans default to dry-run preview.
- All scheduled events run against a local monotonic clock.
- Emergency stop cancels pending automation and disarms without moving controls.
- The physical FLX4 remains the manual override.

Do not move a physical FLX4 continuous control while automation is moving the
same software control; mismatched hardware positions can cause value jumps.
