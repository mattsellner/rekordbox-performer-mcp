"""Compact Windows corner application for RekordBot."""

from __future__ import annotations

import argparse
import asyncio
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .standalone_engine import StandaloneDJEngine
from .standalone_planner import DJBrief
from .standalone_status import RuntimePhase, RuntimeSnapshot

TERMINAL_PHASES = {
    RuntimePhase.IDLE,
    RuntimePhase.COMPLETE,
    RuntimePhase.FAILED,
}

VIBE_LABELS = {
    "Maintain current feel": "maintain",
    "More downtempo": "downtempo",
    "More energetic": "energetic",
    "Deeper / darker": "deeper",
    "More vocal": "vocal",
    "More instrumental": "instrumental",
}


@dataclass(frozen=True)
class SnapshotView:
    headline: str
    detail: str
    now_playing: str
    next_track: str
    clock: str
    health: str
    severity: str


def format_snapshot(snapshot: RuntimeSnapshot) -> SnapshotView:
    current = snapshot.current
    staged = snapshot.staged
    now_playing = (
        f"{current.title} — {current.artist}" if current.title else "No track playing"
    )
    next_track = (
        f"Next: {staged.title} · Deck {staged.deck}"
        if staged.title
        else "Next: not selected"
    )
    clock_parts = []
    if snapshot.bpm is not None:
        clock_parts.append(f"{snapshot.bpm:.1f} BPM")
    if snapshot.bar is not None and snapshot.beat is not None:
        clock_parts.append(f"bar {snapshot.bar}.{snapshot.beat}")
    if snapshot.action_in_bars is not None:
        clock_parts.append(f"action in {snapshot.action_in_bars:.1f} bars")
    if snapshot.phase == RuntimePhase.IDLE:
        health = (
            "Rekordbox monitoring OK · MIDI connects only at Start Set"
            if snapshot.health.rekordbox == "ok"
            else "Warming up passive Rekordbox monitoring · MIDI disconnected"
        )
    else:
        health = (
            f"MIDI {snapshot.health.midi.upper()}  ·  "
            f"Rekordbox {snapshot.health.rekordbox.upper()}  ·  "
            f"Rescue {snapshot.health.rescue.upper()}"
        )
    return SnapshotView(
        headline=snapshot.headline,
        detail=snapshot.detail,
        now_playing=now_playing,
        next_track=next_track,
        clock="  ·  ".join(clock_parts) or "Waiting for live clock",
        health=health,
        severity=snapshot.severity,
    )


class AsyncWorker:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=self.loop.run_forever,
            name="rekordbot-engine",
            daemon=True,
        )
        self.thread.start()

    def submit(self, coroutine) -> None:
        asyncio.run_coroutine_threadsafe(coroutine, self.loop)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RekordBot autonomous DJ and corner status window"
    )
    parser.add_argument("--track", help="Opening track ID or title search")
    parser.add_argument("--count", type=int, default=6, help="Target track count")
    parser.add_argument("--target-bpm", type=float, help="Optional final BPM")
    parser.add_argument("--start", action="store_true", help="Start after launch")
    parser.add_argument("--not-topmost", action="store_true")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Import packaged runtime dependencies and exit",
    )
    parser.add_argument(
        "--guard-capture",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--guard-capture-pair", help=argparse.SUPPRESS)
    parser.add_argument("--stage-capture", help=argparse.SUPPRESS)
    parser.add_argument("--stage-deck", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--stage-title", help=argparse.SUPPRESS)
    parser.add_argument("--stage-artist", help=argparse.SUPPRESS)
    parser.add_argument(
        "--capture-at-monotonic",
        type=float,
        help=argparse.SUPPRESS,
    )
    return parser


def _packaging_self_test() -> None:
    """Fail fast when a dynamically loaded packaged dependency is absent."""
    from importlib.metadata import version

    import mido.backends.rtmidi  # noqa: F401
    import rtmidi  # noqa: F401

    from . import server  # noqa: F401

    version("fastmcp")


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.stage_capture:
        from .stage_capture import stage_to

        if args.stage_deck is None or not args.stage_title:
            raise SystemExit("stage capture requires --stage-deck and --stage-title")
        raise SystemExit(
            stage_to(
                args.stage_capture,
                deck=args.stage_deck,
                title=args.stage_title,
                artist=args.stage_artist,
            )
        )
    if args.guard_capture:
        from .guard_capture import capture_to

        raise SystemExit(capture_to(args.guard_capture))
    if args.guard_capture_pair:
        from .guard_capture import capture_pair_to

        if args.capture_at_monotonic is None:
            raise SystemExit("guard pair requires --capture-at-monotonic")
        raise SystemExit(
            capture_pair_to(
                args.guard_capture_pair,
                capture_at_monotonic=args.capture_at_monotonic,
            )
        )
    if args.self_test:
        _packaging_self_test()
        return
    import tkinter as tk
    from tkinter import messagebox, ttk

    engine = StandaloneDJEngine()
    worker = AsyncWorker()
    root = tk.Tk()
    root.title("RekordBot")
    root.configure(bg="#111318")
    root.attributes("-topmost", not args.not_topmost)
    width, height = 560, 585
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.geometry(
        f"{width}x{height}+{screen_width - width - 18}+{screen_height - height - 68}"
    )
    root.minsize(520, 540)

    def report_error(title: str, message: str) -> None:
        root.after(0, lambda: messagebox.showerror(title, message))

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("Dark.TFrame", background="#111318")
    style.configure("Dark.TLabel", background="#111318", foreground="#e8ebf2")
    style.configure("Muted.TLabel", background="#111318", foreground="#9ca5b5")
    style.configure("Accent.TButton", background="#5b7cfa", foreground="white")

    frame = ttk.Frame(root, padding=14, style="Dark.TFrame")
    frame.pack(fill="both", expand=True)
    headline = ttk.Label(
        frame, text="Ready", style="Dark.TLabel", font=("Segoe UI Semibold", 15)
    )
    headline.pack(anchor="w")
    now_playing = ttk.Label(
        frame, text="No track playing", style="Dark.TLabel", font=("Segoe UI", 11)
    )
    now_playing.pack(anchor="w", pady=(7, 0))
    clock = ttk.Label(frame, text="Waiting for live clock", style="Muted.TLabel")
    clock.pack(anchor="w", pady=(2, 0))
    next_track = ttk.Label(frame, text="Next: not selected", style="Dark.TLabel")
    next_track.pack(anchor="w", pady=(8, 0))
    detail = ttk.Label(
        frame, text="", style="Muted.TLabel", wraplength=395, justify="left"
    )
    detail.pack(anchor="w", pady=(5, 0))
    health = ttk.Label(frame, text="", style="Muted.TLabel", font=("Segoe UI", 8))
    health.pack(anchor="w", pady=(8, 0))

    controls = ttk.Frame(frame, style="Dark.TFrame")
    controls.pack(fill="x", pady=(11, 0))
    hold_button = ttk.Button(
        controls,
        text="Hold Current",
        command=lambda: worker.submit(_safe_async(engine.hold_current, report_error)),
    )
    hold_button.pack(side="left")
    stop_button = ttk.Button(
        controls,
        text="Stop After Current",
        command=lambda: worker.submit(
            _safe_async(engine.stop_after_current, report_error)
        ),
    )
    stop_button.pack(side="left", padx=6)
    emergency_button = ttk.Button(
        controls,
        text="Emergency Stop",
        command=lambda: _confirm_emergency(
            lambda: worker.submit(_safe_async(engine.emergency_stop, report_error)),
            messagebox,
        ),
    )
    emergency_button.pack(side="right")

    ttk.Separator(frame).pack(fill="x", pady=(12, 7))
    setup = ttk.Frame(frame, style="Dark.TFrame")
    setup.pack(fill="x")
    prepared = engine.profile_store.list_profiles(ready_only=True)
    prepared_labels = [engine.track_label(item) for item in prepared]
    track_var = tk.StringVar(value=args.track or "")
    count_var = tk.IntVar(value=args.count)
    bpm_var = tk.StringVar(
        value="" if args.target_bpm is None else str(args.target_bpm)
    )
    target_var = tk.StringVar(value="")
    vibe_var = tk.StringVar(value="Maintain current feel")
    steer_count_var = tk.IntVar(value=3)
    opening_feedback = tk.StringVar(
        value=f"{len(prepared)} analyzed tracks are ready for automation."
    )

    ttk.Label(
        setup,
        text="OPENING TRACK",
        style="Muted.TLabel",
        font=("Segoe UI Semibold", 8),
    ).grid(row=0, column=0, columnspan=3, sticky="w")
    opening_combo = ttk.Combobox(
        setup,
        textvariable=track_var,
        values=prepared_labels,
        width=48,
    )
    opening_combo.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(3, 0))
    use_deck_button = ttk.Button(setup, text="Use loaded Deck 1")
    use_deck_button.grid(row=1, column=2, padx=(6, 0), pady=(3, 0))
    ttk.Label(
        setup,
        textvariable=opening_feedback,
        style="Muted.TLabel",
        wraplength=520,
    ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(3, 9))

    ttk.Label(setup, text="Set length", style="Muted.TLabel").grid(
        row=3, column=0, sticky="w"
    )
    ttk.Label(
        setup,
        text="Finish near BPM (optional)",
        style="Muted.TLabel",
    ).grid(row=3, column=1, sticky="w", padx=(8, 0))
    ttk.Spinbox(setup, from_=2, to=20, textvariable=count_var, width=8).grid(
        row=4, column=0, sticky="w", pady=(2, 9)
    )
    ttk.Entry(setup, textvariable=bpm_var, width=12).grid(
        row=4, column=1, sticky="w", padx=(8, 0), pady=(2, 9)
    )

    ttk.Label(
        setup,
        text="DIRECTION / DESTINATION (OPTIONAL)",
        style="Muted.TLabel",
        font=("Segoe UI Semibold", 8),
    ).grid(row=5, column=0, columnspan=3, sticky="w")
    ttk.Label(setup, text="Destination track", style="Muted.TLabel").grid(
        row=6, column=0, columnspan=3, sticky="w", pady=(3, 0)
    )
    target_combo = ttk.Combobox(
        setup,
        textvariable=target_var,
        values=["", *prepared_labels],
        width=48,
    )
    target_combo.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(2, 6))
    ttk.Label(setup, text="Vibe", style="Muted.TLabel").grid(
        row=8, column=0, sticky="w"
    )
    ttk.Label(
        setup,
        text="Arrive over next transitions",
        style="Muted.TLabel",
    ).grid(row=8, column=1, columnspan=2, sticky="w", padx=(8, 0))
    vibe_combo = ttk.Combobox(
        setup,
        textvariable=vibe_var,
        values=list(VIBE_LABELS),
        state="readonly",
        width=23,
    )
    vibe_combo.grid(row=9, column=0, sticky="w", pady=(2, 8))
    ttk.Spinbox(
        setup,
        from_=2,
        to=6,
        textvariable=steer_count_var,
        width=8,
    ).grid(row=9, column=1, sticky="w", padx=(8, 0), pady=(2, 8))
    setup.columnconfigure(0, weight=1)
    setup.columnconfigure(1, weight=1)

    def select_opening(profile) -> None:
        def update() -> None:
            track_var.set(engine.track_label(profile))
            opening_feedback.set(
                f"Ready: {profile.title} — {profile.artist} · {profile.bpm:.1f} BPM"
            )

        root.after(0, update)

    def use_loaded_deck() -> None:
        opening_feedback.set(
            "Reading Rekordbox Deck 1… The first read can take up to a minute."
        )
        worker.submit(
            _safe_result(
                lambda: engine.loaded_track(1),
                select_opening,
                report_error,
            )
        )

    use_deck_button.configure(command=use_loaded_deck)

    def start_set() -> None:
        try:
            opening = engine.resolve_track(track_var.get())
            destination = (
                engine.resolve_track(target_var.get())
                if target_var.get().strip()
                else None
            )
        except ValueError as exc:
            messagebox.showerror("Choose a prepared track", str(exc))
            return
        try:
            target = float(bpm_var.get()) if bpm_var.get().strip() else None
        except ValueError:
            messagebox.showerror("Invalid BPM", "Target BPM must be a number.")
            return
        brief = DJBrief(
            start_track_id=opening.track_id,
            target_track_count=int(count_var.get()),
            target_bpm=target,
            target_track_id=(destination.track_id if destination is not None else None),
            vibe=VIBE_LABELS[vibe_var.get()],
            name=f"RekordBot set — {opening.title}",
        )
        # Rekordbox selection still uses physical browser/deck hit points. A
        # topmost corner window can cover them even after Rekordbox receives
        # focus, sending search and drag input into this app instead. Relinquish
        # the overlay before preflight; refresh restores it at a terminal state.
        # Publish the active phase on the UI thread first so the 300 ms refresh
        # cannot race the worker and restore topmost while preflight is starting.
        engine.status_store.publish(
            phase=RuntimePhase.SELECTING,
            headline="Starting set preflight",
            detail="Relinquishing the overlay before Rekordbox staging begins.",
        )
        root.attributes("-topmost", False)
        root.lower()
        worker.submit(_safe_start(engine, brief, report_error))

    def steer_set() -> None:
        worker.submit(
            _safe_result(
                lambda: engine.steer_set(
                    target_query=target_var.get(),
                    vibe=VIBE_LABELS[vibe_var.get()],
                    transition_count=int(steer_count_var.get()),
                ),
                lambda _result: None,
                report_error,
            )
        )

    action_row = ttk.Frame(setup, style="Dark.TFrame")
    action_row.grid(row=10, column=0, columnspan=3, sticky="ew", pady=(3, 0))
    start_button = ttk.Button(
        action_row,
        text="Plan, preflight, and start set",
        style="Accent.TButton",
        command=start_set,
    )
    start_button.pack(side="left", fill="x", expand=True)
    steer_button = ttk.Button(
        action_row,
        text="Queue steering",
        command=steer_set,
    )
    steer_button.pack(side="left", padx=(6, 0))
    ttk.Label(
        setup,
        text=(
            "Start Set uses the full set length. Queue steering preserves the "
            "already-armed next track, then uses the transition count above."
        ),
        style="Muted.TLabel",
        wraplength=510,
    ).grid(row=11, column=0, columnspan=3, sticky="w", pady=(6, 0))

    severity_colors = {
        "info": "#e8ebf2",
        "success": "#75e6a4",
        "warning": "#ffd166",
        "error": "#ff6b6b",
    }
    overlay_topmost: bool | None = None

    def refresh() -> None:
        nonlocal overlay_topmost
        snapshot = engine.status_store.read()
        if (
            snapshot.phase == RuntimePhase.IDLE
            and snapshot.current.state == "manual"
            and snapshot.current.track_id
            and not track_var.get().strip()
        ):
            track_var.set(
                engine.track_label(engine.profile_store.get(snapshot.current.track_id))
            )
        view = format_snapshot(snapshot)
        headline.configure(
            text=view.headline, foreground=severity_colors[view.severity]
        )
        now_playing.configure(text=view.now_playing)
        next_track.configure(text=view.next_track)
        clock.configure(text=view.clock)
        detail.configure(text=view.detail)
        health.configure(text=view.health)
        active = snapshot.phase not in TERMINAL_PHASES
        desired_topmost = bool(not args.not_topmost and not active)
        if desired_topmost != overlay_topmost:
            root.attributes("-topmost", desired_topmost)
            overlay_topmost = desired_topmost
        hold_button.configure(state="normal" if active else "disabled")
        stop_button.configure(state="normal" if active else "disabled")
        emergency_button.configure(state="normal" if active else "disabled")
        start_button.configure(state="disabled" if active else "normal")
        steer_button.configure(state="normal" if active else "disabled")
        use_deck_button.configure(state="disabled" if active else "normal")
        opening_combo.configure(state="disabled" if active else "normal")
        root.after(300, refresh)

    tray = _start_tray(root, engine, worker, messagebox, report_error)

    def close_to_taskbar() -> None:
        if tray is not None:
            root.withdraw()
        else:
            root.iconify()

    root.protocol("WM_DELETE_WINDOW", close_to_taskbar)
    worker.submit(engine.monitor_manual_playback())
    refresh()
    if args.start and args.track:
        root.after(300, start_set)
    root.mainloop()


async def _safe_start(
    engine: StandaloneDJEngine,
    brief: DJBrief,
    report_error: Callable[[str, str], None],
) -> None:
    try:
        await engine.start_set(brief)
    except Exception as exc:  # noqa: BLE001 - GUI task boundary
        report_error("Set could not start", str(exc))


async def _safe_async(
    action: Callable[[], Any],
    report_error: Callable[[str, str], None],
) -> None:
    try:
        result = action()
        if asyncio.iscoroutine(result):
            await result
    except Exception as exc:  # noqa: BLE001 - GUI command boundary
        report_error("Command failed", str(exc))


async def _safe_result(
    action: Callable[[], Any],
    on_success: Callable[[Any], None],
    report_error: Callable[[str, str], None],
) -> None:
    try:
        result = action()
        if asyncio.iscoroutine(result):
            result = await result
        on_success(result)
    except Exception as exc:  # noqa: BLE001 - GUI command boundary
        report_error("Command failed", str(exc))


def _confirm_emergency(action: Callable[[], None], messagebox) -> None:
    if messagebox.askyesno(
        "Emergency stop",
        "Cancel automation and silence both decks now?",
    ):
        action()


def _start_tray(
    root,
    engine: StandaloneDJEngine,
    worker: AsyncWorker,
    messagebox,
    report_error: Callable[[str, str], None],
):
    """Create a Windows tray icon when the optional desktop extra is present."""
    try:
        import pystray
        from PIL import Image, ImageDraw
    except ImportError:
        return None

    image = Image.new("RGB", (64, 64), "#111318")
    draw = ImageDraw.Draw(image)
    draw.ellipse((8, 8, 56, 56), outline="#5b7cfa", width=6)
    draw.ellipse((27, 27, 37, 37), fill="#75e6a4")

    def on_open(_icon=None, _item=None) -> None:
        root.after(0, lambda: (root.deiconify(), root.lift()))

    def on_hold(_icon=None, _item=None) -> None:
        worker.submit(_safe_async(engine.hold_current, report_error))

    def on_stop(_icon=None, _item=None) -> None:
        worker.submit(_safe_async(engine.stop_after_current, report_error))

    def on_quit(icon, _item=None) -> None:
        def finish() -> None:
            snapshot = engine.status_store.read()
            if snapshot.phase not in TERMINAL_PHASES:
                messagebox.showwarning(
                    "Set still active",
                    "Use Stop After Current or Emergency Stop before exiting the DJ engine.",
                )
                return
            icon.stop()
            root.destroy()

        root.after(0, finish)

    icon = pystray.Icon(
        "rekordbot",
        image,
        "RekordBot",
        menu=pystray.Menu(
            pystray.MenuItem("Open", on_open, default=True),
            pystray.MenuItem("Hold Current", on_hold),
            pystray.MenuItem("Stop After Current", on_stop),
            pystray.MenuItem("Exit", on_quit),
        ),
    )
    icon.run_detached()
    return icon


if __name__ == "__main__":
    main()
