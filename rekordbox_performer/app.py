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
    technique: str
    critical: str


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
    technique_name = (snapshot.transition_technique or "").replace("_", " ").title()
    technique = (
        f"Technique  {technique_name}"
        if technique_name
        else "Technique  Waiting for transition plan"
    )
    critical = (
        f"Critical handoff in {snapshot.critical_in_bars:.1f} bars"
        if snapshot.critical_in_bars is not None
        else "Critical handoff not yet scheduled"
    )
    return SnapshotView(
        headline=snapshot.headline,
        detail=snapshot.detail,
        now_playing=now_playing,
        next_track=next_track,
        clock="  ·  ".join(clock_parts) or "Waiting for live clock",
        health=health,
        severity=snapshot.severity,
        technique=technique,
        critical=critical,
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
    import pyrekordbox  # noqa: F401
    import rtmidi  # noqa: F401

    from . import server  # noqa: F401

    version("fastmcp")
    version("pyrekordbox")


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
    root.configure(bg="#090d18")
    root.attributes("-topmost", not args.not_topmost)
    root._rekordbot_expanded_live = False
    width, height = 620, 820
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.geometry(
        f"{width}x{height}+{screen_width - width - 18}+{screen_height - height - 68}"
    )
    root.minsize(580, 780)

    def report_error(title: str, message: str) -> None:
        root.after(0, lambda: messagebox.showerror(title, message))

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("Dark.TFrame", background="#090d18")
    style.configure("Card.TFrame", background="#121a2b")
    style.configure("Dark.TLabel", background="#090d18", foreground="#f5f7ff")
    style.configure("Card.TLabel", background="#121a2b", foreground="#f5f7ff")
    style.configure("Muted.TLabel", background="#090d18", foreground="#8f9bb3")
    style.configure("CardMuted.TLabel", background="#121a2b", foreground="#8f9bb3")
    style.configure(
        "Accent.TButton",
        background="#6c7cff",
        foreground="white",
        borderwidth=0,
        padding=(14, 8),
    )
    style.map("Accent.TButton", background=[("active", "#8491ff")])
    style.configure("TButton", padding=(10, 7))
    style.configure("TEntry", fieldbackground="#182238", foreground="#f5f7ff")
    style.configure("TCombobox", fieldbackground="#182238", foreground="#f5f7ff")
    style.configure("TSpinbox", fieldbackground="#182238", foreground="#f5f7ff")

    frame = ttk.Frame(root, padding=18, style="Dark.TFrame")
    frame.pack(fill="both", expand=True)
    ttk.Label(
        frame,
        text="REKORDBOT  /  AUTONOMOUS PERFORMANCE",
        style="Muted.TLabel",
        font=("Segoe UI Semibold", 9),
    ).pack(anchor="w")
    status_card = ttk.Frame(frame, padding=16, style="Card.TFrame")
    status_card.pack(fill="x", pady=(10, 0))
    headline = ttk.Label(
        status_card,
        text="Ready",
        style="Card.TLabel",
        font=("Segoe UI Semibold", 18),
    )
    headline.pack(anchor="w")
    now_playing = ttk.Label(
        status_card,
        text="No track playing",
        style="Card.TLabel",
        font=("Segoe UI Semibold", 11),
    )
    now_playing.pack(anchor="w", pady=(7, 0))
    clock = ttk.Label(
        status_card,
        text="Waiting for live clock",
        style="CardMuted.TLabel",
    )
    clock.pack(anchor="w", pady=(2, 0))
    next_track = ttk.Label(
        status_card,
        text="Next: not selected",
        style="Card.TLabel",
    )
    next_track.pack(anchor="w", pady=(8, 0))
    technique = ttk.Label(
        status_card,
        text="Technique  Waiting for transition plan",
        style="CardMuted.TLabel",
        font=("Segoe UI Semibold", 9),
    )
    technique.pack(anchor="w", pady=(8, 0))
    critical = ttk.Label(
        status_card,
        text="Critical handoff not yet scheduled",
        style="CardMuted.TLabel",
    )
    critical.pack(anchor="w", pady=(2, 0))
    detail = ttk.Label(
        status_card,
        text="",
        style="CardMuted.TLabel",
        wraplength=550,
        justify="left",
    )
    detail.pack(anchor="w", pady=(5, 0))
    health = ttk.Label(
        status_card,
        text="",
        style="CardMuted.TLabel",
        font=("Segoe UI", 8),
    )
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

    # During live automation the setup window becomes a compact, click-through
    # overlay. It remains visible above Rekordbox without intercepting browser,
    # deck, or waveform input and without forcing the user to minimize it.
    live_overlay = tk.Toplevel(root)
    live_overlay.withdraw()
    live_overlay.overrideredirect(True)
    live_overlay.configure(bg="#0b1020")
    live_overlay.attributes("-topmost", not args.not_topmost)
    live_overlay.attributes("-alpha", 0.95)
    overlay_width, overlay_height = 420, 218
    live_overlay.geometry(
        f"{overlay_width}x{overlay_height}+"
        f"{screen_width - overlay_width - 18}+"
        f"{screen_height - overlay_height - 68}"
    )
    overlay_shell = tk.Frame(
        live_overlay,
        bg="#121a2b",
        highlightbackground="#33415f",
        highlightthickness=1,
        padx=16,
        pady=14,
    )
    overlay_shell.pack(fill="both", expand=True)
    tk.Label(
        overlay_shell,
        text="REKORDBOT  LIVE",
        bg="#121a2b",
        fg="#7f8cff",
        font=("Segoe UI Semibold", 9),
    ).pack(anchor="w")
    overlay_headline = tk.Label(
        overlay_shell,
        text="Ready",
        bg="#121a2b",
        fg="#f5f7ff",
        font=("Segoe UI Semibold", 16),
        anchor="w",
    )
    overlay_headline.pack(fill="x", pady=(5, 0))
    overlay_now = tk.Label(
        overlay_shell,
        text="No track playing",
        bg="#121a2b",
        fg="#dce3f7",
        font=("Segoe UI Semibold", 10),
        anchor="w",
    )
    overlay_now.pack(fill="x", pady=(5, 0))
    overlay_next = tk.Label(
        overlay_shell,
        text="Next: not selected",
        bg="#121a2b",
        fg="#9da9c2",
        font=("Segoe UI", 9),
        anchor="w",
    )
    overlay_next.pack(fill="x", pady=(2, 0))
    overlay_technique = tk.Label(
        overlay_shell,
        text="Technique  Waiting for transition plan",
        bg="#121a2b",
        fg="#7de2b8",
        font=("Segoe UI Semibold", 9),
        anchor="w",
    )
    overlay_technique.pack(fill="x", pady=(8, 0))
    overlay_clock = tk.Label(
        overlay_shell,
        text="Waiting for live clock",
        bg="#121a2b",
        fg="#9da9c2",
        font=("Segoe UI", 9),
        anchor="w",
    )
    overlay_clock.pack(fill="x", pady=(2, 0))
    overlay_health = tk.Label(
        overlay_shell,
        text="",
        bg="#121a2b",
        fg="#74819b",
        font=("Segoe UI", 8),
        anchor="w",
    )
    overlay_health.pack(fill="x", pady=(8, 0))

    def make_overlay_click_through() -> None:
        if args.not_topmost:
            return
        try:
            import ctypes

            widget_handle = live_overlay.winfo_id()
            hwnd = ctypes.windll.user32.GetParent(widget_handle) or widget_handle
            get_style = ctypes.windll.user32.GetWindowLongW
            set_style = ctypes.windll.user32.SetWindowLongW
            style_value = get_style(hwnd, -20)
            set_style(hwnd, -20, style_value | 0x20 | 0x80 | 0x08000000)
        except (AttributeError, OSError):
            # The overlay remains useful on non-Windows test hosts; only the
            # pass-through window style is platform-specific.
            return

    live_overlay.update_idletasks()
    make_overlay_click_through()
    live_overlay_active = False

    ttk.Separator(frame).pack(fill="x", pady=(16, 10))
    setup = ttk.Frame(frame, padding=14, style="Card.TFrame")
    setup.pack(fill="both", expand=True)
    prepared = engine.profile_store.list_profiles(ready_only=True)
    prepared_labels = [engine.track_label(item) for item in prepared]
    try:
        playlists = engine.playlists()
        playlist_status = f"{len(playlists)} found"
    except Exception:  # noqa: BLE001 - optional read-only Rekordbox catalog
        playlists = []
        playlist_status = "catalog unavailable"
    playlist_by_label = {item.label: item for item in playlists}
    track_var = tk.StringVar(value=args.track or "")
    playlist_var = tk.StringVar(value="")
    count_var = tk.IntVar(value=args.count)
    endless_var = tk.BooleanVar(value=False)
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
        style="CardMuted.TLabel",
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
        style="CardMuted.TLabel",
        wraplength=560,
    ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(3, 9))

    ttk.Label(
        setup,
        text="REKORDBOX PLAYLIST (OPTIONAL)",
        style="CardMuted.TLabel",
        font=("Segoe UI Semibold", 8),
    ).grid(row=3, column=0, columnspan=2, sticky="w")
    ttk.Label(
        setup,
        text=playlist_status,
        style="CardMuted.TLabel",
    ).grid(row=3, column=2, sticky="e")
    playlist_combo = ttk.Combobox(
        setup,
        textvariable=playlist_var,
        values=["", *playlist_by_label],
        state="readonly",
        width=48,
    )
    playlist_combo.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(3, 9))

    ttk.Label(setup, text="Set length", style="CardMuted.TLabel").grid(
        row=5, column=0, sticky="w"
    )
    ttk.Label(
        setup,
        text="Finish near BPM (optional)",
        style="CardMuted.TLabel",
    ).grid(row=5, column=1, sticky="w", padx=(8, 0))
    ttk.Spinbox(setup, from_=2, to=100, textvariable=count_var, width=8).grid(
        row=6, column=0, sticky="w", pady=(2, 9)
    )
    ttk.Entry(setup, textvariable=bpm_var, width=12).grid(
        row=6, column=1, sticky="w", padx=(8, 0), pady=(2, 9)
    )
    ttk.Checkbutton(setup, text="Endless set", variable=endless_var).grid(
        row=6, column=2, sticky="e", padx=(8, 0), pady=(2, 9)
    )

    ttk.Label(
        setup,
        text="DIRECTION / DESTINATION (OPTIONAL)",
        style="CardMuted.TLabel",
        font=("Segoe UI Semibold", 8),
    ).grid(row=7, column=0, columnspan=3, sticky="w")
    ttk.Label(setup, text="Destination track", style="CardMuted.TLabel").grid(
        row=8, column=0, columnspan=3, sticky="w", pady=(3, 0)
    )
    target_combo = ttk.Combobox(
        setup,
        textvariable=target_var,
        values=["", *prepared_labels],
        width=48,
    )
    target_combo.grid(row=9, column=0, columnspan=3, sticky="ew", pady=(2, 6))
    ttk.Label(setup, text="Vibe", style="CardMuted.TLabel").grid(
        row=10, column=0, sticky="w"
    )
    ttk.Label(
        setup,
        text="Arrive over next transitions",
        style="CardMuted.TLabel",
    ).grid(row=10, column=1, columnspan=2, sticky="w", padx=(8, 0))
    vibe_combo = ttk.Combobox(
        setup,
        textvariable=vibe_var,
        values=list(VIBE_LABELS),
        state="readonly",
        width=23,
    )
    vibe_combo.grid(row=11, column=0, sticky="w", pady=(2, 8))
    ttk.Spinbox(
        setup,
        from_=2,
        to=6,
        textvariable=steer_count_var,
        width=8,
    ).grid(row=11, column=1, sticky="w", padx=(8, 0), pady=(2, 8))
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
        nonlocal live_overlay_active
        try:
            playlist = playlist_by_label.get(playlist_var.get())
            opening = (
                engine.resolve_track(track_var.get())
                if track_var.get().strip()
                else None
            )
            if playlist is None and opening is None:
                raise ValueError("Choose an opening track or a Rekordbox playlist.")
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
        if playlist is None:
            brief = DJBrief(
                start_track_id=opening.track_id,
                target_track_count=int(count_var.get()),
                target_bpm=target,
                target_track_id=(
                    destination.track_id if destination is not None else None
                ),
                vibe=VIBE_LABELS[vibe_var.get()],
                name=f"RekordBot set — {opening.title}",
            )
            start_action = lambda: engine.start_set(
                brief,
                endless=endless_var.get(),
            )
        else:
            start_action = lambda: engine.start_playlist_set(
                playlist.playlist_id,
                opening_track_id=(opening.track_id if opening is not None else None),
                vibe=VIBE_LABELS[vibe_var.get()],
                endless=endless_var.get(),
            )
        # Swap the interactive setup window for a non-activating pass-through
        # live overlay. Rekordbox keeps focus for physical search/load/capture
        # operations while RekordBot remains continuously visible.
        engine.status_store.publish(
            phase=RuntimePhase.SELECTING,
            headline="Starting set preflight",
            detail="Compact live view active while Rekordbox staging begins.",
        )
        live_overlay_active = True
        root._rekordbot_expanded_live = False
        root.withdraw()
        live_overlay.deiconify()
        live_overlay.lift()
        worker.submit(_safe_result(start_action, lambda _result: None, report_error))

    def steer_set() -> None:
        root._rekordbot_expanded_live = False
        root.withdraw()
        live_overlay.deiconify()
        live_overlay.lift()
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

    def continue_set() -> None:
        nonlocal live_overlay_active
        live_overlay_active = True
        root._rekordbot_expanded_live = False
        root.withdraw()
        live_overlay.deiconify()
        live_overlay.lift()
        worker.submit(
            _safe_result(
                lambda: engine.continue_set(
                    target_query=target_var.get(),
                    vibe=VIBE_LABELS[vibe_var.get()],
                    transition_count=int(steer_count_var.get()),
                    endless=endless_var.get(),
                ),
                lambda _result: None,
                report_error,
            )
        )

    action_row = ttk.Frame(setup, style="Card.TFrame")
    action_row.grid(row=12, column=0, columnspan=3, sticky="ew", pady=(3, 0))
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
    continue_button = ttk.Button(
        action_row,
        text="Continue set",
        command=continue_set,
    )
    continue_button.pack(side="left", padx=(6, 0))
    ttk.Label(
        setup,
        text=(
            "Playlist mode maps every song once. Queue steering redirects an active "
            "set; Continue Set extends the final playing song. Endless Set keeps "
            "planning safe transitions until Stop After Current is pressed."
        ),
        style="CardMuted.TLabel",
        wraplength=550,
    ).grid(row=13, column=0, columnspan=3, sticky="w", pady=(6, 0))

    severity_colors = {
        "info": "#e8ebf2",
        "success": "#75e6a4",
        "warning": "#ffd166",
        "error": "#ff6b6b",
    }

    def refresh() -> None:
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
        technique.configure(text=view.technique)
        critical.configure(text=view.critical)
        detail.configure(text=view.detail)
        health.configure(text=view.health)
        overlay_headline.configure(
            text=view.headline, fg=severity_colors[view.severity]
        )
        overlay_now.configure(text=view.now_playing)
        overlay_next.configure(text=view.next_track)
        overlay_technique.configure(text=view.technique)
        overlay_clock.configure(text=(f"{view.clock}  ·  {view.critical}"))
        overlay_health.configure(text=view.health)
        active = snapshot.phase not in TERMINAL_PHASES
        if (
            live_overlay_active
            and snapshot.phase not in TERMINAL_PHASES
            and not root._rekordbot_expanded_live
            and live_overlay.state() == "withdrawn"
        ):
            live_overlay.deiconify()
            live_overlay.lift()
        hold_button.configure(state="normal" if active else "disabled")
        stop_button.configure(state="normal" if active else "disabled")
        emergency_button.configure(state="normal" if active else "disabled")
        start_button.configure(state="disabled" if active else "normal")
        steer_button.configure(state="normal" if active else "disabled")
        continue_button.configure(state="disabled" if active else "normal")
        use_deck_button.configure(state="disabled" if active else "normal")
        opening_combo.configure(state="disabled" if active else "normal")
        playlist_combo.configure(state="disabled" if active else "readonly")
        root.after(300, refresh)

    tray = _start_tray(
        root,
        live_overlay,
        engine,
        worker,
        messagebox,
        report_error,
    )

    def close_to_taskbar() -> None:
        snapshot = engine.status_store.read()
        if snapshot.phase not in TERMINAL_PHASES:
            root._rekordbot_expanded_live = False
            live_overlay.deiconify()
            live_overlay.lift()
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
    live_overlay,
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

    image = Image.new("RGB", (64, 64), "#090d18")
    draw = ImageDraw.Draw(image)
    draw.ellipse((8, 8, 56, 56), outline="#5b7cfa", width=6)
    draw.ellipse((27, 27, 37, 37), fill="#75e6a4")

    def on_open(_icon=None, _item=None) -> None:
        def show() -> None:
            snapshot = engine.status_store.read()
            if snapshot.phase in TERMINAL_PHASES:
                live_overlay.withdraw()
                root._rekordbot_expanded_live = False
                root.attributes("-topmost", True)
                root.deiconify()
                root.lift()
            else:
                live_overlay.withdraw()
                root._rekordbot_expanded_live = True
                root.attributes("-topmost", False)
                root.deiconify()
                root.lift()

        root.after(0, show)

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
                    "Use Stop After Current or Emergency Stop before exiting "
                    "the DJ engine.",
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
