"""Display focus dispatchers for Sway, i3, macOS Yabai, and AppleScript."""

from abc import ABC, abstractmethod
import json
import logging
import shutil
import subprocess
import sys

logger = logging.getLogger(__name__)


class DisplayBackend(ABC):
    """Abstract interface for querying display geometry and dispatching focus."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the backend."""

    @abstractmethod
    def refresh_outputs(self) -> list[str]:
        """Query display server and return sorted display identifiers (left-to-right)."""

    @abstractmethod
    def focus_display(self, target_identifier: str, dry_run: bool = False) -> bool:
        """Dispatch focus to the specified display identifier."""


class SwayI3Backend(DisplayBackend):
    """Linux Sway and i3 window manager backend via IPC."""

    def __init__(self, ipc_cmd: str | None = None):
        if ipc_cmd:
            self.ipc_cmd = ipc_cmd
        elif shutil.which("swaymsg"):
            self.ipc_cmd = "swaymsg"
        elif shutil.which("i3-msg"):
            self.ipc_cmd = "i3-msg"
        else:
            self.ipc_cmd = "swaymsg"

    @property
    def name(self) -> str:
        return self.ipc_cmd

    def refresh_outputs(self) -> list[str]:
        if not shutil.which(self.ipc_cmd):
            logger.warning("IPC binary '%s' not found on PATH", self.ipc_cmd)
            return ["OUTPUT-LEFT", "OUTPUT-CENTER", "OUTPUT-RIGHT"]

        try:
            cmd = [self.ipc_cmd, "-t", "get_outputs"]
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(res.stdout)
            active = [o for o in data if o.get("active", False)]
            active.sort(key=lambda o: o.get("rect", {}).get("x", 0))
            names = [o["name"] for o in active if "name" in o]
            return names if names else ["OUTPUT-DEFAULT"]
        except Exception as exc:
            logger.warning("Failed to query %s outputs: %s. Using defaults.", self.ipc_cmd, exc)
            return ["OUTPUT-LEFT", "OUTPUT-CENTER", "OUTPUT-RIGHT"]

    def focus_display(self, target_identifier: str, dry_run: bool = False) -> bool:
        cmd = [self.ipc_cmd, "focus", "output", target_identifier]
        if dry_run:
            logger.info("[DRY-RUN] [%s] %s", self.name, " ".join(cmd))
            return True

        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except subprocess.SubprocessError as exc:
            logger.error("Failed to focus output via %s: %s", self.ipc_cmd, exc)
            return False


class OsascriptBackend(DisplayBackend):
    """macOS fallback backend using built-in osascript (JXA / CoreGraphics)."""

    def __init__(self, ipc_cmd: str = "osascript"):
        self.ipc_cmd = ipc_cmd
        self._display_coords: dict[str, tuple[float, float]] = {}

    @property
    def name(self) -> str:
        return "osascript"

    def refresh_outputs(self) -> list[str]:
        if not shutil.which(self.ipc_cmd):
            logger.warning("'%s' not found on PATH", self.ipc_cmd)
            return ["Display-1", "Display-2", "Display-3"]

        script = """
        ObjC.import('AppKit');
        var screens = $.NSScreen.screens;
        var out = [];
        for (var i = 0; i < screens.count; i++) {
            var s = screens.objectAtIndex(i);
            var f = s.frame;
            out.push({
                name: ObjC.unwrap(s.localizedName) || ('Display-' + (i + 1)),
                x: f.origin.x,
                y: f.origin.y,
                w: f.size.width,
                h: f.size.height
            });
        }
        JSON.stringify(out);
        """
        try:
            cmd = [self.ipc_cmd, "-l", "JavaScript", "-e", script]
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            screens = json.loads(res.stdout.strip())
            screens.sort(key=lambda s: s.get("x", 0.0))

            names = []
            self._display_coords.clear()
            for s in screens:
                name = s.get("name", "Display")
                names.append(name)
                cx = s.get("x", 0.0) + (s.get("w", 1920.0) / 2.0)
                cy = s.get("y", 0.0) + (s.get("h", 1080.0) / 2.0)
                self._display_coords[name] = (cx, cy)

            return names if names else ["Display-1"]
        except Exception as exc:
            logger.warning("Failed to query macOS screens via osascript: %s", exc)
            return ["Display-Left", "Display-Center", "Display-Right"]

    def focus_display(self, target_identifier: str, dry_run: bool = False) -> bool:
        coords = self._display_coords.get(target_identifier, (960.0, 540.0))
        cx, cy = coords
        script = f"""
        ObjC.import('CoreGraphics');
        $.CGWarpMouseCursorPosition($.CGPointMake({cx}, {cy}));
        """
        cmd = [self.ipc_cmd, "-l", "JavaScript", "-e", script]

        if dry_run:
            logger.info("[DRY-RUN] [%s] Warp mouse to %s (%.0f, %.0f)", self.name, target_identifier, cx, cy)
            return True

        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except subprocess.SubprocessError as exc:
            logger.error("Failed to focus screen via osascript: %s", exc)
            return False


class YabaiBackend(DisplayBackend):
    """macOS yabai tiling window manager backend with automatic osascript fallback."""

    def __init__(self, ipc_cmd: str = "yabai", fallback: DisplayBackend | None = None):
        self.ipc_cmd = ipc_cmd
        self.fallback = fallback or OsascriptBackend()
        self._name_to_index: dict[str, int] = {}

    @property
    def name(self) -> str:
        return "yabai"

    def refresh_outputs(self) -> list[str]:
        if not shutil.which(self.ipc_cmd):
            logger.info("yabai not found on PATH. Falling back to %s.", self.fallback.name)
            return self.fallback.refresh_outputs()

        try:
            cmd = [self.ipc_cmd, "-m", "query", "--displays"]
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            displays = json.loads(res.stdout.strip())
            # Sort horizontally by frame.x coordinate
            displays.sort(key=lambda d: d.get("frame", {}).get("x", 0.0))

            names = []
            self._name_to_index.clear()
            for d in displays:
                idx = d.get("index", 1)
                name = f"Display-{idx}"
                names.append(name)
                self._name_to_index[name] = idx

            return names if names else ["Display-1"]
        except Exception as exc:
            logger.warning("yabai query failed: %s. Falling back to %s.", exc, self.fallback.name)
            return self.fallback.refresh_outputs()

    def focus_display(self, target_identifier: str, dry_run: bool = False) -> bool:
        yabai_index = self._name_to_index.get(target_identifier)
        if yabai_index is None:
            # Identifier might belong to fallback
            return self.fallback.focus_display(target_identifier, dry_run=dry_run)

        cmd = [self.ipc_cmd, "-m", "display", "--focus", str(yabai_index)]
        if dry_run:
            logger.info("[DRY-RUN] [%s] %s", self.name, " ".join(cmd))
            return True

        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except subprocess.SubprocessError as exc:
            logger.warning("yabai focus command failed: %s. Falling back to %s.", exc, self.fallback.name)
            return self.fallback.focus_display(target_identifier, dry_run=dry_run)


class IpcClient:
    """Unified client managing display discovery and focus dispatching across platforms."""

    def __init__(
        self,
        backend: str = "auto",
        ipc_cmd: str | None = None,
        dry_run: bool = False,
    ):
        self.dry_run = dry_run
        self.backend = self._resolve_backend(backend, ipc_cmd)
        self._outputs: list[str] = []
        self.refresh_outputs()

    def _resolve_backend(self, backend: str, ipc_cmd: str | None) -> DisplayBackend:
        if backend == "sway":
            return SwayI3Backend(ipc_cmd=ipc_cmd or "swaymsg")
        if backend == "i3":
            return SwayI3Backend(ipc_cmd=ipc_cmd or "i3-msg")
        if backend == "yabai":
            return YabaiBackend(ipc_cmd=ipc_cmd or "yabai")
        if backend == "osascript":
            return OsascriptBackend(ipc_cmd=ipc_cmd or "osascript")

        # Auto detection
        if ipc_cmd:
            if "yabai" in ipc_cmd:
                return YabaiBackend(ipc_cmd=ipc_cmd)
            if "osascript" in ipc_cmd:
                return OsascriptBackend(ipc_cmd=ipc_cmd)
            return SwayI3Backend(ipc_cmd=ipc_cmd)

        if sys.platform == "darwin":
            if shutil.which("yabai"):
                return YabaiBackend()
            return OsascriptBackend()

        # Linux / Unix default
        if shutil.which("swaymsg"):
            return SwayI3Backend("swaymsg")
        if shutil.which("i3-msg"):
            return SwayI3Backend("i3-msg")
        if shutil.which("yabai"):
            return YabaiBackend()

        return SwayI3Backend("swaymsg")

    @property
    def outputs(self) -> list[str]:
        """Return cached sorted display identifiers (left-to-right)."""
        return list(self._outputs)

    def refresh_outputs(self) -> list[str]:
        """Query backend for active display identifiers sorted horizontally."""
        names = self.backend.refresh_outputs()
        self._outputs = names if names else ["Display-1"]
        return self._outputs

    def get_output_for_screen(self, screen_index: int) -> str | None:
        """Map coarse screen index (0=Left, 1=Center, 2=Right) to physical display identifier."""
        if not self._outputs:
            self.refresh_outputs()

        n = len(self._outputs)
        if n == 0:
            return None
        if n == 1:
            return self._outputs[0]
        if n == 2:
            return self._outputs[0] if screen_index == 0 else self._outputs[1]

        idx = max(0, min(screen_index, n - 1))
        return self._outputs[idx]

    def focus_screen(self, screen_index: int) -> bool:
        """Issue focus command to the display identifier associated with screen_index."""
        target_output = self.get_output_for_screen(screen_index)
        if not target_output:
            return False

        return self.backend.focus_display(target_output, dry_run=self.dry_run)
