"""IPC client for Sway and i3 window managers."""

import json
import logging
import shutil
import subprocess

logger = logging.getLogger(__name__)


class IpcClient:
    """Manages display discovery and output focus dispatching via Sway/i3 IPC."""

    def __init__(self, ipc_cmd: str | None = None, dry_run: bool = False):
        self.dry_run = dry_run
        if ipc_cmd:
            self.ipc_cmd = ipc_cmd
        elif shutil.which("swaymsg"):
            self.ipc_cmd = "swaymsg"
        elif shutil.which("i3-msg"):
            self.ipc_cmd = "i3-msg"
        else:
            self.ipc_cmd = "swaymsg"

        self._outputs: list[str] = []
        self.refresh_outputs()

    @property
    def outputs(self) -> list[str]:
        """Return cached sorted output names (left-to-right)."""
        return list(self._outputs)

    def refresh_outputs(self) -> list[str]:
        """Query display server for active outputs sorted horizontally by rect.x."""
        if not shutil.which(self.ipc_cmd):
            logger.warning("IPC binary '%s' not found on PATH", self.ipc_cmd)
            self._outputs = ["OUTPUT-LEFT", "OUTPUT-CENTER", "OUTPUT-RIGHT"]
            return self._outputs

        try:
            cmd = [self.ipc_cmd, "-t", "get_outputs"]
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(res.stdout)
            active = [o for o in data if o.get("active", False)]
            # Sort horizontally by X geometry coordinate
            active.sort(key=lambda o: o.get("rect", {}).get("x", 0))
            names = [o["name"] for o in active if "name" in o]
            if names:
                self._outputs = names
            else:
                self._outputs = ["OUTPUT-DEFAULT"]
        except Exception as exc:
            logger.warning("Failed to query IPC outputs: %s. Using fallback defaults.", exc)
            if not self._outputs:
                self._outputs = ["OUTPUT-LEFT", "OUTPUT-CENTER", "OUTPUT-RIGHT"]

        return self._outputs

    def get_output_for_screen(self, screen_index: int) -> str | None:
        """Map coarse screen index (0=Left, 1=Center, 2=Right) to physical output name."""
        if not self._outputs:
            self.refresh_outputs()

        n = len(self._outputs)
        if n == 0:
            return None
        if n == 1:
            return self._outputs[0]
        if n == 2:
            # 2 monitors: 0=Left, 1/2=Right
            return self._outputs[0] if screen_index == 0 else self._outputs[1]

        # 3 or more monitors
        idx = max(0, min(screen_index, n - 1))
        return self._outputs[idx]

    def focus_screen(self, screen_index: int) -> bool:
        """Issue focus command to the display output associated with screen_index."""
        target_output = self.get_output_for_screen(screen_index)
        if not target_output:
            return False

        cmd = [self.ipc_cmd, "focus", "output", target_output]
        if self.dry_run:
            logger.info("[DRY-RUN] Executing: %s", " ".join(cmd))
            return True

        try:
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            return True
        except subprocess.SubprocessError as exc:
            logger.error("IPC dispatch failed: %s", exc)
            return False
