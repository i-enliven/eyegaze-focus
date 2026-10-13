"""Keyboard activity monitoring for freeze-on-typing focus gating."""

import logging
import os
import re
import select
import struct
import sys
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

EVENT_SIZE = struct.calcsize("llHHi")  # 24 bytes on 64-bit Linux
EV_KEY = 1
KEY_PRESS = 1
KEY_REPEAT = 2


class KeyboardMonitor:
    """Monitors system-wide keyboard input to suppress focus changes while typing."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._last_key_time: float = 0.0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._fds: list[int] = []
        self._is_active = False

        if not self.enabled:
            return

        if sys.platform == "darwin":
            self._init_macos()
        elif sys.platform.startswith("linux"):
            self._init_linux()

    def _init_macos(self) -> None:
        try:
            import ctypes
            from ctypes import util

            cg_path = util.find_library("CoreGraphics")
            if not cg_path:
                logger.warning("CoreGraphics library not found on macOS; keyboard gating disabled")
                return

            self._cg = ctypes.CDLL(cg_path)
            self._cg.CGEventSourceSecondsSinceLastEventType.restype = ctypes.c_double
            self._cg.CGEventSourceSecondsSinceLastEventType.argtypes = [ctypes.c_int, ctypes.c_uint32]
            self._is_active = True
            logger.info("Initialized native macOS CoreGraphics keyboard activity monitor")
        except Exception as e:
            logger.warning("Failed to initialize macOS keyboard monitor: %s", e)

    def _find_linux_keyboards(self) -> list[str]:
        devices = []
        devices_file = Path("/proc/bus/input/devices")
        if not devices_file.is_file():
            by_id = Path("/dev/input/by-id")
            if by_id.is_dir():
                for p in by_id.glob("*-kbd"):
                    devices.append(str(p.resolve()))
            return devices

        try:
            content = devices_file.read_text(encoding="utf-8", errors="ignore")
            for section in content.split("\n\n"):
                if "Handlers=" in section:
                    match_h = re.search(r"Handlers=(.*)", section)
                    if match_h and "kbd" in match_h.group(1):
                        for item in match_h.group(1).split():
                            if item.startswith("event"):
                                dev_path = f"/dev/input/{item}"
                                if os.path.exists(dev_path):
                                    devices.append(dev_path)
        except Exception as e:
            logger.debug("Error parsing /proc/bus/input/devices: %s", e)

        return list(dict.fromkeys(devices))

    def _init_linux(self) -> None:
        kb_devices = self._find_linux_keyboards()
        if not kb_devices:
            logger.warning("No keyboard input devices found in /dev/input")
            return

        open_fds = []
        permission_denied = False
        for dev in kb_devices:
            try:
                fd = os.open(dev, os.O_RDONLY | os.O_NONBLOCK)
                open_fds.append(fd)
            except PermissionError:
                permission_denied = True
            except OSError:
                pass

        if permission_denied and not open_fds:
            logger.warning(
                "Cannot open /dev/input keyboard devices (Permission denied). "
                "Ensure your user is in the 'input' group: 'sudo usermod -a -G input $USER' "
                "(requires logging out and back in to take effect). Keyboard gating paused."
            )
            return

        if not open_fds:
            logger.warning("Could not open any keyboard input devices")
            return

        self._fds = open_fds
        self._is_active = True
        self._thread = threading.Thread(target=self._linux_listen_loop, daemon=True, name="eyegaze-keyboard-monitor")
        self._thread.start()
        logger.info("Initialized Linux evdev keyboard monitor on %d input devices", len(self._fds))

    def _linux_listen_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                rlist, _, _ = select.select(self._fds, [], [], 0.5)
            except (ValueError, OSError):
                break

            for fd in rlist:
                try:
                    while True:
                        data = os.read(fd, EVENT_SIZE)
                        if len(data) < EVENT_SIZE:
                            break
                        # struct input_event: timeval (2 longs), type (H), code (H), value (i)
                        _, _, ev_type, _, ev_val = struct.unpack("llHHi", data)
                        if ev_type == EV_KEY:
                            with self._lock:
                                self._last_key_time = time.time()
                except BlockingIOError:
                    pass
                except OSError:
                    break

    def time_since_last_key(self) -> float:
        """Return elapsed seconds since the most recent keystroke."""
        if not self.enabled or not self._is_active:
            return float("inf")

        if sys.platform == "darwin":
            try:
                # stateID 1 = kCGEventSourceStateCombinedSessionState, eventType 10 = kCGEventKeyDown
                return float(self._cg.CGEventSourceSecondsSinceLastEventType(1, 10))
            except Exception:
                return float("inf")

        with self._lock:
            if self._last_key_time == 0.0:
                return float("inf")
            return max(0.0, time.time() - self._last_key_time)

    def is_typing(self, cooldown_sec: float) -> bool:
        """Check if a keystroke occurred within the specified cooldown duration."""
        if cooldown_sec <= 0.0 or not self.enabled:
            return False
        return self.time_since_last_key() < cooldown_sec

    def close(self) -> None:
        """Clean up background threads and open file descriptors."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.2)
        for fd in self._fds:
            try:
                os.close(fd)
            except OSError:
                pass
        self._fds.clear()
        self._is_active = False
