"""Control an isolated OBS instance for local recording comparisons.

The controller records only an explicit X11 window and a dedicated Pulse monitor.
It does not launch OBS, route audio, or start recording as a side effect of setup.
JSONL timestamps describe software requests/events, not physical stimulus onset.
Requires websocket-client only when connecting to OBS 28+ (WebSocket protocol 5).
"""

from __future__ import annotations

import argparse
import base64
import configparser
import hashlib
import json
import os
import re
import secrets
import shlex
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

NAME = "fmri-gym-validation"
VIDEO = "fmri-gym-window"
AUDIO = "fmri-gym-audio"
MONITOR_PATTERN = r"fmri_gym_record(?:_[a-zA-Z0-9_-]+)?\.monitor"


def _private_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("w", encoding="utf-8") as stream:
        os.chmod(path, 0o600)
        stream.write(value)


def _json_write(path: Path, value: Any) -> None:
    _private_write(path, json.dumps(value, indent=2) + "\n")


def _ini_write(path: Path, sections: dict[str, dict[str, Any]]) -> None:
    from io import StringIO

    config = configparser.ConfigParser(interpolation=None)
    config.optionxform = str
    for section, fields in sections.items():
        config[section] = {key: str(value) for key, value in fields.items()}
    result = StringIO()
    config.write(result, space_around_delimiters=False)
    _private_write(path, result.getvalue())


def _recording_settings(output: Path, encoder: str) -> dict[str, Any]:
    options = {
        "libx264": "preset=ultrafast crf=18 tune=zerolatency",
        # OBS 30's custom FFmpeg output does not drain delayed encoder packets at stop.
        "h264_nvenc": "preset=p4 rc=constqp qp=18 delay=0 zerolatency=1 bf=0 rc-lookahead=0",
    }
    if encoder not in options:
        raise ValueError(f"Encoder must be one of {list(options)}")
    return {
        "RecType": "FFmpeg", "FFOutputToFile": "true", "FFFilePath": output,
        "FFExtension": "mkv", "FFFormat": "matroska",
        "FFFormatMimeType": "video/x-matroska", "FFVEncoder": encoder,
        "FFVEncoderId": 27, "FFVCustom": options[encoder], "FFVBitrate": 0,
        "FFVGOPSize": 120, "FFRescale": "false", "FFIgnoreCompat": "false",
        "FFAEncoder": "pcm_s16le", "FFAEncoderId": 65536,
        "FFABitrate": 1536, "FFAudioMixes": 1, "FFACustom": "", "FFMCustom": "",
    }


def prepare(
    state_dir: str | Path, output_dir: str | Path, width: int, height: int,
    fps: int = 60, encoder: str = "libx264", port: int = 4457,
    sample_rate: int = 48000,
) -> dict[str, Any]:
    """Create private OBS settings without launching an application.

    :param state_dir: New directory for config, credentials and control logs.
    :param output_dir: Destination for MKV recordings.
    :return: Nonsecret launch environment/arguments and recording settings.
    :raises ValueError: If dimensions, rate, port or encoder are invalid.
    """
    if width < 16 or height < 16 or width % 2 or height % 2:
        raise ValueError("Use even video dimensions of at least 16 pixels")
    if not 1 <= fps <= 240 or not 1024 <= port <= 65535:
        raise ValueError("fps must be 1..240 and port must be 1024..65535")
    if sample_rate not in (44100, 48000):
        raise ValueError("sample_rate must be 44100 or 48000")
    root, output = Path(state_dir).resolve(), Path(output_dir).resolve()
    settings = _recording_settings(output, encoder)
    root.mkdir(parents=True, mode=0o700, exist_ok=False)
    output.mkdir(parents=True, exist_ok=True)
    obs = root / "xdg-config" / "obs-studio"
    password = secrets.token_urlsafe(36)
    _private_write(root / "websocket-password", password)
    _ini_write(obs / "global.ini", {
        "General": {"FirstRun": "true", "EnableAutoUpdates": "false"},
        "Basic": {"Profile": NAME, "ProfileDir": NAME,
                  "SceneCollection": NAME, "SceneCollectionFile": NAME},
        "BasicWindow": {"PreviewEnabled": "false", "ShowStatusBar": "true",
                        "ShowContextToolbars": "false"},
        "OBSWebSocket": {"FirstLoad": "false", "ServerEnabled": "true",
                         "ServerPort": port, "AuthRequired": "true",
                         "ServerPassword": password, "AlertsEnabled": "false"},
    })
    _ini_write(obs / "basic" / "profiles" / NAME / "basic.ini", {
        "General": {"Name": NAME},
        "Output": {"Mode": "Advanced", "FilenameFormatting": "%CCYY-%MM-%DD_%hh-%mm-%ss"},
        "AdvOut": settings,
        "Video": {"BaseCX": width, "BaseCY": height, "OutputCX": width,
                  "OutputCY": height, "FPSType": 1, "FPSInt": fps,
                  "ColorFormat": "NV12", "ColorSpace": "709", "ColorRange": "Partial"},
        "Audio": {"SampleRate": sample_rate, "ChannelSetup": "Stereo"},
    })
    # An existing scene file prevents OBS's first-run desktop/microphone defaults.
    _json_write(obs / "basic" / "scenes" / f"{NAME}.json", {
        "name": NAME, "current_scene": NAME, "current_program_scene": NAME,
        "scene_order": [{"name": NAME}], "groups": [],
        "sources": [{"name": NAME, "id": "scene", "versioned_id": "scene",
                     "settings": {"items": []}}],
    })
    config = {
        "state_dir": str(root), "output_dir": str(output), "port": port,
        "width": width, "height": height, "fps": fps, "encoder": encoder,
        "sample_rate": sample_rate, "audio_codec": "pcm_s16le",
        "launch_env": {"XDG_CONFIG_HOME": str(root / "xdg-config")},
        "launch_args": ["obs", "--multi", "--profile", NAME, "--collection", NAME,
                        "--scene", NAME, "--disable-missing-files-check"],
    }
    _json_write(root / "controller.json", config)
    return config


class OBSClient:
    """Synchronous OBS v5 controller for a single isolated local instance."""

    def __init__(self, state_dir: str | Path, timeout: float = 15):
        self.root = Path(state_dir).resolve()
        self.config = json.loads((self.root / "controller.json").read_text())
        self.timeout = timeout
        self.ws = None
        self.events: list[dict[str, Any]] = []

    def _log(self, kind: str, data: Any) -> None:
        record = {"monotonic_ns": time.monotonic_ns(), "wall_time_ns": time.time_ns(),
                  "kind": kind, "data": data}
        with (self.root / "control.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")

    def connect(self) -> OBSClient:
        """Authenticate locally and verify the isolated scene/profile.

        :return: This connected client, also usable as a context manager.
        :raises RuntimeError: If OBS is not the expected authenticated instance.
        """
        import websocket

        secret_path = self.root / "websocket-password"
        if secret_path.stat().st_mode & 0o077:
            raise RuntimeError("Password file must have mode 0600")
        self.ws = websocket.create_connection(
            f"ws://127.0.0.1:{self.config['port']}", timeout=self.timeout,
            http_no_proxy=["127.0.0.1", "localhost"],
        )
        try:
            self._identify(secret_path.read_text())
            self._assert_instance()
            self._log("connected", self.request("GetVersion"))
        except BaseException:
            self.close()
            raise
        return self

    def _identify(self, password: str) -> None:
        hello = json.loads(self.ws.recv())
        if hello["op"] != 0 or "authentication" not in hello["d"]:
            raise RuntimeError("Expected an authenticated OBS WebSocket v5 Hello")
        auth = hello["d"]["authentication"]
        secret = base64.b64encode(hashlib.sha256((password + auth["salt"]).encode()).digest())
        challenge = secret + auth["challenge"].encode()
        response = base64.b64encode(hashlib.sha256(challenge).digest()).decode()
        self.ws.send(json.dumps({"op": 1, "d": {
            "rpcVersion": 1, "authentication": response, "eventSubscriptions": 64,
        }}))
        if json.loads(self.ws.recv())["op"] != 2:
            raise RuntimeError("OBS authentication failed")

    def _assert_instance(self) -> None:
        scene = self.request("GetSceneCollectionList")["currentSceneCollectionName"]
        profile = self.request("GetProfileList")["currentProfileName"]
        if scene != NAME or profile != NAME:
            raise RuntimeError("Refusing to control a different OBS scene collection/profile")
        special = self.request("GetSpecialInputs")
        if any(special.values()):
            raise RuntimeError("Global desktop/microphone inputs must all be disabled")

    def request(self, name: str, **data: Any) -> dict[str, Any]:
        """Send an OBS request while preserving interleaved output events.

        :param name: OBS v5 request name.
        :return: The successful response data.
        :raises RuntimeError: If OBS rejects the request.
        """
        if self.ws is None:
            raise RuntimeError("Call connect() first")
        request_id = str(uuid.uuid4())
        sent = time.monotonic_ns()
        self._log("request", {"request": name, "request_id": request_id, "request_data": data})
        self.ws.send(json.dumps({"op": 6, "d": {
            "requestType": name, "requestId": request_id, "requestData": data,
        }}))
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            message = self._receive(deadline)
            if message["op"] != 7 or message["d"]["requestId"] != request_id:
                continue
            response = message["d"]
            self._log("response", {"request": name, "sent_monotonic_ns": sent,
                                   "request_data": data, "response": response})
            if not response["requestStatus"]["result"]:
                raise RuntimeError(f"{name}: {response['requestStatus']}")
            return response.get("responseData", {})
        raise TimeoutError(f"OBS timed out: {name}")

    def _receive(self, deadline: float) -> dict[str, Any]:
        self.ws.settimeout(max(0.001, deadline - time.monotonic()))
        message = json.loads(self.ws.recv())
        if message["op"] == 5:
            self.events.append(message["d"])
            self._log("event", message["d"])
        return message

    def _assert_idle(self) -> None:
        if self.request("GetRecordStatus")["outputActive"]:
            raise RuntimeError("Stop the existing recording before changing sources")

    def _configure_input(
        self, name: str, kind: str, settings: dict[str, Any], existing: set[str],
    ) -> None:
        if name not in existing:
            self.request("CreateInput", sceneName=NAME, inputName=name, inputKind=kind,
                         inputSettings=settings, sceneItemEnabled=False)
            return
        # Removing then recreating a source races OBS's asynchronous source release.
        item = self.request("GetSceneItemId", sceneName=NAME, sourceName=name)["sceneItemId"]
        self.request("SetSceneItemEnabled", sceneName=NAME, sceneItemId=item,
                     sceneItemEnabled=False)
        self.request("SetInputSettings", inputName=name, inputSettings=settings, overlay=False)

    def _verify_window(self, window_id: int, title: str) -> None:
        # OBS 30's property-list callback crashed in qsort/strcmp on this desktop.
        # Read only the chosen window's title without asking OBS to enumerate windows.
        result = subprocess.run(
            ["xprop", "-id", str(window_id), "-notype", "_NET_WM_NAME"],
            check=True, capture_output=True, text=True, timeout=5,
        )
        line = result.stdout.strip()
        if not line.startswith("_NET_WM_NAME = "):
            raise RuntimeError("Selected X11 window has no UTF-8 _NET_WM_NAME title")
        actual = json.loads(line.split(" = ", 1)[1])
        if actual != title:
            raise RuntimeError("The explicit X11 window ID/title no longer matches")

    def _verify_monitor(self, sinkmonitor: str) -> None:
        result = subprocess.run(
            ["pactl", "--format=json", "list", "sinks"],
            check=True, capture_output=True, text=True, timeout=5,
        )
        matches = [sink for sink in json.loads(result.stdout)
                   if sink["name"] == sinkmonitor.removesuffix(".monitor")
                   and sink["monitor_source"] == sinkmonitor]
        if len(matches) != 1:
            raise RuntimeError("Dedicated Pulse sink is absent; create and route it first")

    def setup_sources(self, window_id: int, title: str, sinkmonitor: str) -> dict[str, Any]:
        """Configure only the selected window and a dedicated Pulse monitor.

        :param window_id: Numeric X11 ID of the already opened game window.
        :param title: Exact window title expected for that ID.
        :param sinkmonitor: fmri_gym_record[optional_suffix].monitor; never a default device.
        :return: The verified selection saved for pre-recording checks.
        """
        if window_id <= 0 or not title or not re.fullmatch(MONITOR_PATTERN, sinkmonitor):
            raise ValueError("Require a positive X11 ID, title and fmri_gym_record*.monitor")
        self._assert_instance()
        self._assert_idle()
        existing = self.request("GetInputList")["inputs"]
        kinds = {VIDEO: "xcomposite_input", AUDIO: "pulse_output_capture"}
        if any(kinds.get(item["inputName"]) != item["inputKind"] for item in existing):
            raise RuntimeError("Unexpected inputs: use a fresh isolated configuration")
        names = {item["inputName"] for item in existing}
        self._verify_window(window_id, title)
        self._verify_monitor(sinkmonitor)
        # Empty selection captures an arbitrary window; ID-only selection avoids title fallback.
        self._configure_input(VIDEO, "xcomposite_input", {
            "capture_window": str(window_id), "show_cursor": False,
            "include_border": False, "exclude_alpha": True,
        }, names)
        self._configure_input(AUDIO, "pulse_output_capture", {"device_id": sinkmonitor}, names)
        self.request("SetInputAudioMonitorType", inputName=AUDIO,
                     monitorType="OBS_MONITORING_TYPE_NONE")
        self.request("SetInputVolume", inputName=AUDIO, inputVolumeMul=1.0)
        self.request("SetInputMute", inputName=AUDIO, inputMuted=False)
        self.request("SetInputAudioSyncOffset", inputName=AUDIO, inputAudioSyncOffset=0)
        self._enable_sources()
        selection = {"window_id": window_id, "title": title, "sinkmonitor": sinkmonitor,
                     "setup_monotonic_ns": time.monotonic_ns()}
        _json_write(self.root / "selection.json", selection)
        self._log("sources_ready", selection)
        return selection

    def _enable_sources(self) -> None:
        for name in (VIDEO, AUDIO):
            item = self.request("GetSceneItemId", sceneName=NAME, sourceName=name)["sceneItemId"]
            self.request("SetSceneItemEnabled", sceneName=NAME, sceneItemId=item,
                         sceneItemEnabled=True)
        self.request("SetCurrentProgramScene", sceneName=NAME)

    def _validate_sources(self) -> None:
        self._assert_instance()
        if self.request("GetCurrentProgramScene")["currentProgramSceneName"] != NAME:
            raise RuntimeError("The selected recording scene is no longer the program scene")
        selection = json.loads((self.root / "selection.json").read_text())
        items = self.request("GetSceneItemList", sceneName=NAME)["sceneItems"]
        if {item["sourceName"] for item in items} != {VIDEO, AUDIO} or len(items) != 2:
            raise RuntimeError("The scene must contain exactly the two selected sources")
        if not all(item["sceneItemEnabled"] for item in items):
            raise RuntimeError("Both selected sources must be enabled")
        self._verify_window(selection["window_id"], selection["title"])
        self._verify_monitor(selection["sinkmonitor"])
        expected = [(VIDEO, "capture_window", str(selection["window_id"])),
                    (AUDIO, "device_id", selection["sinkmonitor"])]
        for name, key, value in expected:
            actual = self.request("GetInputSettings", inputName=name)["inputSettings"]
            if actual.get(key) != value:
                raise RuntimeError(f"The {name} selection changed after setup")
        video = self.request("GetVideoSettings")
        required = {"baseWidth": self.config["width"], "baseHeight": self.config["height"],
                    "outputWidth": self.config["width"], "outputHeight": self.config["height"],
                    "fpsNumerator": self.config["fps"], "fpsDenominator": 1}
        if any(video[key] != value for key, value in required.items()):
            raise RuntimeError("OBS dimensions or fps differ from the prepared configuration")
        item = self.request("GetSceneItemId", sceneName=NAME, sourceName=VIDEO)["sceneItemId"]
        transform = self.request("GetSceneItemTransform", sceneName=NAME,
                                 sceneItemId=item)["sceneItemTransform"]
        if (transform["sourceWidth"], transform["sourceHeight"]) != (
            self.config["width"], self.config["height"],
        ):
            raise RuntimeError("Game window content dimensions must match the recording canvas")
        if (transform["scaleX"], transform["scaleY"]) != (1, 1):
            raise RuntimeError("Game capture must use original size without scaling")
        if any(transform[key] != 0 for key in (
            "positionX", "positionY", "rotation", "cropTop", "cropBottom", "cropLeft", "cropRight",
        )):
            raise RuntimeError("Game capture must have no crop, rotation or offset")

    def _wait_state(self, state: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            for index, event in enumerate(self.events):
                if (event["eventType"] == "RecordStateChanged"
                        and event["eventData"]["outputState"] == state):
                    return self.events.pop(index)["eventData"]
            self._receive(deadline)
        raise TimeoutError(f"OBS did not report {state}")

    def start(self) -> dict[str, Any]:
        """Explicitly start recording and wait for the STARTED event.

        :return: Recorder state and software acknowledgement timestamps.
        :raises RuntimeError: If source isolation or configuration checks fail.
        """
        self._assert_idle()
        selection = json.loads((self.root / "selection.json").read_text())
        # OBS 30 Pulse source drops its first 500 ms; settle before experiment markers.
        settle = 1 - (time.monotonic_ns() - selection["setup_monotonic_ns"]) / 1e9
        if settle > 0:
            time.sleep(settle)
        self._validate_sources()
        self.events.clear()
        sent = time.monotonic_ns()
        self.request("StartRecord")
        event = self._wait_state("OBS_WEBSOCKET_OUTPUT_STARTED")
        result = {"request_monotonic_ns": sent, "confirmed_monotonic_ns": time.monotonic_ns(),
                  "event": event, "status": self.request("GetRecordStatus")}
        self._log("record_started", result)
        return result

    def stats(self) -> dict[str, Any]:
        """Return and log OBS counters plus recording status.

        :return: Raw counters; render counters are cumulative, output counters reset per recording.
        """
        result = {"monotonic_ns": time.monotonic_ns(), "stats": self.request("GetStats"),
                  "record": self.request("GetRecordStatus")}
        self._log("stats", result)
        return result

    def stop(self) -> dict[str, Any]:
        """Stop recording cleanly, preserving the final filename and counters.

        :return: Saved output path, recorder state and final statistics.
        """
        self.events.clear()
        result = self.request("StopRecord")
        result["event"] = self._wait_state("OBS_WEBSOCKET_OUTPUT_STOPPED")
        result["final"] = self.stats()
        self._log("record_stopped", result)
        return result

    def close(self) -> None:
        """Close the control connection; it does not stop an ongoing recording."""
        if self.ws is not None:
            self.ws.close()
            self.ws = None

    def __enter__(self) -> OBSClient:  # noqa: PYI034 -- Python 3.10 has no typing.Self.
        return self.connect() if self.ws is None else self

    def __exit__(self, *_: object) -> None:
        self.close()


def main() -> None:
    """Expose preparation and explicit setup/start/stat/stop commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Write private config; do not launch or record")
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--width", type=int, required=True)
    prep.add_argument("--height", type=int, required=True)
    prep.add_argument("--fps", type=int, default=60)
    prep.add_argument("--encoder", choices=["libx264", "h264_nvenc"], default="libx264")
    prep.add_argument("--port", type=int, default=4457)
    prep.add_argument("--sample-rate", type=int, default=48000)
    setup = sub.add_parser("setup", help="Select only the game window and dedicated monitor")
    setup.add_argument("--window-id", type=lambda value: int(value, 0), required=True)
    setup.add_argument("--title", required=True)
    setup.add_argument("--sink-monitor", default="fmri_gym_record.monitor")
    sub.add_parser("start", help="Explicitly start recording")
    sub.add_parser("stop", help="Stop and finalize the recording")
    stat = sub.add_parser("stat", help="Read status/counters without starting recording")
    stat.add_argument("--count", type=int, default=1)
    stat.add_argument("--interval", type=float, default=1)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.state, args.output, args.width, args.height, args.fps,
                         args.encoder, args.port, args.sample_rate)
        launch = ["env", f"XDG_CONFIG_HOME={result['launch_env']['XDG_CONFIG_HOME']}"]
        result["launch_command"] = shlex.join(launch + result["launch_args"])
        print(json.dumps(result, indent=2))
        return
    with OBSClient(args.state) as client:
        if args.command == "setup":
            result = client.setup_sources(args.window_id, args.title, args.sink_monitor)
        elif args.command == "stat":
            if args.count < 1 or args.interval <= 0:
                raise ValueError("count and interval must be positive")
            for index in range(args.count):
                print(json.dumps(client.stats()), flush=True)
                if index + 1 < args.count:
                    time.sleep(args.interval)
            return
        else:
            result = getattr(client, args.command)()
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
