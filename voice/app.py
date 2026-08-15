"""Voice commands to real device actions, entirely on the device.

The default Woof demo uses a bundled ``Hey Wendy`` acoustic wake model to gate
local ASR. An optional ASR phrase mode can keep Parakeet active and match a
leading phrase without training another wake model. Nothing leaves the device.

    wake gate  ->  Parakeet ASR  ->  allowlisted dog action or MCP tool call

    wendy run
    open http://<device>:8080
"""

from __future__ import annotations

import asyncio
import os
import re
import threading
import time
import uuid

import httpx
import uvicorn
from asr import SherpaTranscriber
from capture import Capture
from collie_adapter import BorderCollieAdapter, interpret_command
from devices import list_input_devices, select_input_device
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from frontend import AudioFrontEnd
from mcpclient import MultiMCP
from microphone import run_microphone_session_loop
from model_cache import ensure_model
from observe_page import OBSERVE_HTML
from page import INDEX_HTML
from transcription_policy import should_transcribe
from utterance import UtteranceChunker
from wake_phrase import extract_wake_command
from wakeword import OpenWakeWordSpotter

MODEL_URL = os.environ.get(
    "MODEL_URL",
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2",
)
MODEL_DIR = os.environ.get("MODEL_DIR", "/models")
MODEL_NAME = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
MODEL_SHA256 = os.environ.get(
    "MODEL_SHA256",
    "5793d0fd397c5778d2cf2126994d58e9d56b1be7c04d13c7a15bb1b4eafb16bf",
)
PORT = int(os.environ.get("PORT", "8080"))
AUDIO_DEVICE = os.environ.get("AUDIO_DEVICE", "auto")
MICROPHONE_RETRY_INTERVAL_S = float(
    os.environ.get("MICROPHONE_RETRY_INTERVAL_S", "10.0")
)
MICROPHONE_FRAME_TIMEOUT_S = float(
    os.environ.get("MICROPHONE_FRAME_TIMEOUT_S", "3.0")
)
ACTION_MODE = os.environ.get("ACTION_MODE", "mcp")
BORDER_COLLIE_URL = os.environ.get("BORDER_COLLIE_URL", "http://127.0.0.1:8110")
BORDER_COLLIE_EXPECTED_BUILD_LABEL = os.environ.get(
    "BORDER_COLLIE_EXPECTED_BUILD_LABEL", ""
)
BORDER_COLLIE_EXPECTED_RELEASE_ID = os.environ.get(
    "BORDER_COLLIE_EXPECTED_RELEASE_ID", ""
)
_expected_schema = os.environ.get("BORDER_COLLIE_EXPECTED_CONFIG_SCHEMA", "").strip()
BORDER_COLLIE_EXPECTED_CONFIG_SCHEMA = int(_expected_schema) if _expected_schema else None
CONTINUOUS_TRANSCRIPTION = os.environ.get("CONTINUOUS_TRANSCRIPTION", "0") == "1"
ASR_WAKE_PHRASE = os.environ.get("ASR_WAKE_PHRASE", "").strip()
AUTO_ARM_ACTIONS = os.environ.get("AUTO_ARM_ACTIONS", "0") == "1"

# A path to a custom wake-word model, or a pretrained openWakeWord name
# ("hey_jarvis", "alexa", ...). The bundled "Hey Wendy" model was trained with
# wendylabsinc/wakeword-forge; train your own the same way.
WAKE_WORD = os.environ.get("WAKE_WORD", "/app/hey_wendy.onnx")
WAKE_THRESHOLD = float(os.environ.get("WAKE_THRESHOLD", "0.5"))
# How long after the wake word a command is still accepted.
COMMAND_WINDOW_S = float(os.environ.get("COMMAND_WINDOW_S", "8"))

# MCP servers. They run on this device (host networking): the MCP SDK rejects
# requests whose Host header is not localhost, so cross-device access needs a
# proxy rather than a different URL here.
MCP_URLS = [u.strip() for u in os.environ.get(
    "MCP_URLS", "http://127.0.0.1:3000").split(",") if u.strip()]
LLM_URL = os.environ.get("LLM_URL", "http://127.0.0.1:11434")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:3b")

SYSTEM_PROMPT = (
    "You control devices through the provided tools. When the user asks for "
    "something a tool can do, call that tool. Keep any reply to one sentence."
)


def build_app() -> FastAPI:
    ensure_model(
        model_dir=MODEL_DIR,
        model_url=MODEL_URL,
        expected_sha256=MODEL_SHA256,
        model_name=MODEL_NAME,
    )
    print("[asr] loading Parakeet...", flush=True)
    transcriber = SherpaTranscriber(MODEL_DIR, model_name="parakeet-tdt-0.6b")
    print("[asr] ready", flush=True)

    if CONTINUOUS_TRANSCRIPTION:
        spotter = None
        wake_key = "disabled"
        wake_mode = "disabled"
        print("[wake] disabled for continuous observation mode", flush=True)
    elif ASR_WAKE_PHRASE:
        spotter = None
        wake_key = ASR_WAKE_PHRASE
        wake_mode = "asr_phrase"
        print(f"[wake] always-listening ASR phrase: {wake_key!r}", flush=True)
    else:
        print(f"[wake] loading wake word {WAKE_WORD!r}...", flush=True)
        spotter = OpenWakeWordSpotter(WAKE_WORD, threshold=WAKE_THRESHOLD)
        wake_key = spotter.key
        wake_mode = "acoustic_model"
        print(f"[wake] ready: {wake_key}", flush=True)

    app = FastAPI()
    clients: set[WebSocket] = set()
    commands: asyncio.Queue = asyncio.Queue()
    wake_label = wake_key.replace("_", " ")
    always_transcribe = CONTINUOUS_TRANSCRIPTION or wake_mode == "asr_phrase"
    voice_state: dict[str, dict | None] = {
        "wake": None,
        "command": None,
        "action": None,
    }
    ready = {"ready": False}
    audio_state: dict = {
        "ready": False,
        "device": None,
        "error": "microphone initialization has not completed",
        "attempts": 0,
        "retry_count": 0,
        "retry_interval_s": MICROPHONE_RETRY_INTERVAL_S,
    }
    state: dict = {"loop": None}
    collie = (
        BorderCollieAdapter(
            BORDER_COLLIE_URL,
            expected_build_label=BORDER_COLLIE_EXPECTED_BUILD_LABEL,
            expected_release_id=BORDER_COLLIE_EXPECTED_RELEASE_ID,
            expected_config_schema=BORDER_COLLIE_EXPECTED_CONFIG_SCHEMA,
        )
        if ACTION_MODE == "border_collie"
        else None
    )
    if collie is not None:
        ready["ready"] = True

    async def broadcast(message: dict) -> None:
        for ws in list(clients):
            try:
                await ws.send_json(message)
            except Exception:  # noqa: BLE001 - disconnected browser is isolated
                clients.discard(ws)

    def voice_snapshot() -> dict:
        return {
            "wake_phrase": wake_label,
            "wake": dict(voice_state["wake"]) if voice_state["wake"] else None,
            "command": dict(voice_state["command"]) if voice_state["command"] else None,
            "action": dict(voice_state["action"]) if voice_state["action"] else None,
        }

    async def publish(message: dict) -> None:
        """Remember the latest voice interaction before sending it to browsers."""
        payload = dict(message)
        kind = payload.get("kind")
        if kind == "wake":
            voice_state["wake"] = payload
            voice_state["command"] = None
            voice_state["action"] = None
        elif kind == "command":
            voice_state["command"] = payload
            voice_state["action"] = None
        elif kind == "action":
            voice_state["action"] = payload
        await broadcast(payload)

    # -- audio thread -------------------------------------------------------

    def listen_once() -> None:
        try:
            devices = list_input_devices()
            device = select_input_device(AUDIO_DEVICE, devices)
            if device is None:
                available = ", ".join(f"[{item.index}] {item.name}" for item in devices)
                error = (
                    f"no microphone matched {AUDIO_DEVICE!r}; "
                    f"available inputs: {available or 'none'}"
                )
                audio_state.update(ready=False, device=None, error=error)
                return
        except Exception as exc:  # noqa: BLE001 - hardware discovery boundary
            audio_state.update(
                ready=False,
                device=None,
                error=f"microphone discovery failed: {exc}",
            )
            return
        print(f"[audio] using [{device.index}] {device.name}", flush=True)

        frontend = AudioFrontEnd(target_dbfs=-20.0)
        chunker = UtteranceChunker()
        capture = Capture(device)
        try:
            capture.start()
        except Exception as exc:  # noqa: BLE001 - PortAudio boundary
            audio_state.update(
                ready=False,
                device=device.name,
                error=f"microphone capture failed to start: {exc}",
            )
            print(f"[audio] waiting: {audio_state['error']}", flush=True)
            return
        audio_state.update(ready=True, device=device.name, error=None)
        armed_until = 0.0
        active_wake_id: str | None = None
        last_level_broadcast = 0.0

        if CONTINUOUS_TRANSCRIPTION:
            print(f"[web] continuously transcribing; open http://<device>:{PORT}", flush=True)
        else:
            print(f"[web] listening for '{wake_key}'; open http://<device>:{PORT}", flush=True)
        try:
            for frame in capture.frames(timeout_s=MICROPHONE_FRAME_TIMEOUT_S):
                # In acoustic mode the wake model runs on every raw frame and
                # keeps ASR idle until needed. ASR phrase mode intentionally
                # transcribes each completed utterance to find the configured phrase.
                now = time.monotonic()
                if spotter is not None and spotter.spot(frame):
                    armed_until = time.monotonic() + COMMAND_WINDOW_S
                    active_wake_id = uuid.uuid4().hex[:8]
                    print(f"[wake] heard '{wake_key}'", flush=True)
                    loop = state["loop"]
                    if loop is not None:
                        asyncio.run_coroutine_threadsafe(
                            publish({
                                "kind": "wake",
                                "id": active_wake_id,
                                "wake_phrase": wake_label,
                                "detected_at": time.time(),
                            }),
                            loop,
                        )

                normalised = frontend.process(frame)
                loop = state["loop"]
                if (CONTINUOUS_TRANSCRIPTION and loop is not None
                        and now - last_level_broadcast >= 0.2):
                    last_level_broadcast = now
                    asyncio.run_coroutine_threadsafe(
                        broadcast({"kind": "level",
                                   "input_dbfs": round(frontend.input_dbfs, 1)}), loop)
                utterance = chunker.process(normalised, level_dbfs=frontend.input_dbfs)
                if utterance is None:
                    continue
                if not should_transcribe(
                    continuous=always_transcribe,
                    now=time.monotonic(),
                    armed_until=armed_until,
                ):
                    continue

                result = transcriber.transcribe(utterance, 16000)
                raw_text = result.text.strip()
                if CONTINUOUS_TRANSCRIPTION:
                    command = raw_text
                elif wake_mode == "asr_phrase":
                    wake_command = extract_wake_command(raw_text, wake_label)
                    if wake_command is not None:
                        armed_until = time.monotonic() + COMMAND_WINDOW_S
                        active_wake_id = uuid.uuid4().hex[:8]
                        print(f"[wake] heard '{wake_label}' through ASR", flush=True)
                        loop = state["loop"]
                        if loop is not None:
                            asyncio.run_coroutine_threadsafe(
                                publish({
                                    "kind": "wake",
                                    "id": active_wake_id,
                                    "wake_phrase": wake_label,
                                    "detected_at": time.time(),
                                }),
                                loop,
                            )
                        command = wake_command
                    elif time.monotonic() <= armed_until:
                        command = raw_text
                    else:
                        continue
                else:
                    command = strip_wake_prefix(raw_text, wake_key)
                if not command:
                    # Just the wake phrase on its own: keep the window open for
                    # the command that follows rather than dispatching nothing.
                    if not CONTINUOUS_TRANSCRIPTION:
                        armed_until = time.monotonic() + COMMAND_WINDOW_S
                    continue
                if not CONTINUOUS_TRANSCRIPTION:
                    armed_until = 0.0
                event_kind = "transcript" if CONTINUOUS_TRANSCRIPTION else "command"
                print(f"[{event_kind}] {command}", flush=True)
                loop = state["loop"]
                if loop is None:
                    continue
                event = {
                    "kind": event_kind,
                    "id": uuid.uuid4().hex[:8],
                    "text": command,
                    "wake_id": active_wake_id,
                    "detected_at": time.time(),
                    "audio_ms": result.audio_ms,
                    "input_dbfs": round(frontend.input_dbfs, 1),
                }
                if collie is not None:
                    intent = interpret_command(command)
                    event["valid_dog_command"] = intent is not None
                    if intent is not None:
                        event["display_command"] = (
                            intent.target_fruit
                            if intent.action == "activate_demo"
                            else "stop"
                        )
                active_wake_id = None
                asyncio.run_coroutine_threadsafe(publish(event), loop)
                if not CONTINUOUS_TRANSCRIPTION and ready["ready"]:
                    asyncio.run_coroutine_threadsafe(
                        commands.put((event["id"], event["text"])), loop)
        except Exception as exc:  # noqa: BLE001 - capture thread must report failure
            audio_state.update(
                ready=False,
                error=f"microphone capture stopped unexpectedly: {exc}",
            )
            print(f"[audio] waiting: {audio_state['error']}", flush=True)
        finally:
            capture.stop()
            if audio_state["ready"]:
                audio_state.update(
                    ready=False,
                    error="microphone capture stopped",
                )

    def listen() -> None:
        last_retry_error: str | None = None

        def report_retry(error: str) -> None:
            nonlocal last_retry_error
            if error == last_retry_error:
                return
            last_retry_error = error
            print(
                f"[audio] retrying in {MICROPHONE_RETRY_INTERVAL_S:.2f}s: {error}",
                flush=True,
            )

        run_microphone_session_loop(
            session=listen_once,
            state=audio_state,
            retry_interval_s=MICROPHONE_RETRY_INTERVAL_S,
            sleep=time.sleep,
            on_retry=report_retry,
        )

    # -- LLM + MCP worker ---------------------------------------------------

    async def bridge() -> None:
        """Turn a spoken command into a real MCP tool call.

        The tool list comes from the MCP server itself, so the model can only
        call tools that genuinely exist.
        """
        mcp = MultiMCP(MCP_URLS)
        client = httpx.AsyncClient(timeout=120.0)
        try:
            tools = await asyncio.to_thread(mcp.refresh)
            for err in mcp.errors:
                print(f"[mcp] unavailable - {err}", flush=True)
            if not tools:
                print("[mcp] no tools discovered; commands will be shown but not acted on",
                      flush=True)
                return
            ready["ready"] = True
            print(f"[mcp] tools: {[t['name'] for t in tools]}", flush=True)
            schemas = mcp.to_ollama_tools()

            while True:
                cid, text = await commands.get()
                try:
                    resp = await client.post(f"{LLM_URL}/api/chat", json={
                        "model": LLM_MODEL,
                        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                     {"role": "user", "content": text}],
                        "tools": schemas,
                        "stream": False,
                    })
                    resp.raise_for_status()
                    message = resp.json().get("message", {}) or {}
                    calls = []
                    for call in message.get("tool_calls") or []:
                        fn = call.get("function", {}) or {}
                        name, args = fn.get("name"), fn.get("arguments") or {}
                        # Blocking HTTP; keep it off the event loop.
                        result = await asyncio.to_thread(mcp.call_tool, name, args)
                        calls.append({"tool": name, "args": args,
                                      "result": _text_of(result)})
                    await publish({"kind": "action", "id": cid, "calls": calls})
                except Exception as exc:  # noqa: BLE001 - optional MCP boundary
                    print(f"[mcp] '{text}' failed: {exc}", flush=True)
                    await publish({"kind": "action", "id": cid, "calls": [], "error": str(exc)})
        finally:
            await client.aclose()

    async def collie_bridge() -> None:
        """Dispatch a tiny phrase allowlist without an LLM or general robot tools."""
        assert collie is not None
        startup_mode = "auto-arm requested" if AUTO_ARM_ACTIONS else "starts disarmed"
        print(f"[collie] ready; {startup_mode}", flush=True)
        while True:
            cid, command_text = await commands.get()
            try:
                result = await asyncio.to_thread(
                    collie.dispatch,
                    command_text,
                    activation_id=f"voice-{cid}",
                )
                await publish({"kind": "action", "id": cid, **result})
            except Exception as exc:  # noqa: BLE001 - dog API boundary
                print(f"[collie] '{command_text}' failed: {exc}", flush=True)
                await publish({"kind": "action", "id": cid, "calls": [],
                               "error": str(exc)})

    async def auto_arm_collie() -> None:
        """Enable demo commands once both microphone and dog preflight are ready."""
        assert collie is not None
        last_error = None
        while not collie.armed:
            if not audio_state["ready"]:
                await asyncio.sleep(1.0)
                continue
            try:
                dog = await asyncio.to_thread(collie.arm)
                print(
                    f"[collie] automatically armed for {dog['build_label']}",
                    flush=True,
                )
                return
            except RuntimeError as exc:
                error = str(exc)
                if error != last_error:
                    print(f"[collie] waiting to auto-arm: {error}", flush=True)
                    last_error = error
                await asyncio.sleep(2.0)

    @app.on_event("startup")
    async def _startup() -> None:
        state["loop"] = asyncio.get_running_loop()
        threading.Thread(target=listen, name="listen", daemon=True).start()
        # A background task, so a missing MCP server or LLM never blocks the page.
        if ACTION_MODE != "observe":
            asyncio.create_task(collie_bridge() if collie is not None else bridge())
        if collie is not None and AUTO_ARM_ACTIONS:
            asyncio.create_task(auto_arm_collie())

    @app.get("/healthz")
    async def _healthz() -> dict:
        return {"ok": bool(audio_state["ready"]),
                "wake_word": wake_key, "wake_mode": wake_mode,
                "always_listening": wake_mode in {"acoustic_model", "asr_phrase"},
                "auto_arm_actions": AUTO_ARM_ACTIONS,
                "tools_ready": ready["ready"],
                "clients": len(clients), "action_mode": ACTION_MODE,
                "actions_armed": collie.armed if collie is not None else None,
                "continuous_transcription": CONTINUOUS_TRANSCRIPTION,
                "microphone": dict(audio_state),
                "voice": voice_snapshot()}

    @app.get("/api/voice/status")
    async def _voice_status() -> dict:
        return voice_snapshot()

    @app.get("/api/actions/status")
    async def _action_status() -> dict:
        dog = await asyncio.to_thread(collie.readiness) if collie is not None else None
        return {"mode": ACTION_MODE,
                "armed": collie.armed if collie is not None else None,
                "auto_arm_actions": AUTO_ARM_ACTIONS,
                "always_listening": wake_mode in {"acoustic_model", "asr_phrase"},
                "microphone": dict(audio_state),
                "dog": dog}

    @app.post("/api/actions/arm")
    async def _arm_actions() -> dict:
        if collie is None:
            return {"mode": ACTION_MODE, "armed": None}
        if not audio_state["ready"]:
            raise HTTPException(
                status_code=503,
                detail=str(audio_state["error"] or "microphone is unavailable"),
            )
        try:
            dog = await asyncio.to_thread(collie.arm)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        print("[collie] voice actions armed from the local UI", flush=True)
        return {"mode": ACTION_MODE, "armed": True,
                "microphone": dict(audio_state), "dog": dog}

    @app.post("/api/actions/disarm")
    async def _disarm_actions() -> dict:
        if collie is None:
            return {"mode": ACTION_MODE, "armed": None}
        collie.disarm()
        dog = await asyncio.to_thread(collie.readiness)
        print("[collie] voice actions disarmed", flush=True)
        return {"mode": ACTION_MODE, "armed": False,
                "microphone": dict(audio_state), "dog": dog}

    @app.get("/", response_class=HTMLResponse)
    async def _index() -> str:
        if CONTINUOUS_TRANSCRIPTION:
            return OBSERVE_HTML
        wake_text = wake_key.replace("_", " ")
        return (
            INDEX_HTML
            .replace("{{WAKE}}", wake_text)
            .replace("{{WAKE_TITLE}}", wake_text.title())
        )

    @app.websocket("/ws")
    async def _ws(websocket: WebSocket) -> None:
        await websocket.accept()
        clients.add(websocket)
        try:
            await websocket.send_json({"kind": "snapshot", "voice": voice_snapshot()})
            while True:
                await websocket.receive_text()
        except (WebSocketDisconnect, Exception):  # noqa: BLE001
            clients.discard(websocket)
        finally:
            clients.discard(websocket)

    return app


def strip_wake_prefix(text: str, key: str) -> str:
    """Remove a leading wake phrase from a transcript.

    The utterance chunker buffers continuously, so a command spoken in one
    breath arrives as "hey jarvis turn the light red". The model copes better,
    and the page reads better, with the wake phrase removed.
    """
    words = [w for w in re.split(r"[^a-z0-9]+", key.lower()) if w and not w.isdigit()]
    if not words:
        return text.strip()
    pattern = r"^\W*" + r"\W+".join(re.escape(w) for w in words) + r"\W*"
    return re.sub(pattern, "", text.strip(), flags=re.IGNORECASE).strip()


def _text_of(result) -> str:
    if isinstance(result, dict):
        parts = [i["text"] for i in (result.get("content") or [])
                 if isinstance(i, dict) and i.get("text")]
        if parts:
            return " ".join(parts).strip()
    return str(result) if result else "done"


if __name__ == "__main__":
    uvicorn.run(build_app(), host="0.0.0.0", port=PORT, log_level="warning")
