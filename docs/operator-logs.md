# Operator logs

The app projects high-value events from the durable per-run black box into
compact `demo_event` JSON lines at INFO level. These include:

- Demo Run activation and mission-state transitions;
- every motion command actually sent, including control path, forward speed,
  yaw speed, and reason;
- bounded stage summaries and Home retries/verifications;
- failure epilogue and terminal outcome, reason, failed phase, and final safety
  state.

Routine Uvicorn request-access lines are disabled for the app and media
sidecar. The durable black box, HTTP status, and camera endpoints are unchanged.
The operator projection deliberately omits per-frame guidance and large payloads
so `wendy device logs` remains readable.

`BORDER_COLLIE_OPERATOR_LOG_ENABLED` accepts `0` or `1` and defaults to `1`.
It changes only live INFO projection; setting it to `0` never disables durable
black-box recording or any motion-safety evidence. Units do not apply. Restart
only the app process after changing it; media does not need rebuilding.

Example lines:

```text
demo_event {"run_id":"...","sequence":11,"event":"state","phase":"approach_fruit","reason":"APPROACH_FRUIT_STARTED","message":"approach_fruit started"}
demo_event {"run_id":"...","sequence":52,"event":"motion","phase":"return_home","command_sequence":7,"motion_path":"factory_avoidance","forward_mps":1.0,"yaw_rps":0.0,"reason":"return_home_position"}
demo_event {"run_id":"...","sequence":91,"event":"terminal","phase":"failed","outcome":"FAILED","reason":"RETURN_HOME_FAILURE","failed_phase":"return_home","final_safety_state":"DISARMED_CONFIRMED"}
```
