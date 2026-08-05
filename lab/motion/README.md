# Guarded forward-pulse acceptance

This is the only live movement exposed by the clean foundation. It uses
Unitree's factory `ObstaclesAvoidClient` remote-command path.

Do not enable it until Woof is on the floor, clear of people, edges, cables,
and obstacles, with a human holding the physical remote.

Required deployment gates:

```text
BORDER_COLLIE_HARDWARE_ENABLED=1
BORDER_COLLIE_LAB_MOTION_ENABLED=1
```

Confirm `/api/status` reports connected hardware, fresh pose, disarmed motion,
factory avoidance available, and `can_pulse_forward: true`.

Then issue exactly one request:

```bash
curl -X POST http://woof.local:8110/api/hardware/forward-pulse \
  -H 'content-type: application/json' \
  -d '{"confirmation":"PATH CLEAR - MOVE WOOF FORWARD"}'
```

Acceptance requires visible forward movement, automatic stop, `armed: false`,
zero final velocity, no retained remote API ownership, and a compact result plus
snapshot. Any physical remote input should be treated as manual takeover even
though automatic remote-input detection remains a future adapter.
