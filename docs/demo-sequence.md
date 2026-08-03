# Demo sequence and safety states

The production sequence is:

```text
IDLE -> PREFLIGHT -> CAPTURE_HOME -> WAIT_FOR_COMMAND
     -> TURN_TO_FRUIT -> FIND_FRUIT -> APPROACH_FRUIT -> ARRIVED
     -> SIT_AND_BARK -> STAND -> TURN_TOWARD_HOME -> RETURN_HOME
     -> RESTORE_HEADING -> COMPLETE
```

`STOPPED`, `FAILED`, and `REMOTE_TAKEOVER` are off-ramps from active work.

`REMOTE_TAKEOVER` is latched for the lifetime of the process. Detection of any
valid physical remote input must stop application command output, release the
motion owner, invalidate the run, record the interrupted phase, and require an
application restart. It cannot be cleared through the web UI or API.
