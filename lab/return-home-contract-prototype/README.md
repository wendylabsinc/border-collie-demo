# Return-to-Home contract prototype

**Question:** Can one bounded state model make it impossible to report return
success without fresh pose, measured position/heading tolerance, collision-safe
progress, and confirmed disarm?

This is throwaway, in-memory logic. It does not connect to Woof or import a
motion client. The numeric configuration is illustrative until WDY-2281 and
WDY-2283 qualify the Operating Envelope and motion primitives.

Run it with:

```bash
python3 lab/return-home-contract-prototype/prototype.py
```

The reusable answer is the state model in `return_model.py`; the terminal shell
exists only to push it through success and failure cases.
