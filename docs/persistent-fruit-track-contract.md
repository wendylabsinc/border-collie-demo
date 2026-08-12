# Persistent fruit-track contract

Release `stage-camera-v29-persistent-fruit-track` treats a detector result as
an observation of one mission-lifetime track, not as the track itself.

## Public seam

`PersistentFruitTracker.observe(status, now_s)` consumes a full-frame
observation and an optional crop refinement, then returns a `FruitTrackReport`.
The report independently states:

- identity state and acquisition epoch;
- raw and filtered confidence;
- source freshness and generation;
- geometry route and whether geometry is safe;
- forward, alignment, and bounded-hold authority; and
- Arrival eligibility, counter, and reason.

The states are `UNSEEN`, `CANDIDATE`, `LOCKED`, `DEGRADED`, `LOST`, and
`LOCKED_OFF_AXIS`. Search and approach receive the same tracker instance from
`StageContext`; approach adopts its locked identity while the existing
qualified geometry controller retains steering and Arrival policy.

## Authority rules

- Full-frame evidence is authoritative for identity on every processed frame.
- An agreeing crop can refine confidence/geometry, but cannot acquire identity
  without a full-frame hit and cannot erase a full-frame hit when it misses.
- Acquisition uses the existing fruit threshold and consecutive-frame count.
- Maintenance uses the existing tracking floor plus an EMA for retained
  evidence. One weak or missing fresh frame enters `DEGRADED`; recovery within
  250 ms returns to the same acquisition epoch.
- A second fresh miss, or expiration of the 250 ms deadline, enters `LOST`.
- Off-axis geometry keeps identity `LOCKED_OFF_AXIS`, removes forward
  authority, and permits bounded alignment.
- Duplicate or degraded evidence cannot advance Arrival. The second confirmed
  fresh loss may complete only a Final Approach latch established earlier by
  fresh qualified close geometry.
- Stale evidence, camera failure, generation change, confirmed wrong identity,
  and impossible geometry stop immediately. Existing authority leases,
  watchdogs, disarm, recovery, and takeover gates are unchanged.

## Black box and replay

Every processed frame records full-frame and crop observations, raw and EMA
confidence, boxes/routes, state before/after, authority/reason, and the motion
command produced from that frame. `FruitTrackReplay` runs saved
`perception_sample` events through the production tracker and reports phase
identity resets, degraded recoveries, off-axis preservation, loss timing, and
degraded Arrival violations.

`scripts/replay_v28_fruit_tracks.py` covers all five saved v28 physical records
and retains their exact command counters. Those records were sampled at API
cadence and did not retain every detector frame or separate full/crop routes,
so the replay is intentionally labelled sparse. Exact post-change dropout and
zero-command counts require a new v29 physical run.

## Integration order

The runtime-environment branch at commit `0468bee` changes many of the same
configuration, hardware, and production files. It was not cherry-picked into
this tracker slice because doing so would obscure the identity change and
expand its deployment risk. Rebase that runtime-environment work on top of v29
after this tracker is reviewed and physically characterized.
