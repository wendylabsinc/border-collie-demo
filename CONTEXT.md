# Border Collie Demo

Language for the stage routine in which Woof receives a fruit request, reaches the requested fruit, performs the audience action, and returns Home.

## Language

**Target Fruit**:
The fruit named by the command for the current Demo Run.
_Avoid_: Selected object, goal object

**Qualified Fruit**:
A fruit with current end-to-end acceptance evidence for the complete Demo Run. Pear is temporarily the only Qualified Fruit.
_Avoid_: Supported fruit, detectable fruit

**Supported Fruit**:
A fruit the system may recognize or represent, but which is not necessarily qualified for the complete Demo Run.
_Avoid_: Qualified fruit, working fruit

**Operating Envelope**:
The controlled physical and network conditions under which the Demo Run is qualified to operate. A change to any qualified condition requires the affected acceptance evidence to be repeated.
_Avoid_: Test setup, ideal conditions

**Home**:
The fresh position and heading captured immediately before a Demo Run. It is not a location inferred from the person or a fixed room coordinate.
_Avoid_: Starting area, person-relative position

**Home Distance**:
The planar distance in meters between Woof's latest fresh measured position and Home. It is unavailable rather than estimated when current pose data is not trustworthy.
_Avoid_: Distance traveled, estimated route length

**Arrival**:
The stage-camera profile's common apple, banana, and pear closeout: three fresh
centered near-geometry samples establish Final Approach, Woof continues at the
close-range speed until questionable evidence revokes motion. Two fresh,
advancing, camera-healthy weak or missing frames within the close-loss window
confirm the qualified lower-edge exit, then Woof executes one bounded 0.6 m/s,
1.0 second final push and stops. Duplicate or stale evidence cannot advance
loss confirmation. Wrong identity/generation, camera failure, off-axis loss,
invalid geometry, expired evidence, or an unqualified loss cannot establish
Arrival. The experimental metric/LiDAR profile remains a separate fail-closed
mode and is not enabled by the stage descriptor.
_Avoid_: Fruit contact, fruit disappearance

**Demo Run**:
One attempt beginning with preflight and fresh Home capture and ending in completion, failure, stop, or Remote Takeover.
_Avoid_: Session, mission

**Run Result**:
The durable terminal record of a Demo Run, including its outcome, final phase, reason, and relevant terminal measurements. A stale or frozen camera must produce a failed Run Result with reason `CAMERA_FAILURE`.
_Avoid_: Console output, debug message

**Recoverable Failure**:
An approach failure that leaves motion, pose, captured Home, and posture control trustworthy. Woof records the failed attempt, lies down without barking, stands, and performs one bounded position-only return to the original Home. The Demo Run remains failed and the recovery receives its own outcome.
_Avoid_: Success, retry, resumed run

**Remote Takeover**:
The process-latched terminal state caused by valid physical remote input. It stops autonomous output immediately and requires an application restart before another Demo Run.
_Avoid_: Pause, manual mode
