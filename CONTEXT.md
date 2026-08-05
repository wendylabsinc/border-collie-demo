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
The validated near-fruit condition reached without requiring physical contact. It permits at most one bounded final approach after trustworthy lower-edge disappearance and ends with Woof stopped safely.
_Avoid_: Fruit contact, fruit disappearance

**Demo Run**:
One attempt beginning with preflight and fresh Home capture and ending in completion, failure, stop, or Remote Takeover.
_Avoid_: Session, mission

**Run Result**:
The durable terminal record of a Demo Run, including its outcome, final phase, reason, and relevant terminal measurements. A stale or frozen camera must produce a failed Run Result with reason `CAMERA_FAILURE`.
_Avoid_: Console output, debug message

**Remote Takeover**:
The process-latched terminal state caused by valid physical remote input. It stops autonomous output immediately and requires an application restart before another Demo Run.
_Avoid_: Pause, manual mode
