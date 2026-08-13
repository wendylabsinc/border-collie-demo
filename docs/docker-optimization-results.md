# Docker optimization results

## DLO-PROOF-001 — verified and applied

Observed on 2026-08-03 using DLO 0.5.0b1 and BuildKit with a local image load.
The original 21.214-second observation remains a cold baseline and is not used
as proof of improvement.

The verified candidate stopped reinstalling application source into the builder
virtual environment. Dependencies remain in their stable builder layer;
`src/` is copied into the final image and imported through `/app/src`. Entrypoint,
base image, published port, and hardware safety defaults were unchanged.

| Measurement | Control | Candidate | Result |
| --- | ---: | ---: | ---: |
| Source-edit median, 3 paired trials | 8.743 s | 0.598 s | 8.145 s / 93.16% faster |
| Source-edit p95 | 8.967 s | 0.604 s | no regression |
| Median cached steps | 6 | 8 | +2 cached |
| Median rebuilt steps | 5 | 2 | -3 rebuilt |
| Warm no-op build | 0.502 s | 0.497 s | no regression |
| Dependency-manifest edit | 22.169 s | 8.141 s | no regression |

Verification took 91.301 seconds, giving an estimated break-even of 11.2
representative source-edit deployments. Every configured gate passed: at least
10% and 0.5 seconds of median improvement, no material p95/no-op/dependency
regression, correctness, protected-path safety, budget, and payback within 20
deployments.

The applied image passed the then-current 27 project tests and a container check proving the
installed command resolves `border_collie_demo.api` from `/app/src/`.

## DLO observer overhead

Three additional unchanged builds measured DLO separately from Docker:

| Trial | Docker build | DLO wrapper | Non-build overhead |
| --- | ---: | ---: | ---: |
| 1 | 0.954 s | 0.976767 s | 0.021603 s |
| 2 | 0.536 s | 0.557646 s | 0.020484 s |
| 3 | 0.547 s | 0.569921 s | 0.021611 s |
| **Median** | **0.547 s** | **0.569921 s** | **0.021603 s** |

The measured observer cost is about 3.95% of this very short no-op build and
0.27% of the 8.145-second per-source-edit saving. It therefore does add a small
cost, but the verified Dockerfile improvement is much larger.

## Layer evidence and remaining work

The controlled proof compares rebuilt steps, not elapsed time alone. The final
unchanged image reports 10/10 reused layers, zero new layers, and zero rebuilt
steps. The post-application build reports a 227,625,311-byte image.

The local-load proof records changed layer identities and counts but does not
report changed layer byte totals for each paired source trial. That is an
explicit instrumentation gap rather than evidence that no bytes changed. A
future registry-backed deployment measurement must capture unmatched compressed
bytes and split total deployment time into build, transfer, replacement, and
readiness before claiming equivalent improvement for Woof deployment.

## Production observation: Home capture slice

The first normal post-proof build included the Home capture source change and a
README update. It took 23.949 seconds with 4 cached, 6 rebuilt, and 1 resolved
step; DLO added 0.023543 seconds outside Docker. The 10-layer image reused 6
layers and introduced 4, with a final size of 227,626,218 bytes.

The source-only layout remained correct, but `COPY pyproject.toml README.md ./`
caused the README edit to invalidate the expensive dependency installation.
DLO ranked that copy as the highest-cost volatile instruction. Removing README
from the builder is semantically safe because package metadata does not refer to
it, but DLO 0.5.0b1 cannot use Markdown as a paired benchmark mutation target.
The candidate therefore remains unapplied until the optimizer can prove this
class of documentation-only change.

A second normal build for the perception readiness adapter confirmed the same
pattern: 22.009 seconds, 4 cached and 6 rebuilt steps, 6/10 reused layers, and
0.025463 seconds of DLO overhead. This reinforces the README invalidation
finding rather than a one-off cold-cache result.

## DLO-PROOF-002 — verified and applied

DLO added a semantics-safe Markdown benchmark mutation and proved the focused
candidate `COPY pyproject.toml ./` against three paired README-only edits. The
first two attempts exposed a Colima portability bug: containerized verification
could not bind-mount snapshots created under macOS `/private/tmp`. DLO moved
disposable snapshots into its shared user-cache directory; they are still
deleted immediately after each verification run.

| Measurement | Control | Candidate | Result |
| --- | ---: | ---: | ---: |
| README-edit median, 3 paired trials | 19.721 s | 0.391 s | 19.330 s / 98.02% faster |
| README-edit p95 | 20.248 s | 0.431 s | no regression |
| Median cached steps | 4 | 10 | +6 cached |
| Median rebuilt steps | 6 | 0 | -6 rebuilt |
| Warm no-op build | 0.396 s | 0.400 s | no regression |
| Dependency-manifest edit | 19.726 s | 20.376 s | within configured tolerance |

Verification took 109.366 seconds, giving an estimated break-even of 5.7
representative README-edit deployments. Every configured performance, payback,
protected-change, project-test, and container-runtime gate passed, so DLO
automatically applied candidate `f53c57a25746d2fb56a2`. Package metadata does
not reference README; runtime behavior and dependency invalidation remain
unchanged.

## DLO-PUSH-GATE-001 — local would-push comparison

Measured on 2026-08-03 before any production image publication or Woof
deployment. DLO pushed only to an isolated, disposable registry on
`localhost:5007`; neither comparison image was sent to Docker Hub, the Wendy
registry, or Woof.

The control is the preserved pre-DLO source-install layout in
`benchmarks/docker/Dockerfile.pre-dlo-control`. The candidate is the current
production `Dockerfile`. Both were built from the same working tree and warm
BuildKit cache. DLO inspected the compressed OCI blobs declared by the local
registry manifests.

| Measurement | Without DLO layout | With verified DLO layout | Difference |
| --- | ---: | ---: | ---: |
| Full image compressed layers (empty-tag upper bound) | 227,665,785 B | 227,618,821 B | 46,964 B smaller |
| Full-image build and local push | 7.210 s | 0.935 s | 6.275 s faster |
| One-file source edit: unmatched compressed blobs | 99,567,096 B | 21,657 B | 99,545,439 B / 99.978% less |
| One-file source edit: build and local push | 7.070 s | 0.446 s | 6.624 s / 93.69% faster |
| One-file source edit: rebuilt steps | 5 | 2 | 3 fewer |
| One-file source edit: matching layers | 6/9 | 8/10 | 2 more reused |
| DLO observer cost on the edit measurement | 0.107408 s | 0.109579 s | effectively equal |

The source-edit probe was a deterministic temporary Python file added after
both baseline manifests existed, then removed immediately after the paired
measurement. The control embedded application source into the approximately
99.6 MB compressed virtual-environment layer, so that entire blob changed. The
DLO layout kept that dependency blob stable and transferred only the small
source and web tail layers.

The full-image row is the conservative answer for a registry with none of the
image's blobs. The source-edit row is the representative answer for a registry
that already has the preceding release. DLO's `unmatched_compressed_bytes` is a
deterministic manifest comparison and an upper-bound-style estimate, not a
packet-level network measurement; a registry may already contain some blobs.
The production push gate remains closed until the application work and tests
are complete.

## DLO-WOOF-DEPLOY-001 — partial production-device comparison

Measured on 2026-08-03 with DLO 0.5.0b1 wrapping Wendy deployments to Woof.
Both layouts used Wendy's Docker builder, forced content-defined chunking, the
same `border-collie-demo` app identity, ARM64 target, host-network entitlement,
and hardware/lab-motion disabled. Each successful deployment reached TCP
readiness and the API returned an idle mission with hardware unconfigured.

The first three trials alternated layouts. Their medians were effectively tied
(66.832 seconds control and 67.026 seconds candidate), but they are excluded as
proof: Wendy retains the immediately preceding layout's cache path, so every
alternation paid a layout-switch rebuild. A subsequent consecutive candidate
source edit completed in 11.117 seconds, confirming that the alternating design
was not representative of normal production development.

The corrected benchmark warmed one layout and then made consecutive source-only
edits. The user stopped the matrix after the complete three-run control block
and before the candidate block, so this is deliberately marked partial.

| Phase | Control median, 3 source edits | Candidate, 1 preliminary source edit |
| --- | ---: | ---: |
| End to end | 69.641 s | 11.117 s |
| Build | 42.697 s | 8.879 s |
| Export | 18.892 s | 0.164 s |
| Device transfer marker | 0.012 s | 0.025 s |
| Device unpack | 5.852 s | 0.045 s |
| Replacement | 0.671 s | 0.422 s |
| Readiness | 1.165 s | 1.166 s |

Control end-to-end observations were 70.347, 69.641, and 68.682 seconds. The
single consecutive candidate observation was 58.524 seconds (84.03%) faster
than the control median, but one candidate observation is not enough to claim a
verified Woof speedup. The remaining candidate trials can be run later without
repeating the completed control block if the same device and deployment path
remain representative.

Two CLI setup failures were excluded: an alternate Dockerfile path containing
directory separators was rejected, and the installed Wendy build advertised
but did not support the `buildkit` builder. Wendy also ignored an alternate
root Dockerfile selector when multiple build files existed. The benchmark then
used the supported `docker` builder and temporarily swapped the root Dockerfile
contents, verifying the production digest after every restore.

After stopping, the temporary source probe was removed and the clean verified
candidate was redeployed. Woof is running `border-collie-demo` 0.1.0 at port
8110; its API is healthy, hardware motion is disabled, and all 45 repository
tests pass. The final layout-switch deployment is excluded from performance
statistics.

## Media image follow-up — 2026-08-03

The completed demo now adds a separate Jetson GPU media image containing the
validated 46 MB TensorRT pear engine. DLO `0.5.0b3` captured a 48,529,651-byte
context and began the intentionally unoptimized control warm-up, but Docker's
content store ran out of space while loading the `dustynv/pytorch` base. The
failed attempt took 25.382 seconds and produced no usable control/candidate
pair. No image was pushed and nothing was deployed to Woof.

This is explicitly **not** evidence that either layout is faster. Before the
next deployment, reclaim Docker storage with the operator's approval, warm both
layouts, run paired source-only edits plus no-op/dependency controls, and record
the resulting medians, p95 values, rebuilt steps, and changed layer bytes.

## DLO-WOOF-DEPLOY-002 — completed app redeploy comparison

After DLO was updated, the application service was compared again on Woof with
DLO 0.5.0b3. The preserved pre-DLO control and current candidate each received
three consecutive source-only edits after their own warm-up. Every trial used
Wendy's Docker builder, forced content-defined chunking, and the same production
device and service configuration. Warm-ups and layout switches are excluded.

| Measurement | Without DLO layout | With DLO layout | Difference |
| --- | ---: | ---: | ---: |
| Source-edit median, 3 trials | 10.277 s | 9.321 s | 0.956 s / 9.30% faster |
| Nearest-rank p95 (three-sample maximum) | 10.643 s | 9.403 s | 1.240 s better |
| Median build phase | 8.330 s | 8.270 s | 0.060 s faster |
| Candidate no-op redeploy | — | 8.992 s | no unexpected regression observed |

The absolute improvement exceeds DLO's 0.5-second gate and the small previously
measured observer cost. The relative improvement is 9.30%, just below the strict
10% gate, so this device sample is **promising but not a new strict proof**. It
does not overturn the earlier controlled layer proof; it shows that Wendy's
roughly eight-second fixed build/deployment path dominates this now-small app.

The temporary source marker was removed before the final candidate redeploy,
which completed in 9.233 seconds. Woof then reported a production runtime,
healthy hardware connection, idle mission, and no active motion. All 70 tests,
production-source Ruff checks, and bytecode compilation passed. The app remains
deployed with the verified DLO Dockerfile.

The media sidecar was also retried. Wendy 2026.07.27-003050 exposes `buildkit`
in help but rejected it at runtime as unsupported before any build began. The
Docker fallback remains blocked by local content-store capacity for the large
Jetson GPU image. Consequently the app's activation preflight correctly remains
blocked on camera/perception and bark readiness; no demo motion was started.

## DLO-MEDIA-DEPLOY-001 — deployed with registry fallback

The local Colima Docker volume was expanded from 20 GiB to 60 GiB without
deleting images, volumes, or caches. This removed the capacity blocker and
allowed the 52-layer Jetson/PyTorch media image to build. Woof reused 46 base
snapshots on the initial deployment and applied only the six media tail layers.

Repeated forced chunk-diff attempts then exposed a separate Wendy limit:
`receiving container output: grpc: received message larger than max (5560979
vs. 4194304)`. Representative failed attempts completed their build and diff
work in roughly 25 seconds, including about 21.9 seconds of build/export and
1.1 seconds of chunk query/write, but they are not successful benchmark trials.

The oversized payload was associated with repeated recoverable
`aiortc.codecs.h264` decoder warnings. The media runtime now raises that specific
logger to `ERROR` before WebRTC startup. A regression test emits 50,000 matching
warnings and verifies output remains below Wendy's 4 MiB ceiling. All 71 project
tests, production Ruff checks, and bytecode compilation pass.

Because Wendy's device-side output backlog still exceeded the RPC ceiling, the
final DLO deployment explicitly selected Wendy's registry path. It succeeded:

| Phase | Duration |
| --- | ---: |
| End to end | 160.313 s |
| Build/export | 143.770 s |
| Device unpack | 15.685 s |
| Replacement | 0.301 s |
| Transfer marker | 0.011 s |

The deployed sidecar loaded the TensorRT engine, connected to Woof over WebRTC,
reported advancing 1280×720 source frames, and reported bark ready. The mission
remained idle and no motion was activated. Activation is currently blocked only
because no qualifying pear is visible. This is a successful deployment result,
not a control/candidate optimization proof; the faster chunk-diff observations
remain excluded until a complete deployment can pass the gRPC boundary.

After that successful health check, Woof became unreachable on ports 8110,
8111, and 50052 as well as ICMP. The final connectivity check therefore cannot
assert that the verified services are still running. No additional deployment
or motion command was issued while the device was offline.
