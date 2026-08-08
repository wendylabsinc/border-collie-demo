# MAX versus TensorRT fruit-perception retrospective

Date: 2026-08-07

## Decision

Keep TensorRT as the production fruit-perception runtime on Woof. Do not connect
the current MAX candidates to autonomous motion.

MAX 26.4 can execute native `sm_87` GPU code on Woof, but none of the usable
fruit models met the demo's accuracy, latency, memory, and operational gates.
The best equivalent MAX candidate was YOLO11n at 416 px: it ran at 841.45 ms
median (1.19 FPS), approximately 14 times slower than the observed TensorRT
median and 8.4 times slower than the 100 ms deadline. In its first clear live
pear test, it localized the fruit but ranked it as apple (0.637) rather than pear
(0.249). Local PyTorch produced the same wrong ordering, so that live accuracy
failure belongs primarily to the trained nano model, not the MAX importer.

The fast 66.03 ms MAX result was an untrained 1.25-million-parameter skeleton
with random weights and incomplete postprocessing. It proved that a small native
MAX graph can run quickly; it did not prove that an accurate fruit detector can.

## Runtime comparison

| Aspect | TensorRT control | Best usable MAX candidate | Other MAX evidence | Verdict |
| --- | ---: | ---: | ---: | --- |
| Model | Production general detector plus banana specialist | Trained YOLO11n, 416 px | Trained YOLO11s and original/full graph | TensorRT |
| Median inference | 60.22 ms | 841.45 ms | YOLO11s 416: 2,822.20 ms; full 640: 11,851.05 ms | TensorRT |
| Throughput | About 16.6 inference FPS; camera observed at 14.05 FPS | 1.19 FPS | 0.354 FPS and 0.084 FPS | TensorRT |
| Relative latency | 1.0x | 14.0x TensorRT | 46.9x and 196.8x TensorRT | TensorRT |
| 100 ms motion gate | Passed in the recorded control | Failed by 8.4x | Failed by 28.2x and 118.5x | TensorRT |
| Live fruit behavior | Repeatedly validated in end-to-end pear/apple/banana work | Clear pear ranked as apple | Earlier MAX live scene was inconclusive | TensorRT |
| Held-out accuracy | Exact production-model corpus comparison not recorded | mAP50 0.775, mAP50-95 0.620 | Same-data YOLO11s: 0.843 and 0.657 | Larger model |
| Status responsiveness | 15.94 ms median, 20.64 ms p95 at normal polling | Not integrated into the full status/media service | Long inference cannot satisfy current freshness behavior | TensorRT |
| Warm/cached startup | Not recorded in the control | 33.39 s for the nano MAX app | YOLO11s: 102.03 s; full cached run: 418.59 s including warm/timed execution | TensorRT operationally |
| Runtime memory | Wendy app group: 1.935 GB median | Process RSS about 0.54 GB, but host availability settled near 2.05 GiB because GPU/unified allocations are not represented by RSS | YOLO11s compile peaked at 8.50+ GiB RSS; full compiles reached 13+ GiB | TensorRT |
| Compile behavior | Existing engine is already deployable | About 333 s; peak RSS 8.65 GiB | Repeated 11-30 minute compiles, guard aborts, and one host OOM that killed services | TensorRT |
| Model artifacts | General engine 48.5 MB; specialist checkpoint 19.2 MB | MEF 6.37 MB plus weights 5.25 MB | YOLO11s MEF 6.43 MB plus weights 18.91 MB | MAX artifact size |
| Container/package | Existing production image | New MAX candidate export moved several GB and took about 14 minutes on first deployment | Earlier MAX candidate image was 2.10 GB | TensorRT operationally |
| Thermal behavior | Short GPU bursts observed | Guarded runs stayed within thermal limits | MAX did not prove to be the direct overheating cause; duration, RAM pressure, and battery use were the demonstrated problems | Tie on safety only |
| GPU targeting | Native NVIDIA/Orin runtime | Native `sm_87` proven with PTX JIT disabled | Requires pinned MAX 26.4 and JetPack `ptxas` compatibility path | TensorRT maturity |
| Integration | Already supplies the demo's detection contract | Backend seam exists but candidate is not qualified | Additional importer, compiler, weight-registry, decode, parity, and lifecycle code required | TensorRT |
| Portability | NVIDIA-specific | MAX has a stronger portability goal | Current artifact still needed target-specific compile and platform work | MAX in theory, not yet in practice |

The TensorRT and MAX resource measurements were not captured with a single
identical profiler protocol, so memory and GPU-utilization figures are useful
operational evidence rather than a laboratory-perfect comparison. The latency
gap is large enough that this limitation does not affect the decision.

## Accuracy findings

- The YOLO11n candidate retained about 92% of the same-data YOLO11s mAP50 and
  94% of its mAP50-95 while using 73% fewer parameters.
- Its fixed-threshold held-out F1 was 0.662 at confidence 0.4, but aggregate
  corpus quality did not survive the first important live pear scene.
- On that scene, MAX and local PyTorch agreed closely: both placed a good box
  over the pear and scored apple above pear. This clears the importer of the
  primary class-ordering failure and rejects the candidate training result for
  the stage demo.
- A same-frame production TensorRT comparison was interrupted, so the live
  snapshot is not a completed head-to-head accuracy benchmark. TensorRT's
  advantage here is the already observed end-to-end demo behavior, not a claim
  of completed frozen-frame parity.

## Time spent

The following totals are reconstructed from checked-in JSON results, per-run
training histories, and the full-model attempt log.

| Work category | Recorded time | Included evidence |
| --- | ---: | --- |
| Local training attempts | 1.56 h | Native skeleton versions, distillation/target experiments, YOLO11s, completed YOLO11n, and logged interrupted YOLO11n runs |
| Woof compile/runtime experiments | 2.83 h | Full-model attempts, YOLO11s 640/416, native skeleton benchmark, `sm_87` proof, and YOLO11n compile/restart |
| Image build and Wendy deployment | 0.70 h | Separately reported initial/diagnostic deployments and the latest candidate deployment |
| Categorized total | **5.09 h** | Sum of the recorded categories |

The most conservative subtotal, excluding image-build/deployment intervals
that may overlap some startup/readiness work, is **4.39 hours**. The MAX effort
spanned approximately **22 hours of calendar time**, from the first recorded
research artifact on 2026-08-06 at 12:58 PDT through this retrospective on
2026-08-07 at 10:47 PDT.

Neither number is total human effort. They omit implementation, importer and
model-design work, diagnostics, reading logs, device reboots, battery swaps,
charging, user observation, interrupted commands without complete records, and
idle/overnight time. Therefore the honest statement is:

> We have approximately five hours of recorded machine work and materially more
> engineering time invested across about a day of elapsed project time.

## What MAX did prove

1. MAX 26.4 can compile and execute native `sm_87` GPU code on Woof with PTX
   JIT disabled.
2. A deliberately small native detection skeleton can run at 640 px in 66.03 ms
   before complete decoding/NMS.
3. FP16 and model-size reductions substantially improved MAX: YOLO11n was about
   3.35 times faster than YOLO11s at 416 px.
4. The remaining gap is not one isolated bug. It combines equivalent-model
   graph/runtime cost, memory-heavy compilation, startup behavior, and the
   accuracy lost by shrinking or redesigning the model.

## Recommendation

Stop MAX work for the stage demo. Preserve the probes, compiler code, results,
and backend seam as research artifacts, but restore engineering focus to the
TensorRT path, approach distance, return-home accuracy, camera resilience, and
section-level demo tests.

Reopen MAX only when at least one of these changes is available:

- a newer MAX release with demonstrated Orin/YOLO performance,
- an official optimized detector architecture with trainable weights and a
  supported export path,
- a target device with substantially more memory, or
- a bounded research budget with a frozen same-frame TensorRT acceptance suite.

