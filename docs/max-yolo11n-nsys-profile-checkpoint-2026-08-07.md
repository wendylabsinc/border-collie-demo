# MAX YOLO11n Nsight profiling checkpoint

Date: 2026-08-07
Status: completed; profiling app stopped safely

## Objective

Capture exactly one warm YOLO11n 416 px MAX inference with Nsight Systems and
detailed MAX markers. Five warmups must occur outside the capture range. The
single capture must include input upload, MAX execution, output download, CUDA
APIs, NVTX ranges, OS runtime activity, and CUDA memory usage.

## Result

The bounded Nsight Systems capture completed on Woof after five warmups. It
captured exactly one 416 px FP16 MAX YOLO11n inference. The measured inference
was 854.13 ms, consistent with the prior 841.45 ms median.

The bottleneck is unambiguous: MAX selected naive GPU convolution kernels for
nearly the entire graph.

| Category | GPU time | Share of GPU work |
| --- | ---: | ---: |
| Naive NHWC FP16 convolutions | 846.512 ms | 99.424% |
| Tensor Core / CUTLASS kernels | 1.173 ms | 0.138% |
| Matrix multiplication / GEMV | 0.434 ms | 0.051% |
| Layout transposes | 0.353 ms | 0.042% |
| Softmax | 0.253 ms | 0.030% |
| Host/device copies | 0.134 ms | 0.016% |
| All GPU operations | 851.419 ms | 100% |

The GPU trace spans 853.111 ms and contains 851.419 ms of active GPU work:
99.802% occupancy across the captured interval. Total idle gaps are only
1.691 ms, and the largest gap is 1.118 ms. The slow result therefore is not
caused by CPU launch gaps or starvation.

The CUDA API report attributes 842.336 ms to `cuMemcpyDtoHAsync_v2`. This does
not mean the device-to-host copy itself is slow: the GPU copy event is only
0.007 ms. The API call is where the CPU waits for the preceding convolution
work to finish before the output can be returned.

The two largest individual naive convolution kernels take 133.993 ms and
97.636 ms. The twelve largest naive convolution operations account for 84.48%
of all captured GPU time.

Nsight did not find NVTX range records even though the session requested MAX
`gpu_profiling("detailed")`. The CUDA kernel, API, memory, and timing evidence
is complete enough to identify the bottleneck, but per-MAX-operation NVTX
labels are not available in this capture.

## Conclusion

The tested MAX graph is slow because its convolution lowering chooses
`conv2d_gpu_naive_nhwc_rscf_float16_float16_float16` kernels instead of
optimized Tensor Core/CUTLASS convolution kernels. Reworking DFL decoding,
removing layout copies, reducing transfer overhead, or closing launch gaps
cannot materially fix an 854 ms inference: together those categories are well
under 1% of the captured GPU work.

The next useful experiment must change convolution lowering or graph/operator
construction and prove that the naive-convolution share falls. Until that is
demonstrated, TensorRT remains the production fruit detector.

## Safe stopped state

- `border-collie-max-yolo11n-candidate` is stopped.
- `woof-thermal-monitor` is the only relevant running app.
- No motion-capable code was included or started.
- The profiling attempt completed one warm inference, then the candidate app
  was stopped.
- The cached MEF and weights remain in
  `/var/lib/wendy/volumes/border-collie-max-yolo11n-artifacts/` on Woof.
- Nsight Systems 2024.5.4 was checksum-verified and extracted without package
  installation under `/var/tmp/woof-nsys-2024.5.4/` on Woof.

## Saved implementation

- `lab/max-yolo11s-candidate/profile_once.py` loads only the cached MEF and
  weights, performs five completed warmups, brackets one inference with
  `cudaProfilerStart()` / `cudaProfilerStop()`, and writes a JSON result.
- It calls `InferenceSession.gpu_profiling("detailed")` before model load.
- `lab/max-yolo11s-candidate/profile_once.sh` contains the bounded Nsight
  command and a unique session name.
- The profiling Dockerfile includes Nsight Systems 2024.5.4. The current local
  `CMD` is deliberately `sleep 600` so the next attempt can launch profiling
  from Woof's host into the already-isolated container.
- The user explicitly removed the battery-start requirement. Thermal limits,
  memory isolation, motion isolation, and no-restart behavior remain.

## Profiling mechanics learned

1. `MODULAR_ENABLE_PROFILING=detailed` applied process-wide breaks MAX 26.4
   during `InferenceSession` construction with:

   ```text
   TypeError: InferenceSession.__new__(InferenceSession) is not safe
   ```

   The API-based `session.gpu_profiling("detailed")` path avoids that import-time
   constructor instrumentation while preserving detailed markers.

2. Running the embedded `nsys` target CLI inside the Wendy container repeatedly
   reserves its session name and then reports that same session as already in
   use. There were no matching orphaned `nsys` processes or lock files on the
   host after the container exited.

3. Host-side `nsenter` failed on Woof when reassociating the target mount
   namespace. After the reboot cleared stale Nsight sessions, running Nsight
   inside the dormant container succeeded.

4. Setting a zero battery floor now explicitly permits unavailable post-reboot
   battery telemetry. The independent thermal preflight and postflight remain
   enabled.

## Completed sequence

1. Redeployed the sleeper image with restart disabled and chunking off.
2. Started container-local `nsys profile` with:
   - `--trace=cuda,osrt,nvtx`
   - `--cuda-memory-usage=true`
   - `--capture-range=cudaProfilerApi`
   - `--capture-range-end=stop`
   - a new unique `--session-new` value
3. Confirmed the JSON says `warmups_outside_capture: 5` and
   `profiled_inferences: 1`.
4. Ran `nsys stats` against the `.nsys-rep` and classified time into
   convolution kernels, DFL/softmax, layout copies, launch gaps, and transfers.
5. Stopped the sleeper container and retained the thermal monitor.

The persistent device evidence is stored under
`/var/lib/wendy/volumes/border-collie-max-yolo11n-artifacts/`:

- `max-yolo11n-one-warm.nsys-rep` (127 KiB)
- `max-yolo11n-one-warm.sqlite`
- `max-yolo11n-one-warm-profile.json`
- CSV summaries for CUDA GPU, memory, API, and trace events

## Deployment measurements

- First profiling-image deployment: 955.839 s total.
- DLO phases: build 442.742 s, export 197.483 s, transfer 266.904 s, unpack
  45.834 s, replacement 1.719 s.
- The image was 2.53 GB because Nsight plus its dependencies are instrumentation
  only. It must not replace the production image.
- Subsequent cached instrumentation redeploys took approximately 46-63 s,
  dominated by local image export.
- The successful post-reboot instrumentation deployment took 45.736 s. DLO
  attributed 0.550 s to build, 42.001 s to export, 1.123 s to transfer,
  0.187 s to unpack, and 0.585 s to replacement.

## Remaining administrative work

- Add the sanitized aggregate profiling-image deployment result to the
  `docker-layer-optimizer` repository after profiling resumes. This was deferred
  to honor the immediate stop request.
