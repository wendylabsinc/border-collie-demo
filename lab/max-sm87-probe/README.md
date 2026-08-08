# MAX native `sm_87` probe

This isolated probe compiles a deterministic vector-add MAX graph twice: once
for `sm_87` and once for the negative-control target `sm_80`. It then:

1. inspects both exported MEFs with `cuobjdump --list-elf`;
2. runs the `sm_87` artifact on `Accelerator(0)` with PTX JIT disabled;
3. requires the exact vector result;
4. requires the native `sm_80` CUBIN and its MAX artifact to be rejected under
   the same JIT-disabled environment; records whether the driver reports
   `CUDA_ERROR_NO_BINARY_FOR_GPU` or Jetson's observed
   `CUDA_ERROR_INVALID_SOURCE`; and
5. records MAX, Mojo, PTXAS, cuobjdump, SHA-256, and thermal evidence.

The error-name distinction is recorded instead of hidden. NVIDIA's CUDA 12.6
guide says minor-version CUBIN compatibility is not supported on Tegra, and
Woof's Tegra driver reports error 300 for the incompatible native `sm_80`
image. The architecture gate itself remains exact: the positive artifact must
contain `sm_87`, load natively, and execute with PTX JIT disabled.

It does not load the fruit model and has no camera, DDS, audio, or motion
authority.

The Wendy manifest caps the container at one CPU core. The evidence records
total proof wall time, accumulated CPU time, average cores used, the effective
cgroup quota, per-target compile wall/CPU time, and compiler peak RSS. This
makes the probe a conservative calibration point without allowing MAX to
consume all four of Woof's CPU cores.

## One-core reference measurement

On Woof, the complete two-target proof took 144.064 seconds and averaged 0.987
CPU cores. The `sm_87` compile took 34.917 seconds and peaked at 914,067,456
bytes RSS. The same compile took 25.811 seconds without the cap, so the observed
one-core slowdown was 1.353x. Jetson temperature increased by 0.906 C.

This does not scale linearly to the fruit graph. The previous full-graph attempt
was incomplete after 1,623 seconds and had already reached 13.96 GB RSS. Applying
only the measured CPU slowdown gives a lower bound of 2,196 seconds (36.6
minutes), not an expected completion time. A guarded full attempt should reserve
60-90 minutes, but compiler memory must be reduced or separately capped first;
the CPU quota alone does not prevent memory exhaustion.

Deploy from this directory with:

```bash
dlo deploy --root . --dockerfile Dockerfile --target woof -- \
  wendy --device woof.local run --detach --yes
```

Read the proof at `http://woof.local:8124/status` and stop the app after the
artifacts have been collected.
