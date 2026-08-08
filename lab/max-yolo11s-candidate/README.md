# MAX YOLO11n whole-fruit candidate

This is an isolated, perception-only deployment gate for the `MAX-NATIVE-007`
pear/apple/banana model. It compiles the fixed 640 px YOLO11n detection graph
with MAX 26.4 for Woof's GPU, runs synthetic inference benchmarks, and exposes
status on port 8126. The candidate has 2.58 million parameters and passed local
accuracy, ONNX parity, and importer-operator gates. This app imports no robot
SDK and cannot issue motion commands.

After the separate read-only camera source probe has captured a frame, call
`POST /infer-snapshot?confidence=0.25` to run that JPEG through the loaded MAX
model. The response contains class, confidence, box, and latency only.

The model is experimental. Its raw-camera corpus result is 93.75% class recall,
88.2% IoU-0.5 box recall, zero negative false positives, and 7/7 Go2 stage
validation passes. PyTorch and ONNX raw outputs also pass parity. Autonomous
motion remains disabled until MAX output parity, a newly captured independent
Go2 scene, and live camera behavior are validated on Woof.
