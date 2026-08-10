from scripts.calibrate_forward_range import CalibrationSample, analyze


def test_stationary_calibration_finds_forward_channel_offset_noise_and_latency() -> None:
    samples = []
    for distance in (0.50, 0.30, 0.20, 0.15):
        for delta in (-0.004, -0.002, 0.0, 0.002, 0.004):
            samples.append(
                CalibrationSample(
                    known_clearance_m=distance,
                    ranges_m=(0.0, distance + 0.21 + delta, 0.8, 0.0),
                    pose_age_s=0.03,
                    pear_visible=True,
                    pear_center_error_ratio=0.02,
                )
            )

    result = analyze(samples)

    assert result["qualified"] is True
    assert result["forward_index"] == 1
    assert abs(result["sensor_to_front_envelope_m"] - 0.21) < 1e-6
    assert result["noise_m_p95"] <= 0.0041
    assert result["sensor_latency_s_p95"] == 0.03
    assert result["braking_qualification_required"] is True
