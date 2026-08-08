from scripts.stage_scorecard import score_session


def make_run(
    number,
    fruit="pear",
    outcome="COMPLETED",
    home=0.05,
    min_conf=0.8,
    gpu_peak=None,
    dongle_visible=None,
    errors=0,
    p95=40.0,
    acquire_s=None,
):
    run = {
        "number": number,
        "target_fruit": fruit,
        "outcome": outcome,
        "home_distance_m": home,
        "stage_telemetry": {
            "approach_fruit": {
                "target_confidence": {"min": min_conf, "mean": min_conf, "max": min_conf, "samples": 4},
                "temps_c": {"gpu": {"min": 40.0, "mean": 45.0, "max": gpu_peak, "samples": 4}}
                if gpu_peak is not None
                else None,
            }
        },
        "network": {"error_count": errors, "latency_p95_ms": p95},
        "stage_durations": {"turn_to_fruit": acquire_s} if acquire_s else {},
    }
    if dongle_visible is not None:
        run["preflight_probes"] = {"dongle": {"visible": dongle_visible}}
    return run


def test_all_criteria_pass_on_clean_session():
    session = {
        "target_runs": 4,
        "runs": [
            make_run(1, "pear", dongle_visible=True, gpu_peak=50.0),
            make_run(2, "pear", dongle_visible=True, gpu_peak=52.0),
            make_run(3, "apple", min_conf=0.75, dongle_visible=True, gpu_peak=53.0),
            make_run(4, "apple", min_conf=0.75, dongle_visible=True, gpu_peak=54.0),
        ],
    }
    card = score_session(session)
    assert card["recorded_only"] is True
    criteria = card["criteria"]
    assert criteria["completion"]["passed"] is True
    assert criteria["home_gate"]["passed"] is True
    assert criteria["fruit_coverage"]["passed"] is True
    assert criteria["approach_confidence_floor"]["passed"] is True
    assert criteria["network_stability"]["passed"] is True
    assert criteria["thermal_trend"]["passed"] is True  # +4C < 10C limit
    assert criteria["dongle_visible"]["passed"] is True


def test_failures_are_recorded_not_raised():
    session = {
        "target_runs": 3,
        "runs": [
            make_run(1, "pear", outcome="FAILED", home=0.3, min_conf=0.4, errors=2),
            make_run(2, "banana", min_conf=0.5, gpu_peak=50.0),
            make_run(3, "banana", min_conf=0.6, gpu_peak=65.0),
        ],
    }
    criteria = score_session(session)["criteria"]
    assert criteria["completion"]["passed"] is False
    assert criteria["home_gate"]["passed"] is False
    assert criteria["home_gate"]["max_m"] == 0.3
    assert criteria["fruit_coverage"]["passed"] is False  # pear 0, banana <2? banana has 2 completed
    assert criteria["approach_confidence_floor"]["passed"] is False  # pear 0.4 < 0.65
    assert criteria["network_stability"]["passed"] is False  # cutouts
    assert criteria["thermal_trend"]["passed"] is False  # +15C rise
    # banana floor is the specialist's 0.55, not the app-side 0.20
    banana_entries = [
        e for e in criteria["approach_confidence_floor"]["per_run"] if e["fruit"] == "banana"
    ]
    assert banana_entries[0]["floor"] == 0.55
    assert banana_entries[0]["held"] is False  # 0.5 < 0.55
    assert banana_entries[1]["held"] is True  # 0.6 >= 0.55


def test_unmeasured_criteria_are_null_not_failed():
    session = {"target_runs": 2, "runs": [make_run(1), make_run(2)]}
    criteria = score_session(session)["criteria"]
    assert criteria["thermal_trend"]["passed"] is None
    assert criteria["dongle_visible"]["passed"] is None


def test_empty_session_scores_null():
    criteria = score_session({"target_runs": 10, "runs": []})["criteria"]
    assert all(c["passed"] is None for c in criteria.values())


def test_observations_capture_lighting_signals():
    session = {
        "target_runs": 2,
        "runs": [
            dict(make_run(1, "pear", acquire_s=4.2), lighting_frame={"path": "f/run-01.jpg"}),
            make_run(2, "apple", min_conf=0.72, acquire_s=9.8),
        ],
    }
    obs = score_session(session)["observations"]
    assert obs["approach_min_confidence_by_fruit"]["pear"]["runs"] == 1
    assert obs["acquisition_seconds_by_fruit"] == {"pear": [4.2], "apple": [9.8]}
    assert obs["lighting_frames"] == ["f/run-01.jpg"]
