from types import SimpleNamespace

from media.model_router import FruitModelRouter


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.value


class FakeBoxes:
    def __init__(self, confidences, boxes, class_ids=None):
        self.conf = FakeTensor(confidences)
        self.xyxy = [FakeTensor(box) for box in boxes]
        self.cls = FakeTensor(class_ids or [45] * len(confidences))

    def __len__(self):
        return len(self.conf.value)


class FakeModel:
    def __init__(self, predictions):
        self.predictions = list(predictions)
        self.calls = []

    def predict(self, **options):
        self.calls.append(options)
        prediction = self.predictions.pop(0)
        confidences, boxes, *class_ids = prediction
        return [
            SimpleNamespace(
                boxes=FakeBoxes(
                    confidences,
                    boxes,
                    class_ids[0] if class_ids else None,
                )
            )
        ]


class ColorCrop:
    def __init__(self, pixels) -> None:
        self.pixels = pixels

    def reshape(self, *_shape):
        return self

    def tolist(self):
        return self.pixels


class ColorSource:
    shape = (720, 1280, 3)

    def __init__(self, pixels) -> None:
        self.pixels = pixels

    def __getitem__(self, _key):
        return ColorCrop(self.pixels)


def test_raw_bowl_and_bounded_color_derive_a_mango_motion_candidate() -> None:
    general = FakeModel([])
    mango = FakeModel([([0.128], [[737, 496, 775, 515]])])
    router = FruitModelRouter(
        general_model=general,
        general_class_ids={"apple": 1, "banana": 2, "pear": 3},
        mango_model=mango,
        mango_class_ids={"sports ball": 32, "bowl": 45},
    )

    result = router.predict(
        source=ColorSource([[172, 218, 246]] * 100),
        target_fruit="mango",
        device=0,
    )

    assert result.candidate is not None
    assert result.candidate.confidence == 0.128
    assert result.candidate.bbox_xyxy == (737, 496, 775, 515)
    assert result.route == {
        "mode": "mango_derived",
        "triggered": True,
        "confirmed": True,
        "raw_label": "bowl",
        "raw_confidence": 0.128,
        "raw_bbox_xyxy": [737, 496, 775, 515],
        "derived_identity": "mango",
        "derived_confidence": 1.0,
    }
    assert mango.calls[0]["classes"] == [32, 45]
    assert mango.calls[0]["conf"] == 0.01
    assert general.calls == []


def test_mango_route_accepts_orange_sports_ball_but_rejects_green_one() -> None:
    general = FakeModel([])
    mango = FakeModel(
        [
            ([0.15], [[737, 496, 775, 515]], [32]),
            ([0.15], [[737, 496, 775, 515]], [32]),
        ]
    )
    router = FruitModelRouter(
        general_model=general,
        general_class_ids={"apple": 1, "banana": 2, "pear": 3},
        mango_model=mango,
        mango_class_ids={"sports ball": 32, "bowl": 45},
    )

    orange_mango = router.predict(
        source=ColorSource([[172, 218, 246]] * 100),
        target_fruit="mango",
        device=0,
    )
    green_pear = router.predict(
        source=ColorSource([[0, 180, 50]] * 100),
        target_fruit="mango",
        device=0,
    )

    assert orange_mango.candidate is not None
    assert orange_mango.route["raw_label"] == "sports ball"
    assert orange_mango.route["derived_identity"] == "mango"
    assert green_pear.candidate is None
    assert green_pear.route["raw_label"] == "sports ball"
    assert green_pear.route["derived_identity"] == "unknown"
    assert all(call["classes"] == [32, 45] for call in mango.calls)


def test_mango_raw_and_color_threshold_boundaries_are_fail_closed() -> None:
    general = FakeModel([])
    mango = FakeModel(
        [
            ([0.08], [[737, 496, 775, 515]]),
            ([0.081], [[737, 496, 775, 515]]),
        ]
    )
    router = FruitModelRouter(
        general_model=general,
        general_class_ids={"apple": 1, "banana": 2, "pear": 3},
        mango_model=mango,
        mango_class_ids={"sports ball": 32, "bowl": 45},
        mango_minimum_raw_confidence=0.08,
        mango_minimum_color_confidence=0.80,
    )

    exact_floor = router.predict(
        source=ColorSource([[172, 218, 246]] * 100),
        target_fruit="mango",
        device=0,
    )
    above_floor = router.predict(
        source=ColorSource([[172, 218, 246]] * 100),
        target_fruit="mango",
        device=0,
    )

    assert exact_floor.candidate is None
    assert exact_floor.route["raw_confidence"] == 0.08
    assert above_floor.candidate is not None
    assert above_floor.route["raw_confidence"] == 0.081
    assert above_floor.route["derived_confidence"] >= 0.80


def test_general_banana_proposal_is_confirmed_by_resident_specialist() -> None:
    general = FakeModel([([0.41], [[100, 200, 180, 300]])])
    specialist = FakeModel([([0.82], [[104, 204, 184, 304]])])
    router = FruitModelRouter(
        general_model=general,
        general_class_ids={"apple": 1, "banana": 2, "pear": 3},
        banana_specialist_model=specialist,
        banana_specialist_class_id=0,
        banana_minimum_confidence=0.55,
        banana_minimum_agreement_iou=0.10,
    )

    result = router.predict(source=object(), target_fruit="banana", device=0)

    assert result.candidate is not None
    assert result.candidate.confidence == 0.82
    assert result.candidate.bbox_xyxy == (104, 204, 184, 304)
    assert result.inference_passes == 2
    assert result.route == {
        "mode": "banana_specialist",
        "triggered": True,
        "confirmed": True,
        "general_confidence": 0.41,
        "specialist_confidence": 0.82,
        "agreement_iou": result.route["agreement_iou"],
    }
    assert result.route["agreement_iou"] > 0.80
    assert general.calls[0]["classes"] == [2]
    assert specialist.calls[0]["classes"] == [0]


def test_apple_and_pear_never_spend_specialist_inference() -> None:
    general = FakeModel(
        [
            ([0.81], [[10, 20, 110, 220]]),
            ([0.78], [[30, 40, 130, 240]]),
        ]
    )
    specialist = FakeModel([])
    router = FruitModelRouter(
        general_model=general,
        general_class_ids={"apple": 1, "banana": 2, "pear": 3},
        banana_specialist_model=specialist,
        banana_specialist_class_id=0,
    )

    apple = router.predict(source=object(), target_fruit="apple", device=0)
    pear = router.predict(source=object(), target_fruit="pear", device=0)

    assert apple.candidate is not None and apple.candidate.confidence == 0.81
    assert pear.candidate is not None and pear.candidate.confidence == 0.78
    assert apple.inference_passes == pear.inference_passes == 1
    assert apple.route["mode"] == pear.route["mode"] == "general"
    assert specialist.calls == []


def test_specialist_rejection_returns_no_banana_detection() -> None:
    general = FakeModel([([0.48], [[100, 200, 180, 300]])])
    specialist = FakeModel([([0.91], [[800, 100, 900, 200]])])
    router = FruitModelRouter(
        general_model=general,
        general_class_ids={"apple": 1, "banana": 2, "pear": 3},
        banana_specialist_model=specialist,
        banana_specialist_class_id=0,
        banana_minimum_confidence=0.55,
        banana_minimum_agreement_iou=0.10,
    )

    result = router.predict(source=object(), target_fruit="banana", device=0)

    assert result.candidate is None
    assert result.inference_passes == 2
    assert result.route["triggered"] is True
    assert result.route["confirmed"] is False
    assert result.route["specialist_confidence"] == 0.91
    assert result.route["agreement_iou"] == 0.0


def test_missing_general_banana_proposal_does_not_run_specialist() -> None:
    general = FakeModel([([], [])])
    specialist = FakeModel([])
    router = FruitModelRouter(
        general_model=general,
        general_class_ids={"apple": 1, "banana": 2, "pear": 3},
        banana_specialist_model=specialist,
        banana_specialist_class_id=0,
    )

    result = router.predict(source=object(), target_fruit="banana", device=0)

    assert result.candidate is None
    assert result.inference_passes == 1
    assert result.route == {
        "mode": "banana_specialist",
        "triggered": False,
        "confirmed": False,
        "general_confidence": None,
        "specialist_confidence": None,
        "agreement_iou": None,
    }
    assert specialist.calls == []
