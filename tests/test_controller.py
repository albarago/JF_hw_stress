from jf_hw_stress.controller import Controller, Reading, Thresholds

T = Thresholds(fps_ratio=0.9, ramp_interval=10, fail_secs=5, warmup_secs=6, start_timeout=30)


def readings(*fps, status="running", age=60.0):
    return [Reading(i + 1, status, f, age) for i, f in enumerate(fps)]


def step(controller, clock, seconds, fps_values):
    decision = None
    for _ in range(int(seconds * 2)):
        clock.advance(0.5)
        decision = controller.tick(readings(*fps_values))
        if decision.action != "wait":
            return decision
    return decision


def test_warmup_then_stable_then_add(clock):
    c = Controller(24.0, T, clock)
    assert c.threshold == 21.6
    assert step(c, clock, 5.5, [30]).action == "wait" and c.phase == "warmup"
    step(c, clock, 1, [30])
    assert c.phase == "stable"
    assert step(c, clock, 11, [30]).action == "add"
    assert c.max_stable == 1 and c.phase == "warmup"


def test_failure_needs_the_full_fail_window(clock):
    c = Controller(24.0, T, clock)
    step(c, clock, 7, [30, 30])
    assert step(c, clock, 4, [30, 20]).action == "wait"        # 4 s below: not yet
    assert step(c, clock, 3, [30, 30]).action == "wait"        # recovered: the clock resets
    decision = step(c, clock, 6, [30, 20])
    assert decision.action == "fail" and decision.stream_id == 2 and "20.0 fps" in decision.reason


def test_a_dip_restarts_the_stability_interval(clock):
    c = Controller(24.0, T, clock)
    step(c, clock, 7, [30])
    step(c, clock, 8, [30])
    step(c, clock, 2, [21])                                     # below, but not long enough to fail
    assert step(c, clock, 8, [30]).action == "wait"            # interval restarted at the dip
    assert step(c, clock, 3, [30]).action == "add"


def test_a_crashed_stream_is_a_failure_once(clock):
    c = Controller(24.0, T, clock)
    rs = readings(30) + [Reading(2, "error", None, 3.0, "Invalid data")]
    d = c.tick(rs)
    assert d.action == "fail" and "stream 2" in d.reason and "Invalid data" in d.reason
    c.rearm()
    assert c.tick(rs).action == "wait"


def test_a_stream_that_never_produces_frames_fails(clock):
    c = Controller(24.0, T, clock)
    assert c.tick([Reading(1, "starting", None, 10.0)]).action == "wait"
    assert c.tick([Reading(1, "starting", None, 31.0)]).action == "fail"


def test_ignored_streams_do_not_count(clock):
    c = Controller(24.0, T, clock)
    c.ignore(2)
    assert c.tick(readings(30) + [Reading(2, "error", None, 5.0)]).action == "wait"
