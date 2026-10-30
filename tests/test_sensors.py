from datetime import datetime, timedelta, timezone

import pytest

from simulator.sensors import (
    PRESSURE_RANGE,
    TEMPERATURE_RANGE,
    VIBRATION_RANGE,
    IndustrialMachine,
    simulate,
)

START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def make_machine(seed=42, **kwargs):
    return IndustrialMachine("M-1", seed=seed, start_time=START, **kwargs)


@pytest.mark.parametrize(
    "temperature, expected",
    [
        (74.9, "NORMAL"),
        (75.0, "WARNING"),
        (85.0, "WARNING"),
        (85.1, "CRITICAL"),
    ],
)
def test_temperature_status_thresholds(temperature, expected):
    assert make_machine().get_temperature_status(temperature) == expected


def test_is_overheating_matches_status():
    m = make_machine()
    assert not m.is_overheating(85.0)
    assert m.is_overheating(85.1)


def test_status_defaults_to_current_temperature():
    m = make_machine()
    m.temperature = 90.0
    assert m.get_temperature_status() == "CRITICAL"


def test_same_seed_gives_same_readings():
    a = [make_machine(seed=1).generate_reading() for _ in range(1)]
    m1, m2 = make_machine(seed=1), make_machine(seed=1)
    r1 = [m1.generate_reading() for _ in range(50)]
    r2 = [m2.generate_reading() for _ in range(50)]
    assert r1 == r2
    assert a[0] == r1[0]


def test_different_seeds_differ():
    r1 = make_machine(seed=1).generate_reading()
    r2 = make_machine(seed=2).generate_reading()
    assert r1 != r2


def test_timestamps_advance_by_step():
    m = make_machine(step_seconds=5)
    r = [m.generate_reading() for _ in range(3)]
    assert r[0].timestamp == START
    assert r[1].timestamp - r[0].timestamp == timedelta(seconds=5)
    assert r[2].timestamp - r[1].timestamp == timedelta(seconds=5)


def test_values_stay_within_ranges_in_long_run():
    m = make_machine(seed=7)
    for _ in range(20_000):
        r = m.generate_reading()
        assert TEMPERATURE_RANGE[0] <= r.temperature <= TEMPERATURE_RANGE[1]
        assert VIBRATION_RANGE[0] <= r.vibration <= VIBRATION_RANGE[1]
        assert PRESSURE_RANGE[0] <= r.pressure <= PRESSURE_RANGE[1]


def test_healthy_machine_stays_near_nominal():
    m = make_machine(seed=3)
    temps = [m.generate_reading().temperature for _ in range(5000)]
    assert max(temps) < 75.0
    assert min(temps) > 55.0


def test_healthy_readings_not_labeled_anomaly():
    r = make_machine().generate_reading()
    assert r.is_anomaly is False
    assert r.fault is None


def test_unknown_fault_raises():
    with pytest.raises(ValueError):
        make_machine().inject_fault("nope")


def test_overheating_fault_raises_temperature_and_labels():
    m = make_machine(seed=5)
    m.inject_fault("overheating")
    readings = [m.generate_reading() for _ in range(100)]
    assert all(r.is_anomaly and r.fault == "overheating" for r in readings)
    assert m.is_overheating(readings[-1].temperature)


def test_bearing_wear_raises_vibration():
    m = make_machine(seed=5)
    healthy = [m.generate_reading().vibration for _ in range(200)]
    m.inject_fault("bearing_wear")
    worn = [m.generate_reading().vibration for _ in range(200)]
    assert sum(worn[-20:]) / 20 > sum(healthy[-20:]) / 20 + 1.0


def test_clear_fault_stops_labeling():
    m = make_machine()
    m.inject_fault("overheating")
    m.generate_reading()
    m.clear_fault()
    assert m.generate_reading().is_anomaly is False


def test_simulate_dataframe_shape_and_labels():
    df = simulate(100, seed=1, fault="bearing_wear", fault_start=60, start_time=START)
    assert len(df) == 100
    assert list(df.columns) == [
        "machine_id",
        "timestamp",
        "temperature",
        "vibration",
        "pressure",
        "rotational_speed",
        "is_anomaly",
        "fault",
    ]
    assert not df["is_anomaly"].iloc[:60].any()
    assert df["is_anomaly"].iloc[60:].all()

