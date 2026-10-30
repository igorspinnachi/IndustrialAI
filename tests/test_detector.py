import pytest

from detector.anomaly import AnomalyDetector, build_features, evaluate
from simulator.sensors import simulate


@pytest.fixture(scope="module")
def healthy():
    return simulate(5000, seed=0)


@pytest.fixture(scope="module")
def detector(healthy):
    return AnomalyDetector(seed=0).fit(healthy)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        AnomalyDetector().predict(simulate(10, seed=0))


def test_build_features_adds_rolling_columns():
    f = build_features(simulate(50, seed=0))
    assert "temperature_roll_mean" in f.columns
    assert "vibration_roll_mean" in f.columns
    assert len(f) == 50


def test_rolling_features_do_not_look_ahead():
    df = simulate(50, seed=0)
    full = build_features(df)
    head = build_features(df.iloc[:20])
    assert full.iloc[:20].equals(head)


def test_low_false_alarm_rate_on_healthy_data(detector):
    test = simulate(3000, seed=1)
    assert detector.predict(test).mean() < 0.05


@pytest.mark.parametrize("fault", ["bearing_wear", "overheating"])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_detects_faults(detector, fault, seed):
    test = simulate(2000, seed=seed, fault=fault, fault_start=1000)
    metrics = evaluate(test["is_anomaly"], detector.predict(test))
    assert metrics["recall"] > 0.9
    assert metrics["precision"] > 0.9


def test_single_channel_spike_is_caught(detector):
    """Regresja: sam Isolation Forest przeoczał duże odchylenie jednego kanału."""
    test = simulate(200, seed=5)
    test.loc[100:, "temperature"] = 120.0
    assert detector.predict(test).iloc[100:].mean() > 0.9


def test_scores_higher_for_faulty_data(detector):
    test = simulate(2000, seed=4, fault="bearing_wear", fault_start=1000)
    score = detector.score(test)
    assert score.iloc[1000:].mean() > score.iloc[:1000].mean()
    z = detector.z_score(test)
    assert z.iloc[1000:].mean() > z.iloc[:1000].mean()

