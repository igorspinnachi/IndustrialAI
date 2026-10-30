"""Detekcja anomalii na danych z symulatora.

Detektor hybrydowy, uczony wyłącznie na zdrowych odczytach (novelty detection):
- Isolation Forest wychwytuje nietypowe kombinacje kanałów,
- reguła z-score (max |z| po cechach) wychwytuje duże odchylenie pojedynczego
  kanału, które Isolation Forest potrafi przeoczyć (opiera się na rangach
  wartości w zakresie treningowym, a nie na ich wielkości).

Etykiety `is_anomaly` służą tylko do oceny jakości.
"""

import argparse
from typing import Optional

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from simulator.sensors import FAULT_TYPES, simulate

SENSOR_COLUMNS = ["temperature", "vibration", "pressure", "rotational_speed"]
ROLLING_COLUMNS = ["temperature", "vibration"]
ROLLING_WINDOW = 10


def build_features(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Surowe odczyty + średnie kroczące (łapią powolny dryf przed awarią).

    Średnie liczone są osobno dla każdej maszyny. Początek szeregu
    wypełniany jest dostępnymi dotąd danymi (min_periods=1), bez zaglądania
    w przyszłość.
    """
    features = df[SENSOR_COLUMNS].copy()
    grouped = df.groupby("machine_id")
    for col in ROLLING_COLUMNS:
        features[f"{col}_roll_mean"] = grouped[col].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
    return features


class AnomalyDetector:
    def __init__(
        self,
        contamination: float = 0.01,
        n_estimators: int = 200,
        seed: Optional[int] = 0,
        z_quantile: float = 0.99,
    ):
        self.z_quantile = z_quantile
        self.model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "forest",
                    IsolationForest(
                        n_estimators=n_estimators,
                        contamination=contamination,
                        random_state=seed,
                    ),
                ),
            ]
        )
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "AnomalyDetector":
        """Uczy model na podanych odczytach (powinny być zdrowe)."""
        features = build_features(df)
        self.model.fit(features)
        self._mean = features.mean()
        self._std = features.std().replace(0, 1.0)
        self._z_threshold = float(
            self._max_abs_z(features).quantile(self.z_quantile)
        )
        self._fitted = True
        return self

    def _max_abs_z(self, features: pd.DataFrame) -> pd.Series:
        return ((features - self._mean) / self._std).abs().max(axis=1)

    def score(self, df: pd.DataFrame) -> pd.Series:
        """Wynik anomalii z Isolation Forest (wyższy = bardziej podejrzany)."""
        self._check_fitted()
        raw = self.model.decision_function(build_features(df))
        return pd.Series(-raw, index=df.index, name="anomaly_score")

    def z_score(self, df: pd.DataFrame) -> pd.Series:
        """Największe odchylenie (w odchyleniach std.) wśród cech odczytu."""
        self._check_fitted()
        z = self._max_abs_z(build_features(df))
        return z.rename("max_abs_z")

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """True dla odczytów uznanych za anomalie (Isolation Forest LUB z-score)."""
        self._check_fitted()
        features = build_features(df)
        by_forest = self.model.predict(features) == -1
        by_z = (self._max_abs_z(features) > self._z_threshold).to_numpy()
        return pd.Series(by_forest | by_z, index=df.index, name="predicted_anomaly")

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Detektor nie jest wytrenowany - wywołaj fit().")


def evaluate(y_true: pd.Series, y_pred: pd.Series) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_alarm_rate": float(y_pred[~y_true.astype(bool)].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Trening i ocena detektora anomalii")
    parser.add_argument("--fault", choices=FAULT_TYPES, default="bearing_wear")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--fault-start", type=int, default=1000)
    args = parser.parse_args()

    train = simulate(args.train_size, seed=args.seed)
    test = simulate(
        args.test_size,
        seed=args.seed + 1,
        fault=args.fault,
        fault_start=args.fault_start,
    )

    detector = AnomalyDetector(seed=args.seed).fit(train)
    metrics = evaluate(test["is_anomaly"], detector.predict(test))

    print(f"Awaria: {args.fault} (od kroku {args.fault_start} z {args.test_size})")
    for name, value in metrics.items():
        print(f"  {name}: {value:.3f}")


if __name__ == "__main__":
    main()

