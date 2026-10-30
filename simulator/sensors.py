"""Symulator czujników maszyny przemysłowej.

Każdy parametr to proces z powrotem do średniej (mean reversion) z szumem,
ograniczony do realistycznego zakresu. Temperatura i wibracje zależą od
prędkości obrotowej. Można wstrzyknąć awarię - odczyty z awarii są
oznaczone etykietą (`is_anomaly`, `fault`).
"""

import argparse
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

# Progi temperatury (°C)
TEMP_WARNING_THRESHOLD = 75.0
TEMP_CRITICAL_THRESHOLD = 85.0

# Wartości nominalne
NOMINAL_TEMPERATURE = 65.0
NOMINAL_VIBRATION = 2.5
NOMINAL_PRESSURE = 5.0
NOMINAL_SPEED = 1500.0

# Dozwolone zakresy (min, max)
TEMPERATURE_RANGE = (20.0, 120.0)
VIBRATION_RANGE = (0.0, 15.0)
PRESSURE_RANGE = (0.0, 10.0)
SPEED_RANGE = (0.0, 3000.0)

# Siła powrotu do średniej (0..1) na krok
REVERSION = 0.05

# Wpływ odchylenia prędkości od nominalnej na pozostałe parametry
TEMP_PER_RPM = 0.01  # °C na 1 rpm
VIB_PER_RPM = 0.0005  # jednostki na 1 rpm

FAULT_TYPES = ("bearing_wear", "overheating")


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    low, high = bounds
    return max(low, min(high, value))


@dataclass
class SensorReading:
    machine_id: str
    timestamp: datetime
    temperature: float
    vibration: float
    pressure: float
    rotational_speed: float
    is_anomaly: bool = False
    fault: Optional[str] = None


class IndustrialMachine:
    """Pojedyncza maszyna generująca odczyty co `step_seconds` sekund.

    Parametry:
        machine_id: identyfikator maszyny.
        seed: ziarno generatora losowego (powtarzalność wyników).
        step_seconds: odstęp czasu symulacji między odczytami.
        start_time: czas pierwszego odczytu (domyślnie: teraz, UTC).
    """

    def __init__(
        self,
        machine_id: str,
        seed: Optional[int] = None,
        step_seconds: float = 1.0,
        start_time: Optional[datetime] = None,
    ):
        self.machine_id = machine_id
        self._rng = random.Random(seed)
        self._step = timedelta(seconds=step_seconds)
        self._time = start_time or datetime.now(timezone.utc)

        self.temperature = NOMINAL_TEMPERATURE
        self.vibration = NOMINAL_VIBRATION
        self.pressure = NOMINAL_PRESSURE
        self.rotational_speed = NOMINAL_SPEED

        self.fault: Optional[str] = None
        self._fault_steps = 0

    def inject_fault(self, fault: str) -> None:
        """Rozpoczyna awarię. Jej skutki narastają z każdym krokiem."""
        if fault not in FAULT_TYPES:
            raise ValueError(f"Nieznana awaria: {fault!r}. Dostępne: {FAULT_TYPES}")
        self.fault = fault
        self._fault_steps = 0

    def clear_fault(self) -> None:
        self.fault = None
        self._fault_steps = 0

    def _revert(self, current: float, nominal: float, noise: float) -> float:
        """Jeden krok procesu z powrotem do średniej."""
        return (
            current
            + REVERSION * (nominal - current)
            + self._rng.uniform(-noise, noise)
        )

    def generate_reading(self) -> SensorReading:
        speed = self._revert(self.rotational_speed, NOMINAL_SPEED, 50.0)
        speed = _clamp(speed, SPEED_RANGE)
        speed_delta = speed - NOMINAL_SPEED

        # Temperatura i wibracje podążają za prędkością
        temp_target = NOMINAL_TEMPERATURE + TEMP_PER_RPM * speed_delta
        vib_target = NOMINAL_VIBRATION + VIB_PER_RPM * speed_delta

        temperature = self._revert(self.temperature, temp_target, 1.0)
        vibration = self._revert(self.vibration, vib_target, 0.15)
        pressure = self._revert(self.pressure, NOMINAL_PRESSURE, 0.1)

        if self.fault is not None:
            self._fault_steps += 1
            n = self._fault_steps
            if self.fault == "bearing_wear":
                vibration += 0.02 * n
                temperature += 0.01 * n
            elif self.fault == "overheating":
                temperature += 0.3 * n

        self.rotational_speed = speed
        self.temperature = _clamp(temperature, TEMPERATURE_RANGE)
        self.vibration = _clamp(vibration, VIBRATION_RANGE)
        self.pressure = _clamp(pressure, PRESSURE_RANGE)

        timestamp = self._time
        self._time += self._step

        return SensorReading(
            machine_id=self.machine_id,
            timestamp=timestamp,
            temperature=round(self.temperature, 2),
            vibration=round(self.vibration, 2),
            pressure=round(self.pressure, 2),
            rotational_speed=round(self.rotational_speed, 2),
            is_anomaly=self.fault is not None,
            fault=self.fault,
        )

    def get_temperature_status(self, temperature: Optional[float] = None) -> str:
        """Status dla podanej temperatury (domyślnie bieżącej)."""
        t = self.temperature if temperature is None else temperature
        if t < TEMP_WARNING_THRESHOLD:
            return "NORMAL"
        if t <= TEMP_CRITICAL_THRESHOLD:
            return "WARNING"
        return "CRITICAL"

    def is_overheating(self, temperature: Optional[float] = None) -> bool:
        return self.get_temperature_status(temperature) == "CRITICAL"


def simulate(
    n: int,
    machine_id: str = "MACHINE-001",
    seed: Optional[int] = None,
    step_seconds: float = 1.0,
    fault: Optional[str] = None,
    fault_start: int = 0,
    start_time: Optional[datetime] = None,
) -> pd.DataFrame:
    """Generuje `n` odczytów. Awaria `fault` startuje od kroku `fault_start`."""
    machine = IndustrialMachine(
        machine_id, seed=seed, step_seconds=step_seconds, start_time=start_time
    )
    readings = []
    for i in range(n):
        if fault is not None and i == fault_start:
            machine.inject_fault(fault)
        readings.append(machine.generate_reading())
    return pd.DataFrame([asdict(r) for r in readings])


def main() -> None:
    parser = argparse.ArgumentParser(description="Symulator czujników maszyny")
    parser.add_argument("-n", type=int, default=1000, help="liczba odczytów")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fault", choices=FAULT_TYPES, default=None)
    parser.add_argument("--fault-start", type=int, default=500)
    parser.add_argument("--output", default="sensor_data.csv")
    args = parser.parse_args()

    df = simulate(
        args.n, seed=args.seed, fault=args.fault, fault_start=args.fault_start
    )
    print("Generated readings:", len(df))
    print(df.head())
    df.to_csv(args.output, index=False)
    print(f"Data saved to {args.output}")


if __name__ == "__main__":
    main()
