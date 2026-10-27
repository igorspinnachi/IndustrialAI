
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import random
import pandas as pd


@dataclass
class SensorReading:
    machine_id: str
    timestamp: datetime
    temperature: float
    vibration: float
    pressure: float
    rotational_speed: float


class IndustrialMachine:
    def __init__(self, machine_id: str):
        self.machine_id = machine_id
        self.temperature = 65.0
        self.vibration = 2.5
        self.pressure = 5.0
        self.rotational_speed = 1500.0

    def generate_reading(self) -> SensorReading:
        self.temperature += random.uniform(-1.5, 1.5)
        self.vibration += random.uniform(-0.2, 0.2)
        self.pressure += random.uniform(-0.1, 0.1)
        self.rotational_speed += random.uniform(-50, 50)

        return SensorReading(
            machine_id=self.machine_id,
            timestamp=datetime.now(timezone.utc),
            temperature=round(self.temperature, 2),
            vibration=round(self.vibration, 2),
            pressure=round(self.pressure, 2),
            rotational_speed=round(self.rotational_speed, 2),
        )

    def is_overheating(self) -> bool:
        return self.temperature > 85.0

    def get_temperature_status(self) -> str:
        if self.temperature < 75.0:
            return "NORMAL"
        elif self.temperature <= 85.0:
            return "WARNING"
        else:
            return "CRITICAL"


if __name__ == "__main__":
    machine = IndustrialMachine("MACHINE-001")

    readings = []

    for _ in range(1000):
        reading = machine.generate_reading()
        readings.append(reading)

    print("Generated readings:", len(readings))

    df = pd.DataFrame([asdict(reading) for reading in readings])

    print(df.head())

    df.to_csv("sensor_data.csv", index=False)

    print("Data saved successfully!")
    print("Temperature status:", machine.get_temperature_status())
    print("Is overheating:", machine.is_overheating())
