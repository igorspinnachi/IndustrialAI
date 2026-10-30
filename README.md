# industrial-ai

Symulator czujników maszyny przemysłowej generujący dane szeregów czasowych
(temperatura, wibracje, ciśnienie, prędkość obrotowa) z opcjonalnymi awariami
i etykietami anomalii. Docelowo baza pod detekcję anomalii i predictive maintenance.

## Instalacja

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Użycie

```bash
# 1000 zdrowych odczytów
python -m simulator.sensors --seed 42

# z awarią łożyska od kroku 500
python -m simulator.sensors -n 2000 --fault bearing_wear --fault-start 500 --output data.csv
```

W kodzie:

```python
from simulator.sensors import simulate

df = simulate(1000, seed=42, fault="overheating", fault_start=600)
```

Kolumny: `machine_id, timestamp, temperature, vibration, pressure,
rotational_speed, is_anomaly, fault`.

## Awarie

| Awaria         | Efekt                                              |
|----------------|----------------------------------------------------|
| `bearing_wear` | narastające wibracje i lekki wzrost temperatury    |
| `overheating`  | szybko narastająca temperatura                     |

## Testy

```bash
pytest
```

