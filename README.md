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

## Detekcja anomalii

[detector/anomaly.py](detector/anomaly.py) to detektor hybrydowy uczony tylko na
zdrowych danych: Isolation Forest + reguła z-score. Cechy to surowe odczyty oraz
średnie kroczące temperatury i wibracji (wykrywają powolny dryf).

```bash
python -m detector.anomaly --fault overheating
```

Sam Isolation Forest słabo wykrywał awarię `overheating` (recall 0.26–0.59 zależnie
od ziarna), bo opiera się na rangach wartości, a nie na wielkości odchylenia
pojedynczego kanału. Dodanie reguły z-score podnosi recall do ok. 0.99 przy
niskim odsetku fałszywych alarmów.

## Testy

```bash
pytest
```

