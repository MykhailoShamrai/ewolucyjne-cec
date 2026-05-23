# DE/rand/1/bin z adaptacją F (MSR i PSR) — Zadanie 16

## Struktura projektu

```
src/              — kod źródłowy algorytmów
  de_base.py      — bazowy DE/rand/1/bin
  de_msr.py       — wariant z MSR
  de_psr.py       — wariant z PSR
  utils.py        — helpers (boundary handling, logging)

experiments/      — skrypty eksperymentalne
  run_all.py      — uruchomienie pełnych eksperymentów
  config.py       — parametry eksperymentów (NP, CR, F, budżet, ziarna)

analysis/         — analiza wyników
  statistics.py   — testy statystyczne, tabele
  plots.py        — wykresy zbieżności, przebiegu F

results/          — dane z eksperymentów (CSV/JSON)
figures/          — wygenerowane wykresy
docs/             — dokumentacja końcowa (Springer Nature)
```

## Uruchomienie

```bash
pip install -r requirements.txt
python experiments/run_all.py
python analysis/plots.py
python analysis/statistics.py
```

## Wymagania

- Python 3.10+
- cecpy (https://codeberg.org/ewarchul/cecxx)
- numpy, scipy, matplotlib, pandas
