# Reprodukcja eksperymentów

Instrukcja krok-po-kroku, jak odtworzyć każdą składową projektu: strojenie
hiperparametrów, eksperyment finalny oraz wszystkie tabele i wykresy.
Wszystkie komendy trzeba uruchamiać z katalogu głównego repozytorium.

## 0. Środowisko

Wymagany Python 3.11+ oraz zależności z `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ziarna PRNG

Eksperymenty są w pełni deterministyczne. Każdy przebieg `i` używa ziarna `i`;
dla `--seeds N` uruchamiane są ziarna **1..N**. W eksperymencie finalnym użyto
**30 ziaren (1..30)** dla każdej pary (funkcja, wymiar). Nie ma więc osobnej
listy ziaren - wynika ona wprost z indeksu przebiegu.

---

## Etap 1 - Szukanie najlepszych parametrów

Dla algorytmów z MSR → `comparison_quantile`, PSR → `z_star`.

```bash
# D=10 (domyślnie 15 ziaren)
python -m experiments.sweep --dims 10 --seeds 15 --out results/sweep_10.csv
# D=30 (nocny przebieg)
python -m experiments.sweep --dims 30 --seeds 15 --out results/sweep_30.csv
```

Wyjście: po jednym wierszu CSV na przebieg (config, hp_value, func_id, dim,
seed, error, ...).

## Etap 2 - Szukanie rozmiaru populacji dla wyznaczonych wcześniej parametrów

Generowanie wyników dla różnych rozmiarów populacji.

```bash
python -m experiments.sweep_pop --dims 10 30 --seeds 10 --out results/sweep_pop.csv
```

## Etap 3 - Wybór hiperparametrów z sweepów

Agregacja po randze hiperparametrów oraz ilości osobników w populacji.

```bash
# wybór comparison_quantile / z_star (mean-rank po D10 i D30) -> q=0.45, z*=0.10
python -m analysis.sweep_analysis results/sweep_10.csv results/sweep_30.csv
# wybór NP (domyślnie czyta results/sweep_pop.csv) -> D10=100; D30 base=100, adaptive=150
python -m analysis.pop_size_analysis
```

Wybrane wartości są zapisane na stałe w `experiments/run_experiment.py`
(`ALGO_SPECS`, `POP_SIZE`).

---

## Etap 4 - Eksperyment finalny

Jeden przebieg pełen przebieg dla każdej z 29 funkcji, D10 oraz D30, 30 ziaren od 1 do 30.

Uruchamialiśmy dla wszystkich 5 algorytmó (3 główne + 2 warianty trial jako dodatkowy nieudany eksperyment).

```bash
python -m experiments.run_experiment \
    --algos base MSR-population PSR-population MSR-trial PSR-trial \
    --seeds 30 --dims 10 30 --out-dir results/final
```

Wyjście (w `results/final/`), per (algorytm, wymiar):
- `<algo>_D<dim>_summary.csv` - wynik końcowy każdego przebiegu,
- `<algo>_D<dim>_history.csv` - historia zbieżności + współczynnik F.

## Etap 5 - Analiza i wykresy (z `results/final`)

```bash
python -m analysis.final_stats        # tabele: statystyki opisowe, testy Wilcoxona dla hipotez, najlepsze wyniki dla funkcji
python -m analysis.plot_boxplots      # boxploty rozkładu błędu dla każdej funkcji
python -m analysis.plot_convergence   # krzywe zbieżności (mediana + kwantyle)
python -m analysis.plot_factor        # ewolucja współczynnika F (mediana + kwantyle)
```

Wyjście:
- `analysis/output/tables/*.csv` - `descriptive_stats`, `wilcoxon`,
  `best_of_wins`, `best_of_per_function`,
- `analysis/output/figures/{boxplots,convergence,factor}/D{10,30}/f{n}.png`.

---