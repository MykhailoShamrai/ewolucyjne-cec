# Adaptacja parametru mutacji F w DE/rand/1/bin regułami MSR i PSR

Modyfikacja algorytmu ewolucji różnicowej **DE/rand/1/bin**, w której stały
współczynnik mutacji `F` zastępujemy adaptacją sterowaną regułą sukcesu:
**Median Success Rule (MSR)** oraz **Population Success Rule (PSR)**. Obie
warianty porównujemy z bazowym DE na pełnym benchmarku **CEC 2017** (29 funkcji,
wymiary `D ∈ {10, 30}`, 30 niezależnych ziaren).

## Układ repozytorium

```
src/                 implementacja algorytmów
  de_base.py         DE/rand/1/bin ze stałym F (punkt odniesienia)
  de_msr.py          DEMSR  – adaptacja F regułą Median Success Rule
  de_psr.py          DEPSR  – adaptacja F regułą Population Success Rule
  success_rule.py    wspólny mechanizm wygładzania sygnału i aktualizacji F
  cec_problem.py     adapter funkcji CEC 2017 do interfejsu DE

experiments/         uruchamianie eksperymentów
  config.py          stałe protokołu (TOL, MaxFES, siatki HP, F, CR, ...)
  sweep.py           strojenie comparison_quantile (MSR) i z_star (PSR)
  sweep_pop.py       strojenie rozmiaru populacji NP
  run_experiment.py  eksperyment finalny (wybrane HP zaszyte w ALGO_SPECS/POP_SIZE)

analysis/            agregacja wyników -> tabele i wykresy
results/             dane wyjściowe przebiegów (poza repo – patrz DANE.md)
cecxx/               submoduł: biblioteka benchmarków CEC
tests/               testy jednostkowe (pytest)
```

## Co generuje który skrypt

Wszystkie komendy uruchamiać z katalogu głównego repo. Kolejność i dokładne
argumenty: **[`SCRIPTS.md`](SCRIPTS.md)**.

### Eksperymenty (`experiments/`)

| Skrypt | Generuje | Opis |
|--------|----------|------|
| `python -m experiments.sweep` | `results/sweep_{10,30}.csv` | jeden wiersz na przebieg dla siatki `comparison_quantile`/`z_star` (Etap 1) |
| `python -m experiments.sweep_pop` | `results/sweep_pop.csv` | przebiegi dla różnych `NP` przy wybranych HP (Etap 2) |
| `python -m experiments.run_experiment` | `results/final/<algo>_D<dim>_{summary,history}.csv` | eksperyment finalny: wynik końcowy + historia zbieżności i `F` (Etap 4) |

### Analiza (`analysis/`)

| Skrypt | Generuje | Opis |
|--------|----------|------|
| `python -m analysis.sweep_analysis` | `results/analysis/{per_param_stats,per_function_top,recommended_param}.csv` | wybór `q`/`z*` po średniej randze (Etap 3) |
| `python -m analysis.pop_size_analysis` | `results/analysis/pop_size/*.csv` | wybór `NP` per (algorytm, wymiar) (Etap 3) |
| `python -m analysis.final_stats` | `analysis/output/tables/{descriptive_stats,wilcoxon,best_of_wins,best_of_per_function}.csv` | statystyki opisowe (mediany – Tabela A1), testy Wilcoxona H1–H3, best-of-runs |
| `python -m analysis.plot_boxplots` | `analysis/output/figures/boxplots/D{10,30}/f*.png` | boxploty rozkładu błędu |
| `python -m analysis.plot_convergence` | `analysis/output/figures/convergence/D{10,30}/f*.png` | krzywe zbieżności (mediana + kwantyle) |
| `python -m analysis.plot_factor` | `analysis/output/figures/factor/D{10,30}/f*.png` | ewolucja współczynnika `F` |

## Reprodukcja i dane

- **[`SCRIPTS.md`](SCRIPTS.md)** – pełny pipeline krok po kroku (strojenie →
  eksperyment finalny → tabele i wykresy), wraz z konwencją ziaren PRNG.
- **[`DANE.md`](DANE.md)** – opis archiwum z danymi wyjściowymi (`results/`,
  `analysis/output/`), które leży poza repozytorium, oraz jak je odtworzyć.

Eksperymenty są w pełni deterministyczne: przebieg `i` używa ziarna `i`
(`i = 1..30` w eksperymencie finalnym), więc wyniki odtwarzają się dokładnie.