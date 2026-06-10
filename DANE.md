# Dane eksperymentalne (poza repozytorium)

Wszystkie wyniki numeryczne sa **w pelni odtwarzalne** z kodu i ziaren PRNG
(przebieg `i` uzywa ziarna `i`, `i = 1..30`) wg instrukcji w [`SCRIPTS.md`](SCRIPTS.md).

## Archiwum

- **Nazwa pliku:** `wae-end-<nr_zadania>-2026l-329923-327394-dane.zip`
- **Link (OneDrive):** _TODO: wstawic link_

## Zawartosc (rozpakowac do korzenia repozytorium)

Archiwum odtwarza drzewa `results/` oraz `analysis/output/`:

### Strojenie hiperparametrow (sweepy)

| Plik | Opis |
|------|------|
| `results/sweep_10.csv`, `results/sweep_30.csv` | strojenie `comparison_quantile` (MSR) i `z_star` (PSR), per ziarno (Etap 1) |
| `results/sweep_pop.csv` | strojenie rozmiaru populacji `NP` (Etap 2) |

### Eksperyment finalny

| Plik | Opis |
|------|------|
| `results/final/<algo>_D<dim>_summary.csv` | wynik koncowy kazdego przebiegu: `error`, `n_evals`, `solved`, czas |
| `results/final/<algo>_D<dim>_history.csv` | historia zbieznosci (`best_error`) oraz wspolczynnik `factor` (F), probkowane co kilka generacji |

gdzie `algo` in {`base`, `MSR-population`, `PSR-population`, `MSR-trial`, `PSR-trial`},
`dim` in {`10`, `30`}.

### Analiza (tabele i wykresy)

| Sciezka | Opis |
|---------|------|
| `analysis/output/tables/descriptive_stats.csv` | statystyki opisowe (mediana, srednia, std, best, worst) per (algorytm, wymiar, funkcja) |
| `analysis/output/tables/wilcoxon.csv` | testy Wilcoxona dla hipotez H1--H3 (mediana i srednia) |
| `analysis/output/tables/best_of_wins.csv`, `best_of_per_function.csv` | analiza best-of-runs |
| `analysis/output/figures/boxplots/D{10,30}/f*.png` | boxploty rozkladu bledu (29 funkcji x 2 wymiary) |
| `analysis/output/figures/convergence/D{10,30}/f*.png` | krzywe zbieznosci (mediana + kwantyle) |
| `analysis/output/figures/factor/D{10,30}/f*.png` | ewolucja wspolczynnika F (mediana + kwantyle) |

## Odtworzenie danych bez pobierania archiwum

```bash
python -m experiments.run_experiment \
    --algos base MSR-population PSR-population MSR-trial PSR-trial \
    --seeds 30 --dims 10 30 --out-dir results/final
```

Pelny pipeline -- strojenie (sweepy), eksperyment finalny oraz wszystkie tabele
i wykresy -- krok po kroku opisuje [`SCRIPTS.md`](SCRIPTS.md).
