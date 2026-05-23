# Roadmap — Zadanie 16: DE/rand/1/bin z adaptacją F (MSR i PSR)

**Benchmark:** CEC 2017 (29 funkcji, f2 zdeprecjonowana), D ∈ {10, 30}  
**Biblioteka CEC:** `cecxx` / `cecpy` (https://codeberg.org/ewarchul/cecxx) — wymaganie prowadzącego  
**Autorzy:** Mykhailo Shamrai (329923), Grzegorz Prasek (327394)  
**Budżet ewaluacji:** $10^4 \cdot D$ na uruchomienie  
**Liczba powtórzeń:** 30 (zgodnie z protokołem CEC)  

---

## Etap 0: Środowisko i zależności

### 0.1 Inicjalizacja repozytorium git
- `git init`, pierwszy commit ze strukturą katalogów i `.gitignore`

### 0.2 Środowisko Python
- Utworzyć `venv` (Python 3.10+)
- `pip install -r requirements.txt`

### 0.3 Instalacja cecpy (Python bindings do cecxx)
- Sklonować https://codeberg.org/ewarchul/cecxx
- Zainstalować bindingi Python zgodnie z `docs/bindings/python.md`
- Jeśli `pip install` z pyproject.toml nie działa → build z CMake + pybind11
- Upewnić się, że dane benchmarkowe (macierze rotacji, przesunięcia) są zainstalowane

### 0.4 Weryfikacja
- Napisać skrypt `tests/test_cec.py`:
  - Wywołać `cec2017(func_id=1, x=[0]*10)` — sprawdzić czy zwraca liczbę
  - Wywołać dla D=10 i D=30
  - Porównać z wartością `f*(1) = 100` (znane optimum f1 w CEC 2017)
  - Sprawdzić czy `f(x*) - f* ≈ 0` dla znanych optimów

### ✅ Checkpoint 0
> cecpy działa, funkcje CEC 2017 zwracają poprawne wartości dla obu wymiarów

---

## Etap 1: Bazowy DE/rand/1/bin ze stałym F

### 1.1 Implementacja algorytmu (`src/de_base.py`)
- Inicjalizacja populacji: `U(lb, ub)^D`, rozmiar NP
- Mutacja: `v_i = x_{r1} + F * (x_{r2} - x_{r3})`, r1≠r2≠r3≠i
- Krzyżowanie binarne: `CR`, j_rand zapewniający co najmniej 1 gen z mutanta
- Selekcja zachłanna: `if f(trial) <= f(target) then replace`
- Obsługa ograniczeń: clipping do `[-100, 100]`
- Warunek stopu: wyczerpanie budżetu ewaluacji

### 1.2 Logowanie
- Dla każdej generacji zapisywać: `best_f`, `mean_f`, `median_f`, `worst_f`, `std_f`, `F_t`, `n_evals`
- Zapisywać pełną historię zbieżności (nie tylko końcowy wynik)

### 1.3 Testy jednostkowe (`tests/test_de_base.py`)
- Test na funkcji sferycznej (`sum(x^2)`) — czy zbiega do 0?
- Test na Rastrigin — czy zbiega do okolic optimum?
- Test deterministyczności — ten sam seed → ten sam wynik

### 1.4 Walidacja na CEC 2017
- Uruchomić bazowy DE na f1, f3, f10 (unimodal, multimodal) dla D=10
- 5 powtórzeń, narysować krzywe zbieżności
- Sprawdzić czy `error = f(x_best) - f*` maleje z generacjami

### ✅ Checkpoint 1
> Bazowy DE zbiega na prostych funkcjach. Krzywe zbieżności maleją. Seed daje powtarzalne wyniki.

---

## Etap 2: DE-MSR (Median Success Rule)

### 2.1 Zrozumienie MSR
- Źródło: Kern et al. / Beyer & Sendhoff — MSR w kontekście strategii ewolucyjnych
- Zasada: po każdej generacji liczymy odsetek potomków lepszych od mediany fitness rodziców
- Jeśli `success_rate > target` → zwiększ F (za mała eksploracja)
- Jeśli `success_rate < target` → zmniejsz F (za duża eksploracja)
- Aktualizacja multiplikatywna: `F_{t+1} = F_t * exp(damping * (success_rate - target))`

### 2.2 Implementacja (`src/de_msr.py`)
- Dziedziczy z `DEBase`, nadpisuje `get_F()` i `update_F()`
- Parametry: `F_init`, `F_min`, `F_max`, `target_success` (domyślnie 1/5 lub 1/2 — do zbadania), `damping`
- Clipping F do `[F_min, F_max]`

### 2.3 Testy i walidacja
- Uruchomić DE-MSR na tych samych funkcjach co Etap 1.4
- Narysować: (a) krzywe zbieżności DE-base vs DE-MSR, (b) przebieg F_t w czasie
- Sprawdzić: czy F rośnie na łatwych funkcjach (dobra eksploracja), maleje na trudnych?

### ✅ Checkpoint 2
> DE-MSR adaptuje F sensownie. Na f1 (łatwa) F rośnie lub stabilizuje. Na f10 (trudniejsza) F zachowuje się inaczej. Zbieżność co najmniej porównywalna z bazowym DE.

---

## Etap 3: DE-PSR (Population Success Rule)

### 3.1 Zrozumienie PSR
- Źródło: Hansen — PSR w kontekście CMA-ES
- Zasada: łączymy starą i nową populację (2N osobników), sortujemy po fitness
- Liczymy średnią rangę starej vs nowej populacji
- Sygnał sukcesu: `z = (mean_rank_old - mean_rank_new) / N`
- Aktualizacja: `F_{t+1} = F_t * exp(damping * z)`
- Kluczowa własność: PSR jest **invariant na ranking** (nie zależy od skali f)

### 3.2 Implementacja (`src/de_psr.py`)
- Dziedziczy z `DEBase`, nadpisuje `get_F()` i `update_F()`
- Parametry: `F_init`, `F_min`, `F_max`, `damping`
- Obliczanie rang: `scipy.stats.rankdata` lub ręcznie przez `argsort(argsort(...))`

### 3.3 Testy i walidacja
- Analogicznie do Etapu 2.3
- Dodatkowy test: uruchomić na `f(x)` i `1000*f(x)` — PSR powinien dać identyczny przebieg F (rank-invariance)

### ✅ Checkpoint 3
> DE-PSR adaptuje F. Przebieg F jest niezależny od skalowania funkcji (rank-invariance potwierdzona). Zbieżność co najmniej porównywalna z bazowym.

---

## Etap 4: Framework eksperymentalny

### 4.1 Runner (`experiments/run_all.py`)
- Parametry z `experiments/config.py`:
  - Algorytmy: `[DE-base, DE-MSR, DE-PSR]`
  - Funkcje: CEC 2017 f1–f30 (bez f2), tj. 29 funkcji
  - Wymiary: D ∈ {10, 30}
  - Powtórzenia: 30 (ziarna: 1, 2, ..., 30)
  - Budżet: `10^4 * D` ewaluacji
  - NP=100, CR=0.9, F_static=0.5, F_init=0.5
- Łącznie: 3 × 29 × 2 × 30 = **5220 uruchomień**

### 4.2 Format wyników (`results/`)
- Jeden plik CSV per (algorytm, wymiar): np. `DE-base_D10.csv`
  - Kolumny: `func_id, seed, best_f, error, n_evals`
- Historia zbieżności: `results/convergence/DE-base_D10_f1_seed1.csv`
  - Kolumny: `generation, n_evals, best_f, mean_f, F_t`
- Plik z ziarnami: `results/seeds.json`

### 4.3 Skrypt odtwarzający (`experiments/reproduce.py`)
- Odtwarza wszystkie eksperymenty i wykresy od zera
- Argument `--quick` — tylko 3 funkcje, 5 powtórzeń (do testowania)

### 4.4 Mini-test pipeline
- `python experiments/run_all.py --quick` → 3 funkcje × 3 algo × 5 seeds × 1 wymiar
- Sprawdzić: pliki CSV się tworzą, brak NaN, dane mają sens

### ✅ Checkpoint 4
> Pipeline end-to-end działa. Mini-eksperyment generuje poprawne pliki wynikowe.

---

## Etap 5: Pełne eksperymenty numeryczne

### 5.1 Uruchomienie
- `python experiments/run_all.py` (pełne 5220 uruchomień)
- Szacowany czas: zależy od sprzętu, ale z cecxx powinno być szybkie
- Rozważyć paralelizację (`multiprocessing` / `joblib`)

### 5.2 Walidacja danych
- Sprawdzić kompletność: 29 funkcji × 30 seeds × 3 algo × 2 dims = 5220 wierszy
- Sprawdzić: brak NaN/inf, `error >= 0` (bo CEC definiuje `error = f(x) - f*`)
- Sprawdzić: `n_evals <= max_evals` dla każdego uruchomienia
- Porównać f* z tabelą oficjalnych optimów CEC 2017

### 5.3 Backup danych
- Jeśli `results/` > 5MB → przygotować oddzielne archiwum (OneDrive)
- Jeśli <= 5MB → zostaje w repozytorium

### ✅ Checkpoint 5
> Komplet danych: 5220 uruchomień, brak anomalii, dane spójne.

---

## Etap 6: Analiza statystyczna i wizualizacje

### 6.1 Tabele zbiorcze (`analysis/statistics.py`)
- Dla każdej pary (algorytm, wymiar) i każdej funkcji:
  - mean, median, std, best (min), worst (max) błędu końcowego
- Format: tabela LaTeX gotowa do wklejenia do raportu

### 6.2 Testy statystyczne
- **Test Wilcoxona** (signed-rank, pairwise):
  - DE-base vs DE-MSR, DE-base vs DE-PSR, DE-MSR vs DE-PSR
  - Dla każdej funkcji i wymiaru osobno
  - Raportować p-value, oznaczać `+/=/−` przy α=0.05
- **Test Friedmana** + post-hoc Nemenyi:
  - Ranking zbiorczy po wszystkich funkcjach
  - Critical difference diagram

### 6.3 Wykresy (`analysis/plots.py`)
- **Convergence curves**: `error` vs `n_evals`, uśrednione po 30 seedach, z wstęgą std
  - Osobno dla D=10 i D=30
  - Wybrane reprezentatywne funkcje (np. f1 unimodal, f10 multimodal, f20 hybrid, f25 composition)
- **Przebieg F w czasie**: F_t vs generacja
  - Uśredniony po seedach, z wstęgą std
  - Te same reprezentatywne funkcje
- **Analiza per typ funkcji CEC 2017**:
  - Unimodal: f1, f3
  - Multimodal proste: f4–f10
  - Hybrid: f11–f20
  - Composition: f21–f30
  - Boxplot średniego F lub błędu per kategoria

### 6.4 Wnioski wstępne
- Która metoda adaptacji jest lepsza? Na jakich typach funkcji?
- Czy adaptacja F daje statystycznie istotną poprawę vs stałe F?
- Jak zachowuje się F — czy stabilizuje, oscyluje, dryfuje?

### ✅ Checkpoint 6
> Tabele i wykresy są spójne, czytelne, dają się zinterpretować. Testy statystyczne mają sens.

---

## Etap 7: Dokumentacja końcowa

### 7.1 Szablon
- Pobrać szablon Springer Nature (LNCS) — LaTeX
- Umieścić w `docs/`

### 7.2 Struktura raportu (max 8 stron A4)
1. **Introduction** (~0.5 strony) — motywacja, cel
2. **Background** (~1 strona) — DE/rand/1/bin, MSR, PSR — formalne definicje
3. **Proposed approach** (~1 strona) — jak zaadaptowaliśmy MSR/PSR do DE, pseudokod
4. **Experimental setup** (~1 strona) — CEC 2017, parametry, protokół, cecxx
5. **Results** (~2.5 strony) — tabele, wykresy, testy statystyczne
6. **Discussion** (~1 strona) — interpretacja, zachowanie F, typy funkcji
7. **Conclusion** (~0.5 strony)
8. **References**
- Jeśli są różnice vs dokumentacja wstępna → dodać sekcję "Changes from initial plan"
- Jeśli użyto LLM → dodać stronę z wyszczególnieniem (nie wlicza się w 8 stron)

### 7.3 Skrypt reprodukcji
- `experiments/reproduce.py` — odtwarza eksperymenty
- `analysis/plots.py` — generuje wszystkie wykresy
- `analysis/statistics.py` — generuje tabele
- Udokumentować w README: `pip install -r requirements.txt && python experiments/reproduce.py`

### ✅ Checkpoint 7
> PDF kompiluje się, ≤ 8 stron, wykresy czytelne, tabele poprawne, skrypt reprodukcji działa.

---

## Etap 8: Pakowanie i oddanie

### 8.1 Nazewnictwo
- `wae-end-16-2026l-329923-327394.pdf` — raport
- `wae-end-16-2026l-329923-327394.bundle` — git bundle

### 8.2 Git bundle
- `git bundle create wae-end-16-2026l-329923-327394.bundle --all`
- Upewnić się, że historia jest czysta (brak wrażliwych danych, klucze, hasła)

### 8.3 Przenośność
- Sklonować bundle na czystą maszynę (Linux)
- `pip install -r requirements.txt`
- Uruchomić skrypt reprodukcji — czy działa?

### 8.4 Przygotowanie do prezentacji
- 8–10 minut, niekoniecznie multimedialna
- Kluczowe slajdy: problem → podejście → wyniki → wnioski
- Być gotowym na pytania: co to MSR/PSR, dlaczego takie wyniki, co by zmienili

### ✅ Checkpoint 8
> Bundle działa na Linux. PDF poprawnie nazwany. Gotowi do prezentacji.
|
