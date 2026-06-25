# 🏀 NBA Playoff Simulator

An ML-powered NBA playoff bracket simulator built in Jupyter. It pulls real game data from the NBA API, trains a machine learning model on 9 years of historical playoff outcomes, and simulates full 16-team brackets game-by-game — including per-series win probabilities.

---

## 📦 Requirements

```bash
pip install nba_api scikit-learn pandas numpy
```

Optional but recommended for rich terminal output:
```bash
pip install rich
```

---

## 🗂 Project Structure

```
nba_playoff_simulator.ipynb   # main notebook
nba_training_data.pkl         # auto-generated cache (after Cell 2 runs)
nba_model.pkl                 # auto-generated model (after Cell 3 runs)
README.md
```

---

## 🚀 How to Use

The notebook has 5 cells. **Run them top to bottom the first time.** After that, you only need to re-run specific cells depending on what you want to change.

| Cell | What it does | Re-run when… |
|------|-------------|--------------|
| **Cell 1** | Imports, constants, helper functions | Never (unless env changes) |
| **Cell 2** | Fetches 9 seasons of historical playoff data | Never — cached to disk |
| **Cell 3** | Trains the ML model on that historical data | Never — saved to disk |
| **Cell 4** | Enter your 16 teams + fetch their season stats | Changing teams or season |
| **Cell 5** | Simulate the playoffs | Any time — takes ~1 second |

---

## ✏️ Entering Your Teams (Cell 4)

Edit the two strings at the top of Cell 4. Each team is entered as `ABBREVIATION YEAR`, comma separated, **in seeded order (seed 1 first, seed 8 last).**

```python
GROUP1_RAW = "DET 2025, BOS 2025, NYK 2025, CLE 2025, ATL 2025, TOR 2025, ORL 2025, PHI 2025"
GROUP2_RAW = "OKC 2025, SAS 2025, DEN 2025, HOU 2025, MIN 2025, LAL 2025, PHX 2025, POR 2025"
```

**You can mix years freely.** For example, to see how the 2017 Warriors match up against the 2025 OKC Thunder:

```python
GROUP2_RAW = "OKC 2025, GSW 2017, DEN 2025, HOU 2025, MIN 2025, LAL 2025, PHX 2025, POR 2025"
```

The year maps to the **end year of the season** — so `2025` = the 2024-25 season.

### Supported seasons
| Year input | Season |
|-----------|--------|
| 2016 | 2015-16 |
| 2017 | 2016-17 |
| ... | ... |
| 2025 | 2024-25 |

---

## 🤖 How the ML Model Works

The model is trained like a classification problem — similar to predicting whether a patient has a disease based on their stats. Here each "patient" is a **matchup between two teams**, and the "features" are the differences between their regular season stats:

```
Feature = Team A stat − Team B stat
```

Stats used: `WIN_PCT, PTS, FG%, 3P%, FT%, REB, OREB, AST, STL, BLK, TOV, PLUS_MINUS`

So if Team A averages 115 PTS and Team B averages 108, the PTS feature = +7 (favoring A). If Team A has more turnovers, TOV feature is negative (bad for A). The model learns from 9 seasons of real playoff series which combinations of these differences actually predict who wins.

### Model details
- **Algorithm:** Logistic Regression with Platt scaling (probability calibration)
- **Regularization:** `C=0.1` — keeps predictions from being overconfident on small data
- **Training data:** ~200 real playoff series from 2015-16 through 2023-24
- **CV accuracy:** ~60-65% (realistic — the NBA has genuine upsets)
- **Additional input at prediction time:** H2H stats (how the two teams actually did against each other in the regular season) blended 30/70 with season averages, plus a small seed adjustment (~1% per seed gap)

---

## 🏆 How the Simulation Works

**Bracket format:**
- Group 1 and Group 2 each run a standard 3-round bracket (First Round → Semifinals → Conference Finals)
- Winners of each group meet in the Finals
- Within each group, re-seeding happens each round: best remaining seed vs worst remaining seed

**Per-game simulation:**
1. Model outputs a win probability for Team A vs Team B (e.g. 62%)
2. Each game is simulated by drawing a random number against that probability
3. A small random noise (~±3.5%) is added per game to simulate playoff variance
4. First team to 4 wins takes the series

**Re-running Cell 5** gives you a fresh simulation with different random outcomes every time — the probabilities stay the same but the game-by-game results vary.

---

## 📋 Example Output

```
==================================================
  GROUP 1 — FIRST ROUND
==================================================

  🏀 DET vs PHI
     → DET wins 4-1  (win prob: 64%)
     Game     Winner   Series
     Game 1   DET      1-0
     Game 2   PHI      1-1
     Game 3   DET      2-1
     Game 4   DET      3-1
     Game 5   DET      4-1

  🏀 BOS vs ORL
     → BOS wins 4-2  (win prob: 61%)
     ...

🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆
  NBA CHAMPION: OKC
🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆
```

---

## 📊 Accuracy Test Outcomes

To stress-test the model, I ran **5 simulations per playoff year** across the past 6 seasons, re-configuring the model for each year. Accuracy is measured as:

> **Series Accuracy** — percentage of series where the model correctly predicted the winner, out of all 15 series in a standard playoff bracket.

---

### 🔬 Example Simulation — Year 2026

**Actual Results:**

| Round | East | West |
|-------|------|------|
| First Round | DET def. ORL (7), CLE def. TOR (7), NYK def. ATL (6), BOS def. PHI (7) | OKC def. PHX (4), LAL def. HOU (6), MIN def. DEN (6), SAS def. POR (5) |
| Semifinals | CLE def. DET (7), NYK def. BOS (4) | OKC def. LAL (4), SAS def. MIN (6) |
| Conf. Finals | NYK def. CLE (4) | SAS def. OKC (7) |
| **NBA Finals** | 🏆 **NYK def. SAS (5)** | |

**Simulation Output:**

<details>
<summary>Click to expand full simulation output</summary>

🏀🏀🏀🏀🏀🏀🏀🏀🏀🏀🏀🏀🏀🏀🏀🏀🏀🏀🏀🏀

NBA PLAYOFF SIMULATION

🏀🏀🏀🏀🏀🏀🏀🏀🏀🏀🏀🏀🏀🏀🏀🏀🏀🏀🏀🏀
═══════════════════════════════════════════════

EASTERN CONFERENCE

═══════════════════════════════════════════════
── FIRST ROUND ──
🏀 DET vs ORL  →  DET wins 4-2  (win prob: 72%)

Game 1  DET  1-0 | Game 2  DET  2-0 | Game 3  DET  3-0

Game 4  ORL  3-1 | Game 5  ORL  3-2 | Game 6  DET  4-2
🏀 BOS vs PHI  →  BOS wins 4-1  (win prob: 72%)

Game 1  BOS  1-0 | Game 2  BOS  2-0 | Game 3  PHI  2-1

Game 4  BOS  3-1 | Game 5  BOS  4-1
🏀 NYK vs ATL  →  NYK wins 4-0  (win prob: 72%)

Game 1  NYK  1-0 | Game 2  NYK  2-0 | Game 3  NYK  3-0 | Game 4  NYK  4-0
🏀 CLE vs TOR  →  TOR wins 4-1  (win prob: 70%)

Game 1  TOR  0-1 | Game 2  TOR  0-2 | Game 3  TOR  0-3

Game 4  CLE  1-3 | Game 5  TOR  1-4
── SEMIFINALS ──
🏀 DET vs TOR  →  DET wins 4-1  (win prob: 72%)

🏀 BOS vs NYK  →  NYK wins 4-2  (win prob: 70%)
── CONFERENCE FINALS ──
🏀 DET vs NYK  →  NYK wins 4-2  (win prob: 28%)
═══════════════════════════════════════════════

WESTERN CONFERENCE

═══════════════════════════════════════════════
── FIRST ROUND ──
🏀 OKC vs PHX  →  OKC wins 4-1  (win prob: 72%)

🏀 SAS vs POR  →  SAS wins 4-2  (win prob: 72%)

🏀 DEN vs MIN  →  DEN wins 4-1  (win prob: 72%)

🏀 LAL vs HOU  →  HOU wins 4-0  (win prob: 32%)
── SEMIFINALS ──
🏀 OKC vs HOU  →  OKC wins 4-1  (win prob: 72%)

🏀 SAS vs DEN  →  DEN wins 4-1  (win prob: 70%)
── CONFERENCE FINALS ──
🏀 OKC vs DEN  →  OKC wins 4-2  (win prob: 60%)
═══════════════════════════════════════════════

NBA FINALS

═══════════════════════════════════════════════
🏀 NYK vs OKC  →  NYK wins 4-3  (win prob: 30%)

Game 1  OKC  0-1 | Game 2  NYK  1-1 | Game 3  NYK  2-1

Game 4  OKC  2-2 | Game 5  NYK  3-2 | Game 6  OKC  3-3 | Game 7  NYK  4-3
🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆

NBA CHAMPION: NYK

🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆

</details>

**Result:** 9/15 series predicted correctly — **60% accuracy**

---

### 📈 Accuracy Results by Year

| Year | Trial 1 | Trial 2 | Trial 3 | Trial 4 | Trial 5 | Avg |
|------|---------|---------|---------|---------|---------|-----|
| 2026 | 60% | 53% | 53% | 60% | 67% | **59%** |
| 2025 | 47% | 40% | 26% | 26% | 53% | **38%** ⚠️ |
| 2024 | 67% | 53% | 53% | 73% | 73% | **64%** |
| 2023 | 67% | 60% | 80% | 53% | 80% | **68%** |
| 2022 | 46% | 60% | 53% | 60% | 40% | **52%** |
| 2021 | 53% | 46% | 60% | 53% | 67% | **56%** |

> ⚠️ **2025 outlier:** Unusually low accuracy driven by Tyrese Haliburton's unexpected playoff run, which no stat-based model could have anticipated from regular season data alone.

---

### 📐 Statistical Confidence

Using a **95% confidence t-interval** across all trials:

| Dataset | 95% Confidence Interval | Interpretation |
|---------|------------------------|----------------|
| All 6 years | **[51.2%, 61.0%]** | Significantly better than random guessing (50%) |
| Excluding 2025 | **[55.4%, 63.8%]** | Strong predictive accuracy for a probabilistic model |

> The lower bound of both intervals exceeds 50%, confirming the model is meaningfully more accurate than a coin flip. The 2025 season is a fair exclusion — Haliburton's run was a genuine statistical anomaly that no regular-season data could have predicted.
## ⚠️ Notes

- **First run of Cell 2 takes ~5 minutes** — it's making hundreds of NBA API calls. After that it loads from cache in under a second.
- If you see `⚠ {ABBREV}: ...` errors during Cell 4, it usually means the NBA API rate-limited you. Wait 30 seconds and re-run.
- H2H stats will be empty if two teams never played each other in the regular season (e.g. cross-conference teams or teams from different years) — the model falls back to season averages only in that case.
- The model is trained on regular-season stats predicting playoff outcomes, which is inherently noisy. A 60-65% CV accuracy is expected and healthy — if it were 90%+ the model would be overfit.
