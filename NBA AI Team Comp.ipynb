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

## ⚠️ Notes

- **First run of Cell 2 takes ~5 minutes** — it's making hundreds of NBA API calls. After that it loads from cache in under a second.
- If you see `⚠ {ABBREV}: ...` errors during Cell 4, it usually means the NBA API rate-limited you. Wait 30 seconds and re-run.
- H2H stats will be empty if two teams never played each other in the regular season (e.g. cross-conference teams or teams from different years) — the model falls back to season averages only in that case.
- The model is trained on regular-season stats predicting playoff outcomes, which is inherently noisy. A 60-65% CV accuracy is expected and healthy — if it were 90%+ the model would be overfit.
