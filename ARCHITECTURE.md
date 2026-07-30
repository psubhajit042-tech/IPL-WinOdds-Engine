# 🏗️ IPL Win Predictor — System Architecture

---

## 1. System Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          IPL Win Predictor System                       │
│                                                                         │
│  ┌──────────────┐    ┌─────────────────────┐    ┌────────────────────┐  │
│  │  RAW DATA    │───▶│  DATA PREPROCESSING  │───▶│   MODEL TRAINING   │  │
│  │              │    │                     │    │                    │  │
│  │ matches.csv  │    │  load_data()        │    │  GroupShuffleSplit │  │
│  │ deliveries   │    │  preprocess_data()  │    │  by match_id       │  │
│  │  .csv        │    │                     │    │                    │  │
│  └──────────────┘    └─────────┬───────────┘    │  ┌──────────────┐  │  │
│                                │                │  │ Logistic Reg  │  │  │
│                                ▼                │  │ Random Forest│  │  │
│                     ┌──────────────────┐       │  │ XGBoost       │  │  │
│                     │ cleaned_data.csv │       │  └──────────────┘  │  │
│                     │ (90,353 rows)    │       │                    │  │
│                     └────────┬─────────┘       │  GroupKFold CV    │  │
│                              │                 │  (5 folds)         │  │
│                              │                 └────────┬───────────┘  │
│                              │                          │              │
│                              │                          ▼              │
│                              │              ┌──────────────────────┐   │
│                              │              │   SERIALIZED MODELS  │   │
│                              │              │                      │   │
│                              │              │  pipe_lr.pkl         │   │
│                              │              │  pipe_rf.pkl         │   │
│                              │              │  pipe_xgb.pkl        │   │
│                              │              └──────────┬───────────┘   │
│                              │                         │               │
│               ┌──────────────┼─────────────────────────┤               │
│               │              │                         │               │
│               ▼              ▼                         ▼               │
│  ┌────────────────┐  ┌─────────────────┐   ┌──────────────────────┐    │
│  │ VISUALIZATION  │  │   PREDICTOR     │   │  LIVE PREDICTION DEMO│    │
│  │                │  │                 │   │                      │    │
│  │ model_compare  │  │ match_progression│   │  Manual match state │    │
│  │  .png          │  │ (over-by-over)  │   │  → win% / lose%     │    │
│  │                │  │                 │   │                      │    │
│  │ match_981009   │  │ pipe_lr.pkl ────┼──▶│  predict_win_prob() │    │
│  │  _progression  │  │                 │   │                      │    │
│  │  .png          │  │ match_1237181    │   └──────────────────────┘    │
│  │                │  │  _progression    │                               │
│  │ match_1237181  │  │  .png           │                               │
│  │  _progression  │  └─────────────────┘                               │
│  │  .png          │                                                      │
│  └────────────────┘                                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Schema

### Raw — `matches.csv` (1,095 rows)

| Column | Type | Example | Used For |
|--------|------|---------|----------|
| `id` | int | 335982 | Primary key, merges with deliveries |
| `city` | str | "Mumbai" | Feature (venue) |
| `team1` | str | "Mumbai Indians" | Feature (after rename filter) |
| `team2` | str | "Chennai Super Kings" | Feature (after rename filter) |
| `winner` | str | "Mumbai Indians" | Ground truth label |
| `date` | str | "2008-04-18" | Not used in model |
| `venue` | str | "Wankhede Stadium" | Not used (city used instead) |

### Raw — `deliveries.csv` (260,920 rows)

| Column | Type | Example | Used For |
|--------|------|---------|----------|
| `match_id` | int | 335982 | Joins with matches.id |
| `inning` | int | 1 or 2 | Filter: keep 2nd innings only |
| `batting_team` | str | "Mumbai Indians" | Feature |
| `bowling_team` | str | "Chennai Super Kings" | Feature |
| `over` | int | 1–20 | Feature engineering |
| `ball` | int | 1–6 | Feature engineering |
| `total_runs` | int | 4 | Runs scored on this ball |
| `player_dismissed` | str/NaN | "MS Dhoni" | Wicket tracking |

### Cleaned — `cleaned_data.csv` (90,353 rows × 11 columns)

| Column | Type | Role | Example |
|--------|------|------|---------|
| `match_id` | int | **Grouping key** (NOT a feature) | 981009 |
| `batting_team` | str | Categorical feature | "Kolkata Knight Riders" |
| `bowling_team` | str | Categorical feature | "Chennai Super Kings" |
| `city` | str | Categorical feature | "Kolkata" |
| `runs_left` | int | Numerical feature | 10 |
| `balls_left` | int | Numerical feature | 24 |
| `wickets` | int | Numerical feature | 7 |
| `total_runs_x` | int | Numerical feature (1st innings target) | 120 |
| `crr` | float | Numerical feature | 8.25 |
| `rrr` | float | Numerical feature | 2.50 |
| `result` | int | **Label** (1=batting team wins, 0=loses) | 1 |

### After OneHotEncoder (Model Input)

| Encoded | Dimensions | Notes |
|---------|-----------|-------|
| `batting_team` | 7 binary cols | 8 teams − 1 (drop='first') |
| `bowling_team` | 7 binary cols | 8 teams − 1 (drop='first') |
| `city` | ~30+ binary cols | IPL cities − 1 |
| `runs_left`, `balls_left`, `wickets`, `total_runs_x`, `crr`, `rrr` | 6 float cols | Passed through as-is |
| **Total** | **~50+ dimensions** | Varies by number of unique cities |

---

## 3. Anti-Leakage Architecture

### The Problem (old code)

```
          Match 981009 balls ──┬── train (balls 1-60)  ──▶ RF sees these
                                │
                                └── test  (balls 61-120) ──▶ RF "predicts" these
                                                             ║
                                  RF memorizes trajectory    ║
                                  → 99.8% "accuracy"         ║
                                  (not real learning)        ║
```

A naive random row split placed balls from the *same match* in both train and test. Tree models memorized match trajectories, reporting an impossible 99.8% accuracy.

### The Fix (current code)

```
          Match 981009 ──────────────── ALL in TRAIN ──────────────▶ Model
          Match 1237181 ────────────── ALL in TEST  ──────────────▶ Predict
          Match 435672 ──────────────── ALL in TEST  ──────────────▶ Predict
          Match 111203 ──────────────── ALL in TRAIN ─────────────▶ Model
              │
              └── GroupShuffleSplit / GroupKFold by match_id
                  guarantees zero ball overlap per match

          Result: RF drops to 75.6% (honest)
                  LR stays at 77.3% (was already honest)
                  XGBoost: 76.8% (honest)
```

### Design Layers

```
┌───────────────────────────────────────────────────────────┐
│              Anti-Leakage Design Layers                    │
│                                                           │
│  Layer 1: DATA SPLIT                                     │
│  ├── GroupShuffleSplit(test_size=0.2) by match_id         │
│  └── No match contributes balls to both train & test       │
│                                                           │
│  Layer 2: CROSS-VALIDATION                                │
│  ├── GroupKFold(n_splits=5) by match_id                    │
│  └── Each fold: 4/5 of matches train, 1/5 test             │
│                                                           │
│  Layer 3: match_id COLUMN HANDLING                         │
│  ├── Kept in cleaned_data.csv as column 0                  │
│  ├── Extracted as groups before X/y split                   │
│  ├── Dropped from X before any model sees it               │
│  └── Never encoded or used as a predictive feature        │
│                                                           │
│  Result: Tree models can NO LONGER memorize match          │
│          trajectories. Honest generalization measured.    │
└───────────────────────────────────────────────────────────┘
```

---

## 4. Model Pipeline Architecture (Per Model)

```
                    ┌─────────────────────────────────┐
                    │        sklearn Pipeline          │
                    │                                 │
    Raw Input ────▶│  step1: ColumnTransformer        │
    (1 row)        │  ┌───────────────────────────┐  │
                    │  │ OneHotEncoder              │  │
    {               │  │ cols: [batting_team,        │  │
     batting_team,  │  │        bowling_team,        │  │
     bowling_team,  │  │        city]               │  │
     city,          │  │ drop='first'               │  │
     runs_left,     │  │ handle_unknown='ignore'    │  │
     balls_left,    │  │ sparse_output=False        │  │
     wickets,       │  └───────────────────────────┘  │
     total_runs_x,  │  ┌───────────────────────────┐  │
     crr,           │  │ remainder='passthrough'    │  │
     rrr            │  │ (runs_left, balls_left,   │  │
    }               │  │  wickets, total_runs_x,   │  │
                    │  │  crr, rrr passed as-is)   │  │
                    │  └───────────────────────────┘  │
                    │               │                 │
                    │               ▼                 │
                    │  step2: Classifier ◄────────────┤
                    │  ├── LogisticRegression          │
                    │  │   (solver='liblinear')        │
                    │  ├── RandomForest                │
                    │  │   (random_state=42)            │
                    │  └── XGBoost                     │
                    │      (n_estimators=300,           │
                    │       max_depth=4,               │
                    │       lr=0.1, subsample=0.8)     │
                    │               │                 │
                    └───────────────┼─────────────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────────┐
                    │   Output: predict_proba()       │
                    │                                 │
                    │   [P(lose), P(win)]             │
                    │   e.g. [0.0038, 0.9962]         │
                    │        = 0.38% lose, 99.62% win  │
                    └─────────────────────────────────┘
```

---

## 5. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Group-aware split by match_id** | Prevents same-match trajectory leakage (RF was 99.8% leaked → 75.6% honest) |
| **match_id as column 0, not feature** | Enables grouping without contaminating the feature matrix |
| **OneHotEncoder(drop='first')** | Avoids dummy variable trap (multicollinearity) for linear models |
| **handle_unknown='ignore'** | Some cities only appear in test folds (group-CV) — encode as zeros |
| **LogReg solver='liblinear'** | Good for small datasets with OneHot-encoded features |
| **XGBoost max_depth=4** | Deliberately shallow to avoid overfitting noisy ball-level data |
| **LR as default predictor model** | Best honest accuracy (77.3%), smoothest probability curves |
| **2nd innings only** | Win probability prediction is meaningful only during the chase |
| **Ball-level granularity** | Enables real-time over-by-over progression analysis |
| **pickle serialization** | Simple, preserves sklearn Pipeline objects for deployment |
