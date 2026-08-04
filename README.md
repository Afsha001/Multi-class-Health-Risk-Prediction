<h1 align="center">Predicting Student Health Risk</h1>

<p align="center">
A machine learning solution for the Kaggle Playground Series S6E7 competition — a 3-class classification problem predicting student health risk from lifestyle and physiological indicators.
</p>

---

### **Problem Statement**

Universities and health programs increasingly rely on wearable and self-reported lifestyle data to flag students who may be at health risk before conditions become serious. This project simulates that scenario: given a student's sleep patterns, activity levels, stress, diet, and physiological metrics, predict whether they fall into one of three health categories — **at-risk**, **unhealthy**, or **fit**.

The real-world value of a model like this lies in early intervention — a campus wellness program could use such a classifier to proactively reach out to students showing risk signals, rather than waiting for a health issue to surface on its own. The core modeling challenge mirrors real health-screening data: severe class imbalance, since most people in any population are not in an extreme health category, and the minority classes (the ones most worth catching) are the hardest to detect.

---

### **Dataset**

- **~690,000** training rows, **~296,000** test rows
- **Target:** `health_condition` — 3 classes, severely imbalanced (~86% at-risk, ~8% unhealthy, ~6% fit)
- **Features:** sleep duration, heart rate, BMI, calorie expenditure, step count, exercise duration, water intake, stress level, sleep quality, physical activity level, diet type, smoking/alcohol habits, gender
- **Metric:** Balanced accuracy (macro-averaged recall) — chosen specifically because it penalizes ignoring minority classes, unlike plain accuracy

---

### **Approach**

The project followed a disciplined, phase-based methodology, prioritizing honest validation over chasing leaderboard scores at every step.

**1. Exploratory Data Analysis**
Identified severe class imbalance, confirmed `stress_level` and `sleep_duration` as the dominant predictive signals, flagged `heart_rate` and `water_intake` as near-noise, and verified missingness in key columns was random (not informative) rather than assuming so.

**2. Baseline Model**
Median imputation for numeric features, ordinal encoding for naturally ordered categoricals (stress, sleep quality, activity level), one-hot encoding for unordered categoricals. Trained LightGBM with `class_weight='balanced'` under 5-fold stratified cross-validation, validated with out-of-fold (OOF) predictions, confusion matrices, and gain-based feature importance (not default split-count importance, which was shown to be misleading on this dataset).

**3. Feature Engineering**
Tested interaction and ratio features (stress×sleep, activity×calorie ratios, etc.) individually against the baseline. All were within noise — none kept. This confirmed the tree-based model was already capturing these interactions implicitly.

**4. Threshold & Weight Tuning**
Tested per-class probability weighting to directly optimize balanced accuracy, validated with proper nested cross-validation (not just in-sample tuning, which would have overstated the gain). Result: no real improvement beyond noise.

**5. Hyperparameter Tuning (Optuna)**
Searched `num_leaves`, `max_depth`, `min_child_samples`, `subsample`, `colsample_bytree`, and regularization terms. This was the only step that produced a real, reproducible gain, confirmed across two independent fold seeds.

**6. Model Comparison (Phase 4)**
Trained CatBoost (native categorical handling) and XGBoost under the same CV protocol. Both underperformed the tuned LightGBM. Ensembling (simple averaging, weighted blends) was tested and also gave no improvement beyond noise — the alternative models weren't diverse enough in their errors to add value.

**7. Targeted Diagnostics**
Investigated LightGBM's specific error patterns — a small subset of "contradictory" rows (e.g., unhealthy students with low stress) where the dominant feature pointed the wrong way. Confirmed via subgroup analysis that these cases were too rare (~0.2% of data) to be worth engineering around, and represent likely irreducible noise rather than a fixable gap.

**8. Robustness Checks**
Seed bagging (8 models, different random seeds, averaged) and model-based imputation of missing `stress_level` were both tested as final polish steps. Neither improved on the tuned single-seed model — consistent with a model that had already converged to a stable, near-ceiling solution for this feature set.

---

### **Key Finding**

Across five independent experiments spanning feature engineering, threshold tuning, model diversity, ensembling, and seed bagging, **only hyperparameter tuning on LightGBM produced a real, reproducible improvement.** Everything else was honestly tested and honestly ruled out. This is a deliberately conservative, well-validated result rather than one inflated by overfitting to cross-validation or the public leaderboard.

---

### **Tech Stack**

| Category | Tools |
|---|---|
| Language | Python |
| Data handling | pandas, NumPy |
| Modeling | LightGBM, XGBoost, CatBoost, scikit-learn (Random Forest) |
| Hyperparameter tuning | Optuna |
| Validation | Stratified K-Fold cross-validation, out-of-fold prediction analysis |
| Visualization | Matplotlib, Seaborn |
| Environment | Kaggle Notebooks |

---

### **Results**

| Approach | OOF Balanced Accuracy | Public LB | Outcome |
|-----------------------------------------------|--------|-------|---------------------------|
| Baseline LightGBM (`class_weight='balanced'`) | 0.9494 | 0.9500 | Established working floor |
| + Hand-crafted interaction/ratio features     | 0.9493–0.9495    | No improvement — not kept |
| + Per-class threshold/weight tuning | 0.9494  | —                | No improvement — not kept |
| + Optuna hyperparameter tuning| **0.9496* | — | **Real, reproducible gain — kept** |
| CatBoost (native categorical) | 0.9490 | — | Underperformed LightGBM |
| XGBoost | 0.9491 | — | Underperformed LightGBM |
| Equal-weight 3-model ensemble | 0.9496 | — | No improvement over tuned LightGBM alone |
| Seed bagging (8 seeds) | 0.9496 | — | No improvement |
| Model-based `stress_level` imputation | — | — | Tested, no meaningful gain |
| **Final model: Tuned LightGBM** | **0.9496** | **~0.9500** | **Submitted** |

*All balanced accuracy scores rounded to 4 decimal places. CV and public leaderboard scores tracked closely throughout, confirming no overfitting to cross-validation.*

---

### **What This Project Demonstrates**

- End-to-end ML pipeline design: EDA → preprocessing → baseline → iterative improvement → validated final model
- Rigorous, honest experimentation — reporting negative results rather than only showcasing what worked
- Correct handling of severe class imbalance under a metric (balanced accuracy) that punishes naive approaches
- Awareness of common pitfalls: misleading default feature importance, in-sample tuning bias, small-subgroup overfitting risk
- Practical Kaggle workflow: checkpointing expensive computations, structuring a lean inference-only submission notebook separate from the exploratory notebook, to meet competition runtime constraints
