# Dataset note: ML classification tool

*Owner: Member 1 (ML & Dataset). For: Member 6 (Quality & scenarios) and the README.*

## Source and access

| Item | Value |
|---|---|
| Dataset | Statlog (German Credit Data) |
| Creator | Prof. Dr. Hans Hofmann, Universität Hamburg (1994) |
| Repository | UCI Machine Learning Repository, doi:10.24432/C5NC77 |
| License | CC BY 4.0: redistribution allowed with attribution |
| File in repo | `ml/data/german.data.csv`, the original `german.data` as comma-separated values with no header row |
| Checksum | SHA-256 `ec12a88b9fc14d74ba646ea0410cf7ff4533bec2eb61652f8ad76796bbfec017`, verified every time the data is loaded |
| Documentation | `ml/data/german.names`, the original attribute description |
| Re-download | `ml.preprocess.download_dataset()` pulls a GitHub mirror of the UCI file and rejects it if the checksum differs |

## Target

`credit_class` (attribute 21) records the original lender's own rating of each applicant: `1` means good and `2` means bad.
The model target is **`high_risk = 1` when `credit_class == 2`**.

The API returns these labels:
- `high_risk`: the profile resembles applicants this lender rated *bad*.
- `low_risk`: the profile resembles applicants this lender rated *good*.

`predicted_probability` is the model's estimated P(high_risk). It is **not calibrated** to any real population.
The label threshold is fixed at 0.5 and was not tuned on the test set.

## Size, missing values, class balance

- 1,000 rows and 20 raw attributes (7 numerical, 13 categorical).
- **No missing values.** The "unknown / none" categories in the source are real category values, not blanks.
- Class balance: 700 low_risk vs 300 high_risk (30% high_risk). Logistic regression uses `class_weight="balanced"`.
- Split: stratified 80/20 with `random_state=42`, giving 800 train and 200 test rows. Both splits are 30% high_risk.

## Feature selection: 14 of the 20 attributes are model inputs

**Inputs (all required):**
- Numeric: `duration_months`, `credit_amount`, `installment_rate_pct`, `existing_credits`
- Categorical: `checking_account_status`, `credit_history`, `purpose`, `savings_account`, `employment_duration`, `other_debtors`, `property`, `other_installment_plans`, `housing`, `job`

**Excluded from training and inference:**

| Attribute | Reason |
|---|---|
| `personal_status_sex` | Protected attribute (sex) combined with marital status |
| `age_years` | Protected / demographic attribute |
| `foreign_worker` | Protected attribute (national origin / migration status) |
| `num_dependents` | Household composition, a family-status proxy, not financial |
| `telephone` | Contact-information indicator, not financial |
| `present_residence_years` | Residential tenure; not financial, and a proxy for age |

The dataset has no direct identifiers such as names or IDs.
If a caller sends one of the excluded fields, the tool **rejects** the request with `invalid_features` rather than ignoring the field.
That way nobody can believe an excluded field influenced the estimate.

## Known limitations (say these in the demo)

1. **Proxy bias remains.** Dropping protected columns does not make the model fair. For example, `job` includes "non-resident", `housing` correlates with age (mean age 30 for renters vs 44 for "for free"), and `employment_duration` correlates with age. No fairness metric was computed against the excluded attributes.
2. **The data is old and comes from one place:** a single south-German bank in the 1970s, with amounts in Deutsche Mark. The values cannot be meaningfully converted to present-day EGP or EUR.
3. **Selection bias.** The data contains only applicants who were *granted* credit. Rejected applicants never appear, so the model never learned from them.
4. **A counter-intuitive pattern.** `credit_history = critical_or_other_credits` is associated with *lower* risk in this data (17% bad vs 32% for `existing_paid_duly`), which is a well-known quirk of this dataset. Present it as a pattern in the data, not as lending logic.
5. **The data is small.** With 1,000 rows, the 5-fold CV ROC AUC has a standard deviation of about 0.06. The hold-out AUC (0.80) is above the CV mean (0.76), so treat CV as the more honest figure.
6. **Validity range.** Numeric inputs outside the *training-split* range (duration 4–60 months, amount 276–15,945 DM) are accepted with an extrapolation warning. Values outside hard bounds are rejected (duration 1–120, amount 1–100,000, the coded fields 1–4).
7. **A few category values are rare** (e.g. `purpose = retraining` has 9 rows), so estimates for them are less reliable. `purpose = vacation` is documented in the source but never occurs in the data, so the tool does not accept it.

## Metrics (from `ml/model_metrics.json`)

| | Logistic regression (selected) | Random forest (comparison) | Majority-class dummy |
|---|---|---|---|
| 5-fold CV ROC AUC (train) | 0.765 ± 0.059 | 0.791 ± 0.045 | 0.500 |
| 5-fold CV balanced accuracy | 0.683 ± 0.055 | 0.722 ± 0.050 | 0.500 |

The selected model on the hold-out test set (200 rows, evaluated once):
- ROC AUC 0.802, accuracy 0.755, balanced accuracy 0.773.
- For the high_risk class: precision 0.563, recall 0.817.
- Confusion matrix (rows actual, columns predicted, order [low, high]): [[102, 38], [11, 49]].

Logistic regression was chosen **in advance** as the transparent baseline, even though the random forest scores slightly higher.
The course does not grade accuracy, and a linear model with inspectable coefficients is easier to defend and explain.

**Global feature importance** (permutation, drop in test ROC AUC): `checking_account_status` matters by far the most, followed by `purpose`, `credit_history`, `duration_months` and `credit_amount`.
This measures what *the model relies on*. It does **not** show what *causes* credit risk.
