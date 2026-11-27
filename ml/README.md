# ML classification tool (Member 1)

This module turns a validated financial profile into an **educational** credit-risk estimate (`low_risk` / `high_risk`).
It runs on its own, with no agent, LLM, ChromaDB or Neo4j needed.

## Run it

Open a terminal in the `credit-risk-agent` folder and run **one command at a time**. Works on Python 3.10–3.13.

```bash
python -m pip install -r requirements-ml.txt   # once
python -m pytest -v                            # expect: 53 passed
python -m tools.ml_tool                        # runs the 9 examples in ml/examples/sample_profiles.json
python -m ml.train                             # optional: rebuilds the artifacts (~6 s)
```

scikit-learn must be exactly **1.9.1**, because the saved model only loads on the version that created it.
If the tool returns `model_unavailable` with a "saved with scikit-learn ..." message, re-run the install line.
If you can't install 1.9.1, run `python -m ml.train` to retrain on your own version.

A fresh Python process can predict straight from the committed artifact, without retraining:

```python
from tools.ml_tool import predict_credit_risk
predict_credit_risk({...14 fields...})
```

## Files

| File | What it is |
|---|---|
| `ml/preprocess.py` | Column names, code→value maps, feature lists, excluded fields, and the `ColumnTransformer` |
| `ml/train.py` | Split, CV model comparison, final fit, metrics, permutation importance, and artifact saving |
| `ml/artifacts/credit_risk_pipeline.joblib` | The full preprocessing + LogisticRegression pipeline |
| `ml/feature_schema.json` | The input contract (types, allowed values, ranges, feature order), plus the artifact's SHA-256 |
| `ml/model_metrics.json` | Dataset audit, CV and hold-out metrics, importance, coefficients, environment |
| `ml/DATASET_NOTE.md` | Source, license, target, exclusions and limitations (for M6) |
| `ml/examples/sample_profiles.json` | 9 documented synthetic examples (4 valid, 5 invalid) with expected results |
| `tools/ml_tool.py` | `predict_credit_risk`, `compare_hypothetical`, `get_feature_schema`, `build_langchain_tool` |
| `tests/test_ml_tool.py` | 53 tests |

## Contract

`predict_credit_risk(features: dict) -> dict` **never raises**. It always returns the team envelope.

**Input:** all 14 fields are required. Categorical values are case-insensitive. Numbers must be integers (`12.0` is accepted, `"12"` is not).

```json
{
  "checking_account_status": "0_to_200_dm",
  "duration_months": 24,
  "credit_history": "existing_paid_duly",
  "purpose": "furniture_equipment",
  "credit_amount": 3500,
  "savings_account": "100_to_500_dm",
  "employment_duration": "1_to_4_years",
  "installment_rate_pct": 3,
  "other_debtors": "none",
  "property": "car_or_other",
  "other_installment_plans": "none",
  "housing": "own",
  "existing_credits": 1,
  "job": "skilled_employee"
}
```

| Field | Values |
|---|---|
| `duration_months` | integer 1–120 (training range 4–60) |
| `credit_amount` | integer 1–100000 DM (training range 276–15945) |
| `installment_rate_pct` | 1–4 (coded band of the installment as % of disposable income) |
| `existing_credits` | 1–4 (credits already held at this bank) |
| `checking_account_status` | `negative_balance`, `0_to_200_dm`, `200_dm_or_more`, `no_checking_account` |
| `credit_history` | `no_credits_or_all_paid`, `all_paid_at_this_bank`, `existing_paid_duly`, `past_payment_delay`, `critical_or_other_credits` |
| `purpose` | `car_new`, `car_used`, `furniture_equipment`, `radio_television`, `domestic_appliances`, `repairs`, `education`, `retraining`, `business`, `other` |
| `savings_account` | `lt_100_dm`, `100_to_500_dm`, `500_to_1000_dm`, `1000_dm_or_more`, `unknown_or_none` |
| `employment_duration` | `unemployed`, `lt_1_year`, `1_to_4_years`, `4_to_7_years`, `7_years_or_more` |
| `other_debtors` | `none`, `co_applicant`, `guarantor` |
| `property` | `real_estate`, `savings_agreement_or_life_insurance`, `car_or_other`, `unknown_or_none` |
| `other_installment_plans` | `bank`, `stores`, `none` |
| `housing` | `rent`, `own`, `for_free` |
| `job` | `unemployed_or_unskilled_nonresident`, `unskilled_resident`, `skilled_employee`, `management_self_employed_highly_qualified` |

`get_feature_schema()` returns this same information as a dict, so the UI form and the agent prompt can be generated from it.

**Success:**

```json
{
  "tool": "predict_credit_risk",
  "status": "ok",
  "data": {
    "risk_label": "high_risk",
    "risk_label_description": "Pattern resembles applicants the original lender rated 'bad' credit risk.",
    "predicted_probability": 0.5841,
    "probability_of": "high_risk",
    "decision_threshold": 0.5,
    "model_version": "credit-risk-logreg-1.0.0",
    "schema_version": "1.0.0",
    "used_features": ["duration_months", "credit_amount", "..."]
  },
  "error": null,
  "warnings": ["Educational estimate ... not a lending decision ...", "Excluded from the model by design: ..."]
}
```

**Failure** (example: only 3 of the 14 fields were sent):

```json
{
  "tool": "predict_credit_risk",
  "status": "error",
  "data": null,
  "error": {
    "code": "missing_features",
    "message": "Some required financial details are missing. Ask the user for them; do not guess.",
    "details": {"missing_fields": ["installment_rate_pct", "existing_credits", "checking_account_status", "..."],
                "unknown_fields": [], "excluded_fields_supplied": [], "type_errors": {}, "value_errors": {}}
  },
  "warnings": []
}
```

| `error.code` | When | What the agent should do |
|---|---|---|
| `missing_features` | A required field is absent or `null` | Copy `details.missing_fields` into `state.missing_fields` and take the `ask_user` action |
| `invalid_features` | Unknown or excluded field, wrong type, value outside the allowed set or range | Tell the user which value is wrong (see `details`) and ask again |
| `invalid_input` | `features` is not an object | Malformed LLM action: log it and re-decide |
| `model_unavailable` | Artifact or schema missing or unreadable, or the wrong scikit-learn version is installed | Report that the ML tool is unavailable; continue with other tools if useful |
| `schema_mismatch` | The artifact hash or columns don't match `feature_schema.json` | Same as above; the fix is `python -m ml.train` |
| `prediction_failed` | The estimator raised an exception (details are logged, never returned) | Same as above |

When validation fails, the estimator is **never loaded or called**. This is tested.

## Handoff to M4 (registering it as a LangGraph tool)

```python
from tools.ml_tool import build_langchain_tool        # needs langchain-core
ml_tool = build_langchain_tool()                       # StructuredTool named "predict_credit_risk"

llm_with_tools = llm.bind_tools([ml_tool, rag_tool, graph_tool])
# in tool_node:   result = ml_tool.invoke({"features": features_dict})   -> the envelope above
```

The tool's description already lists every required field and its allowed values, and it tells the LLM to ask rather than guess.
`tests/test_ml_tool.py::test_langchain_tool_wrapper` covers one successful and one failing call through this interface.

**Stretch goal for M6: hypothetical comparison**

```python
from tools.ml_tool import compare_hypothetical
compare_hypothetical(base_profile, {"duration_months": 12})
# -> data.baseline, data.hypothetical, data.changed_fields, data.probability_change
```

Both predictions come from the same saved model. The result's first warning labels the output HYPOTHETICAL and states that it is not causal.

## Discussion prep: short answers

- **What is the target?** The original lender's good/bad rating of each applicant. `high_risk` means the lender rated them bad. It is a historical label, not a default outcome we observed.
- **How was leakage prevented?** A stratified 80/20 split was made before any fitting. The scaler and encoder are fitted inside the Pipeline, so only on training folds. The model was chosen with CV on the train split, and the test split was used once. No feature is derived from the target.
- **Why logistic regression?** It was the baseline chosen in advance: linear, with inspectable coefficients and native probabilities. One-hot encoding with fixed categories plus scaling is the standard, auditable preprocessing. The random forest scores a little higher in CV, but accuracy isn't graded and LR is easier to defend.
- **Missing field?** The function returns `missing_features` with the exact list of missing fields. Nothing is imputed, and the model isn't called.
- **Why is this an estimate, not a decision?** The data is 1,000 rows from a 1970s German bank, the probabilities are uncalibrated, proxy bias remains, there are no rejected applicants in the data, and no regulatory validation was done. It is an educational prototype.
