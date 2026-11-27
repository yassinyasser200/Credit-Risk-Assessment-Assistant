"""ML classification tool: predict_credit_risk(features: dict) -> dict

Owner: Member 1 (ML & Dataset). Integrated by Member 4 (Agent).

Contract (common tool envelope from the master plan, section 9):

    success  -> {"tool": "predict_credit_risk", "status": "ok",
                 "data": {"risk_label", "risk_label_description", "predicted_probability",
                          "probability_of", "decision_threshold", "model_version",
                          "schema_version", "used_features"},
                 "error": None, "warnings": [...]}

    failure  -> {"tool": "predict_credit_risk", "status": "error", "data": None,
                 "error": {"code", "message", "details"}, "warnings": [...]}

Error codes (machine-readable, stable):
    invalid_input       features is not a dict
    missing_features    at least one required feature is absent or None -> details.missing_fields
    invalid_features    unknown fields, wrong types, values outside allowed sets/ranges
    model_unavailable   artifact or schema file missing / unreadable
    schema_mismatch     artifact does not match feature_schema.json (hash or columns)
    prediction_failed   estimator raised unexpectedly (no stack trace is exposed)

Rules this module guarantees:
    * The estimator is never called unless validation passed completely.
    * Missing values are never imputed or guessed.
    * Protected / excluded attributes (age, sex, foreign_worker, ...) are rejected, not ignored,
      so a caller can never believe they influenced the estimate.
    * Output is an educational estimate, never an approval/denial.
"""

from __future__ import annotations

import json
import logging
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

TOOL_NAME = "predict_credit_risk"

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARTIFACT_PATH = REPO_ROOT / "ml" / "artifacts" / "credit_risk_pipeline.joblib"
DEFAULT_SCHEMA_PATH = REPO_ROOT / "ml" / "feature_schema.json"

# Environment overrides (used by tests and by deployments that move the artifact).
ENV_ARTIFACT = "CREDIT_RISK_ARTIFACT_PATH"
ENV_SCHEMA = "CREDIT_RISK_SCHEMA_PATH"

LIMITATION_NOTE = (
    "Educational estimate from a small model trained on 1,000 historical German loan records "
    "(1970s, Deutsche Mark). It is not a lending decision, not a credit score, and not financial advice."
)

logger = logging.getLogger(__name__)


class ModelUnavailable(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


# ---------------------------------------------------------------------------
# Envelope helpers
# ---------------------------------------------------------------------------
def _ok(data: dict, warnings: list[str]) -> dict:
    return {"tool": TOOL_NAME, "status": "ok", "data": data, "error": None, "warnings": warnings}


def _error(code: str, message: str, details: dict | None = None, warnings: list[str] | None = None) -> dict:
    return {
        "tool": TOOL_NAME,
        "status": "error",
        "data": None,
        "error": {"code": code, "message": message, "details": details or {}},
        "warnings": warnings or [],
    }


# ---------------------------------------------------------------------------
# Loading (cached per path so a long-running agent loads the model once)
# ---------------------------------------------------------------------------
def _paths() -> tuple[Path, Path]:
    return (
        Path(os.environ.get(ENV_ARTIFACT, DEFAULT_ARTIFACT_PATH)),
        Path(os.environ.get(ENV_SCHEMA, DEFAULT_SCHEMA_PATH)),
    )


@lru_cache(maxsize=4)
def _load_schema_cached(schema_path: str) -> dict:
    p = Path(schema_path)
    if not p.exists():
        raise ModelUnavailable("model_unavailable", "Feature schema file is missing.")
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelUnavailable("model_unavailable", "Feature schema file is unreadable.") from exc


@lru_cache(maxsize=4)
def _load_pipeline_cached(artifact_path: str, schema_path: str):
    import hashlib

    import joblib  # imported lazily so schema-only callers don't pay for it

    schema = _load_schema_cached(schema_path)
    p = Path(artifact_path)
    if not p.exists():
        raise ModelUnavailable(
            "model_unavailable",
            "Model artifact is missing. Run `python -m ml.train` from the repository root.",
        )
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    expected = schema.get("artifact", {}).get("sha256")
    if expected and digest != expected:
        # Also protects against loading a tampered pickle: joblib files execute code on load.
        raise ModelUnavailable(
            "schema_mismatch",
            "Model artifact does not match feature_schema.json (hash differs). Re-run training.",
        )
    import sklearn

    trained_with = schema.get("artifact", {}).get("sklearn_version")
    if trained_with and trained_with != sklearn.__version__:
        # A scikit-learn pickle is only reliable on the exact version that saved it.
        raise ModelUnavailable(
            "model_unavailable",
            f"Model was saved with scikit-learn {trained_with} but {sklearn.__version__} is installed. "
            "Run `pip install -r requirements-ml.txt` (or retrain with `python -m ml.train`).",
        )
    try:
        pipeline = joblib.load(p)
    except Exception as exc:  # corrupted / incompatible pickle
        raise ModelUnavailable("model_unavailable", "Model artifact could not be loaded.") from exc

    cols = list(getattr(pipeline, "feature_names_in_", []))
    if cols != list(schema["feature_order"]):
        raise ModelUnavailable(
            "schema_mismatch", "Pipeline input columns do not match the feature schema."
        )
    return pipeline, schema


def clear_cache() -> None:
    """Drop cached schema/pipeline (e.g. after retraining in the same process)."""
    for fn in (_load_schema_cached, _load_pipeline_cached):
        getattr(fn, "cache_clear", lambda: None)()


def get_feature_schema() -> dict:
    """Return the input schema (for the agent's clarification step and the UI form)."""
    _, schema_path = _paths()
    return _load_schema_cached(str(schema_path))


def required_features() -> list[str]:
    return list(get_feature_schema()["required_features"])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _validate(features: Any, schema: dict) -> tuple[dict | None, dict | None, list[str]]:
    """Return (clean_row, error_envelope, warnings). Exactly one of row / error is set."""
    warnings: list[str] = []
    if not isinstance(features, dict):
        return None, _error("invalid_input", "features must be a JSON object mapping field names to values.",
                            {"received_type": type(features).__name__}), warnings

    spec = schema["features"]
    required = schema["required_features"]
    excluded = schema.get("excluded_features", {})

    missing = [f for f in required if features.get(f) is None]
    unknown = sorted(k for k in features if k not in spec)
    excluded_supplied = [k for k in unknown if k in excluded]
    other_unknown = [k for k in unknown if k not in excluded]

    type_errors: dict[str, str] = {}
    value_errors: dict[str, str] = {}
    clean: dict[str, Any] = {}

    for name in required:
        if name in missing:
            continue
        value = features[name]
        fspec = spec[name]
        if fspec["type"] == "integer":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                type_errors[name] = f"expected an integer, got {type(value).__name__}"
                continue
            if isinstance(value, float):
                if not math.isfinite(value) or not value.is_integer():
                    type_errors[name] = "expected a whole number"
                    continue
                value = int(value)
            if not (fspec["min"] <= value <= fspec["max"]):
                value_errors[name] = f"{value} is outside the valid range [{fspec['min']}, {fspec['max']}]"
                continue
            if not (fspec["training_min"] <= value <= fspec["training_max"]):
                warnings.append(
                    f"{name}={value} is outside the training range "
                    f"[{fspec['training_min']}, {fspec['training_max']}]; the estimate is an extrapolation."
                )
            clean[name] = value
        else:  # categorical
            if not isinstance(value, str):
                type_errors[name] = f"expected a string category, got {type(value).__name__}"
                continue
            norm = value.strip().lower()
            if norm not in fspec["allowed_values"]:
                value_errors[name] = (
                    f"'{value}' is not an allowed value; choose one of {sorted(fspec['allowed_values'])}"
                )
                continue
            clean[name] = norm

    if missing:
        return None, _error(
            "missing_features",
            "Some required financial details are missing. Ask the user for them; do not guess.",
            {
                "missing_fields": missing,
                # Report other problems too, so the agent can fix everything in one clarification turn.
                "unknown_fields": other_unknown,
                "excluded_fields_supplied": excluded_supplied,
                "type_errors": type_errors,
                "value_errors": value_errors,
            },
        ), warnings

    if unknown or type_errors or value_errors:
        details: dict[str, Any] = {}
        if other_unknown:
            details["unknown_fields"] = other_unknown
        if excluded_supplied:
            details["excluded_fields_supplied"] = {k: excluded[k] for k in excluded_supplied}
        if type_errors:
            details["type_errors"] = type_errors
        if value_errors:
            details["value_errors"] = value_errors
        details["allowed_fields"] = required
        return None, _error("invalid_features", "Some supplied features are invalid.", details), warnings

    return clean, None, warnings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def predict_credit_risk(features: dict) -> dict:
    """Estimate a credit-risk category for one validated financial profile.

    Never raises for bad input or a missing model; always returns the tool envelope.
    """
    artifact_path, schema_path = _paths()
    try:
        schema = _load_schema_cached(str(schema_path))
    except ModelUnavailable as exc:
        return _error(exc.code, exc.message)

    row, err, warnings = _validate(features, schema)
    if err is not None:
        return err  # the estimator is NOT loaded or called on invalid input

    try:
        pipeline, schema = _load_pipeline_cached(str(artifact_path), str(schema_path))
    except ModelUnavailable as exc:
        return _error(exc.code, exc.message, warnings=warnings)

    import pandas as pd

    frame = pd.DataFrame([[row[c] for c in schema["feature_order"]]], columns=schema["feature_order"])
    try:
        classes = list(pipeline.classes_)
        proba = pipeline.predict_proba(frame)[0]
        p_high = float(proba[classes.index(1)])
    except Exception:  # never leak a stack trace to the LLM/UI
        logger.exception("predict_credit_risk: estimator failure")
        return _error("prediction_failed", "The model failed to produce an estimate.", warnings=warnings)

    threshold = float(schema["target"]["decision_threshold_on_p_high_risk"])
    label = "high_risk" if p_high >= threshold else "low_risk"

    warnings.append(LIMITATION_NOTE)
    warnings.append(
        "Excluded from the model by design: " + ", ".join(schema.get("excluded_features", {}))
        + ". Remaining financial variables may still act as proxies for them."
    )
    return _ok(
        {
            "risk_label": label,
            "risk_label_description": schema["target"]["labels"][label],
            "predicted_probability": round(p_high, 4),
            "probability_of": "high_risk",
            "decision_threshold": threshold,
            "model_version": schema["model_version"],
            "schema_version": schema["schema_version"],
            "used_features": list(schema["feature_order"]),
        },
        warnings,
    )


def compare_hypothetical(base_features: dict, changes: dict) -> dict:
    """Stretch goal: re-run the SAME saved model on a baseline and a modified profile.

    Returns {"tool": "compare_hypothetical", "status", "data": {baseline, hypothetical,
    changed_fields, probability_change}, "error", "warnings"}.
    """
    name = "compare_hypothetical"
    if not isinstance(base_features, dict) or not isinstance(changes, dict) or not changes:
        out = _error("invalid_input", "base_features and a non-empty changes object are required.")
        out["tool"] = name
        return out

    base = predict_credit_risk(base_features)
    if base["status"] != "ok":
        base["tool"] = name
        base["error"]["message"] = "Baseline profile invalid: " + base["error"]["message"]
        return base

    hypo_features = {**base_features, **changes}
    hypo = predict_credit_risk(hypo_features)
    if hypo["status"] != "ok":
        hypo["tool"] = name
        hypo["error"]["message"] = "Hypothetical profile invalid: " + hypo["error"]["message"]
        return hypo

    changed = {k: {"from": base_features.get(k), "to": v} for k, v in changes.items() if base_features.get(k) != v}
    delta = round(hypo["data"]["predicted_probability"] - base["data"]["predicted_probability"], 4)
    return {
        "tool": name,
        "status": "ok",
        "data": {
            "baseline": base["data"],
            "hypothetical": hypo["data"],
            "changed_fields": changed,
            "probability_change": delta,
        },
        "error": None,
        "warnings": [
            "HYPOTHETICAL scenario: shows how this model's estimate responds to the changed inputs. "
            "It is not causal and does not guarantee any real lending outcome.",
            *dict.fromkeys(base["warnings"] + hypo["warnings"]),
        ],
    }


# ---------------------------------------------------------------------------
# LangGraph / LangChain registration (for Member 4)
# ---------------------------------------------------------------------------
def tool_description() -> str:
    schema = get_feature_schema()
    lines = [
        "Estimate an EDUCATIONAL credit-risk category (low_risk / high_risk) for one financial profile. "
        "Call ONLY when the user has supplied every required field below; if anything is missing, ask the "
        "user instead of guessing. Never pass age, sex, marital status, foreign-worker status, dependents, "
        "telephone or residence length. Returns status ok|error; on missing_features, "
        "error.details.missing_fields lists what to ask for.",
        "Required fields:",
    ]
    for name in schema["required_features"]:
        f = schema["features"][name]
        if f["type"] == "integer":
            lines.append(f"- {name} (integer {f['min']}-{f['max']}, {f['unit']}): {f['description']}")
        else:
            lines.append(f"- {name} (one of: {', '.join(f['allowed_values'])})")
    return "\n".join(lines)


def build_langchain_tool():
    """Return a LangChain StructuredTool wrapping predict_credit_risk (requires langchain-core).

    Usage in the agent (M4):
        from tools.ml_tool import build_langchain_tool
        ml_tool = build_langchain_tool()
        llm_with_tools = llm.bind_tools([ml_tool, rag_tool, graph_tool])
        # or: ToolNode([ml_tool, rag_tool, graph_tool])
    """
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field

    class PredictCreditRiskInput(BaseModel):
        features: dict[str, Any] = Field(
            description="Mapping of every required feature name to its value, exactly as listed in the tool description."
        )

    def _run(features: dict[str, Any]) -> dict:
        return predict_credit_risk(features)

    return StructuredTool.from_function(
        func=_run,
        name=TOOL_NAME,
        description=tool_description(),
        args_schema=PredictCreditRiskInput,
    )


if __name__ == "__main__":  # quick manual check: python -m tools.ml_tool
    example_path = REPO_ROOT / "ml" / "examples" / "sample_profiles.json"
    examples = json.loads(example_path.read_text())
    for ex in examples:
        print(f"\n# {ex['id']}: {ex['description']}")
        print(json.dumps(predict_credit_risk(ex["features"]), indent=2))
