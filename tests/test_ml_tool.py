"""Unit tests for the ML classification tool (Member 1).

Run from the repository root:
    python -m pytest tests/test_ml_tool.py -v
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import joblib
import pytest

from tools import ml_tool
from tools.ml_tool import compare_hypothetical, get_feature_schema, predict_credit_risk

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = json.loads((ROOT / "ml" / "examples" / "sample_profiles.json").read_text())
VALID = next(e for e in EXAMPLES if e["id"] == "valid_borderline")["features"]
PROTECTED = {"age_years", "personal_status_sex", "foreign_worker"}


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    monkeypatch.delenv(ml_tool.ENV_ARTIFACT, raising=False)
    monkeypatch.delenv(ml_tool.ENV_SCHEMA, raising=False)
    ml_tool.clear_cache()
    yield
    ml_tool.clear_cache()


def profile(**overrides):
    return {**VALID, **overrides}


def assert_envelope(result, status):
    assert set(result) == {"tool", "status", "data", "error", "warnings"}
    assert result["tool"] == "predict_credit_risk"
    assert result["status"] == status
    assert isinstance(result["warnings"], list)
    if status == "ok":
        assert result["error"] is None and isinstance(result["data"], dict)
    else:
        assert result["data"] is None
        assert set(result["error"]) == {"code", "message", "details"}
        assert "Traceback" not in json.dumps(result)


# --------------------------------------------------------------------------- happy path
def test_valid_profile_returns_documented_output():
    r = predict_credit_risk(VALID)
    assert_envelope(r, "ok")
    d = r["data"]
    assert d["risk_label"] in {"low_risk", "high_risk"}
    assert 0.0 <= d["predicted_probability"] <= 1.0
    assert d["probability_of"] == "high_risk"
    assert d["model_version"] == get_feature_schema()["model_version"]
    assert d["used_features"] == get_feature_schema()["feature_order"]
    assert (d["risk_label"] == "high_risk") == (d["predicted_probability"] >= d["decision_threshold"])
    assert any("not a lending decision" in w for w in r["warnings"])


@pytest.mark.parametrize("example", EXAMPLES, ids=[e["id"] for e in EXAMPLES])
def test_documented_examples(example):
    r = predict_credit_risk(example["features"])
    exp = example["expected"]
    assert r["status"] == exp["status"]
    if exp["status"] == "ok":
        assert r["data"]["risk_label"] == exp["risk_label"]
        assert math.isclose(r["data"]["predicted_probability"], exp["predicted_probability"], abs_tol=1e-4)
    else:
        assert r["error"]["code"] == exp["error_code"]


def test_prediction_is_deterministic():
    a, b = predict_credit_risk(VALID), predict_credit_risk(dict(VALID))
    assert a == b


def test_categorical_values_are_normalised():
    r = predict_credit_risk(profile(housing="  OWN "))
    assert r["status"] == "ok"
    assert r["data"]["predicted_probability"] == predict_credit_risk(VALID)["data"]["predicted_probability"]


def test_whole_number_float_is_accepted():
    assert predict_credit_risk(profile(duration_months=24.0)) == predict_credit_risk(VALID)


def test_extrapolation_warning_outside_training_range():
    r = predict_credit_risk(profile(duration_months=90))
    assert r["status"] == "ok"
    assert any("outside the training range" in w for w in r["warnings"])


# --------------------------------------------------------------------------- missing features
def test_missing_feature_returns_structured_error_and_never_calls_estimator(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("estimator must not be loaded for invalid input")

    monkeypatch.setattr(ml_tool, "_load_pipeline_cached", boom)
    partial = {k: v for k, v in VALID.items() if k != "savings_account"}
    r = predict_credit_risk(partial)
    assert_envelope(r, "error")
    assert r["error"]["code"] == "missing_features"
    assert r["error"]["details"]["missing_fields"] == ["savings_account"]


def test_none_value_counts_as_missing():
    r = predict_credit_risk(profile(credit_amount=None))
    assert r["error"]["code"] == "missing_features"
    assert r["error"]["details"]["missing_fields"] == ["credit_amount"]


def test_empty_profile_lists_every_required_field():
    r = predict_credit_risk({})
    assert r["error"]["code"] == "missing_features"
    assert r["error"]["details"]["missing_fields"] == get_feature_schema()["required_features"]


def test_missing_and_invalid_reported_together():
    partial = {k: v for k, v in VALID.items() if k != "job"}
    partial["credit_amount"] = "lots"
    r = predict_credit_risk(partial)
    assert r["error"]["code"] == "missing_features"
    assert "credit_amount" in r["error"]["details"]["type_errors"]


# --------------------------------------------------------------------------- wrong types / values
@pytest.mark.parametrize(
    "field,value",
    [
        ("credit_amount", "3500"),       # numeric as string
        ("duration_months", True),       # bool is not an integer here
        ("duration_months", 12.5),       # non-whole number
        ("credit_amount", float("nan")),
        ("credit_amount", float("inf")),
        ("housing", 1),                  # number for a categorical
        ("purpose", ["car_new"]),
    ],
)
def test_wrong_types_are_rejected(field, value):
    r = predict_credit_risk(profile(**{field: value}))
    assert_envelope(r, "error")
    assert r["error"]["code"] == "invalid_features"
    assert field in r["error"]["details"]["type_errors"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("duration_months", 0),
        ("duration_months", 121),
        ("credit_amount", -100),
        ("credit_amount", 1_000_000),
        ("installment_rate_pct", 7),
        ("existing_credits", 0),
        ("purpose", "crypto"),
        ("purpose", "vacation"),  # documented in german.names but never occurs in the data
    ],
)
def test_out_of_range_or_unknown_values_are_rejected(field, value):
    r = predict_credit_risk(profile(**{field: value}))
    assert r["error"]["code"] == "invalid_features"
    assert field in r["error"]["details"]["value_errors"]


def test_unknown_field_is_rejected():
    r = predict_credit_risk(profile(favourite_colour="blue"))
    assert r["error"]["code"] == "invalid_features"
    assert r["error"]["details"]["unknown_fields"] == ["favourite_colour"]


@pytest.mark.parametrize("field", sorted(PROTECTED))
def test_protected_attributes_are_rejected_not_silently_used(field):
    r = predict_credit_risk(profile(**{field: "x"}))
    assert r["error"]["code"] == "invalid_features"
    assert field in r["error"]["details"]["excluded_fields_supplied"]


@pytest.mark.parametrize("bad", [None, "a profile", 42, [VALID]])
def test_non_dict_input(bad):
    r = predict_credit_risk(bad)
    assert_envelope(r, "error")
    assert r["error"]["code"] == "invalid_input"


# --------------------------------------------------------------------------- artifact problems
def test_missing_artifact_returns_model_unavailable(monkeypatch, tmp_path):
    monkeypatch.setenv(ml_tool.ENV_ARTIFACT, str(tmp_path / "nope.joblib"))
    r = predict_credit_risk(VALID)
    assert_envelope(r, "error")
    assert r["error"]["code"] == "model_unavailable"


def test_missing_schema_returns_model_unavailable(monkeypatch, tmp_path):
    monkeypatch.setenv(ml_tool.ENV_SCHEMA, str(tmp_path / "nope.json"))
    r = predict_credit_risk(VALID)
    assert r["error"]["code"] == "model_unavailable"


def test_tampered_artifact_is_refused(monkeypatch, tmp_path):
    bad = tmp_path / "pipeline.joblib"
    bad.write_bytes((ROOT / "ml/artifacts/credit_risk_pipeline.joblib").read_bytes() + b"\x00")
    monkeypatch.setenv(ml_tool.ENV_ARTIFACT, str(bad))
    r = predict_credit_risk(VALID)
    assert r["error"]["code"] == "schema_mismatch"


def test_corrupted_artifact_with_matching_schema(monkeypatch, tmp_path):
    import hashlib

    bad = tmp_path / "pipeline.joblib"
    bad.write_bytes(b"not a pickle")
    schema = get_feature_schema() | {}
    schema["artifact"] = {**schema["artifact"], "sha256": hashlib.sha256(b"not a pickle").hexdigest()}
    sp = tmp_path / "schema.json"
    sp.write_text(json.dumps(schema))
    monkeypatch.setenv(ml_tool.ENV_ARTIFACT, str(bad))
    monkeypatch.setenv(ml_tool.ENV_SCHEMA, str(sp))
    r = predict_credit_risk(VALID)
    assert r["error"]["code"] == "model_unavailable"


def test_wrong_sklearn_version_gives_clear_message(monkeypatch):
    import sklearn

    monkeypatch.setattr(sklearn, "__version__", "0.0.1")
    r = predict_credit_risk(VALID)
    assert r["error"]["code"] == "model_unavailable"
    assert "requirements-ml.txt" in r["error"]["message"]


def test_estimator_failure_is_contained(monkeypatch):
    class Broken:
        classes_ = [0, 1]

        def predict_proba(self, X):
            raise RuntimeError("internal detail that must not leak")

    monkeypatch.setattr(ml_tool, "_load_pipeline_cached", lambda a, s: (Broken(), get_feature_schema()))
    r = predict_credit_risk(VALID)
    assert r["error"]["code"] == "prediction_failed"
    assert "internal detail" not in json.dumps(r)


# --------------------------------------------------------------------------- schema <-> artifact consistency
def test_schema_matches_saved_pipeline_exactly():
    schema = get_feature_schema()
    pipe = joblib.load(ROOT / schema["artifact"]["path"])
    assert list(pipe.feature_names_in_) == schema["feature_order"] == schema["required_features"]
    enc = pipe.named_steps["preprocess"].named_transformers_["cat"]
    for name, cats in zip(enc.feature_names_in_, enc.categories_):
        assert sorted(cats.tolist()) == sorted(schema["features"][name]["allowed_values"])


def test_no_id_or_protected_features_are_model_inputs():
    schema = get_feature_schema()
    inputs = set(schema["feature_order"])
    assert inputs.isdisjoint(PROTECTED)
    assert inputs.isdisjoint(schema["excluded_features"])
    assert not any("id" == f or f.endswith("_id") for f in inputs)


def test_fresh_process_predicts_without_retraining():
    code = (
        "import json, sys; from tools.ml_tool import predict_credit_risk;"
        f"r = predict_credit_risk({VALID!r});"
        "assert 'ml.train' not in sys.modules, 'training code was imported';"
        "print(json.dumps(r))"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout)["status"] == "ok"


# --------------------------------------------------------------------------- stretch: hypothetical comparison
def test_hypothetical_comparison_uses_same_model():
    r = compare_hypothetical(VALID, {"duration_months": 12, "checking_account_status": "no_checking_account"})
    assert r["status"] == "ok"
    d = r["data"]
    assert d["baseline"]["model_version"] == d["hypothetical"]["model_version"]
    assert d["baseline"] == predict_credit_risk(VALID)["data"]
    assert set(d["changed_fields"]) == {"duration_months", "checking_account_status"}
    assert math.isclose(
        d["probability_change"],
        d["hypothetical"]["predicted_probability"] - d["baseline"]["predicted_probability"],
        abs_tol=1e-4,
    )
    assert "HYPOTHETICAL" in r["warnings"][0]


def test_hypothetical_with_invalid_change_is_controlled():
    r = compare_hypothetical(VALID, {"installment_rate_pct": 9})
    assert r["status"] == "error" and r["error"]["code"] == "invalid_features"
    assert compare_hypothetical(VALID, {})["error"]["code"] == "invalid_input"


# --------------------------------------------------------------------------- LangGraph registration
def test_langchain_tool_wrapper():
    pytest.importorskip("langchain_core")
    tool = ml_tool.build_langchain_tool()
    assert tool.name == "predict_credit_risk"
    for f in get_feature_schema()["required_features"]:
        assert f in tool.description
    ok = tool.invoke({"features": VALID})
    assert ok["status"] == "ok"
    err = tool.invoke({"features": {"duration_months": 12}})
    assert err["status"] == "error" and err["error"]["code"] == "missing_features"
