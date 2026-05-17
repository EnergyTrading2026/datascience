"""Tests for ``optimization.config_validation``.

Covers:

- Errors are collected (not raised) by ``validate_plant_payload``.
- A maximally-broken payload returns *all* errors at once — the contract
  Tasks 3-5's operator API will rely on.
- Each warning rule fires when its threshold is crossed, doesn't fire when
  it isn't, and never marks the config ``not ok``.
- ``validate_plant_config`` on an already-constructed ``PlantConfig`` returns
  ``errors == ()`` (by construction the dataclass __post_init__ raised before
  the instance could be returned).
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from optimization.config import (
    CONFIG_SCHEMA_VERSION,
    BoilerParams,
    CHPParams,
    HeatPumpParams,
    PlantConfig,
    StorageParams,
)
from optimization.config_validation import (
    DT_H_REQUIRED,
    MIN_UP_DOWN_WARN_STEPS,
    STORAGE_USABLE_RANGE_WARN_RATIO,
    Codes,
    ConfigValidationError,
    Issue,
    ValidationResult,
    validate_plant_config,
    validate_plant_payload,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _legacy_payload() -> dict:
    """Fresh copy of the legacy default's JSON dict — safe to mutate per test."""
    return PlantConfig.legacy_default().to_dict()


def _has_code(issues: tuple[Issue, ...] | list[Issue], code: str) -> bool:
    return any(i.code == code for i in issues)


def _issues_with_code(
    issues: tuple[Issue, ...] | list[Issue], code: str
) -> list[Issue]:
    return [i for i in issues if i.code == code]


# --------------------------------------------------------------------------- #
# Result shape
# --------------------------------------------------------------------------- #


def test_legacy_default_payload_validates_clean():
    """The shipped default must not produce any errors or warnings.

    If a warning ever fires on the legacy default, either the default is
    wrong or the warning's threshold is.
    """
    result = validate_plant_payload(_legacy_payload(), CONFIG_SCHEMA_VERSION)
    assert result.ok, f"legacy default produced errors: {result.errors}"
    assert result.warnings == (), (
        f"legacy default produced warnings (review thresholds): {result.warnings}"
    )


def test_validation_result_ok_distinguishes_errors_from_warnings():
    """``ok`` should be True with warnings only, False with any error."""
    only_warnings = ValidationResult(warnings=(
        Issue(severity="warning", code="X", path="x", message="m"),
    ))
    assert only_warnings.ok is True
    with_error = ValidationResult(errors=(
        Issue(severity="error", code="Y", path="y", message="n"),
    ))
    assert with_error.ok is False


def test_issue_includes_path_for_offending_field():
    """Errors/warnings must include a path so a UI can highlight the field."""
    payload = _legacy_payload()
    payload["boilers"][0]["q_min_mw_th"] = 999.0  # min > max
    result = validate_plant_payload(payload, CONFIG_SCHEMA_VERSION)
    errs = _issues_with_code(result.errors, Codes.MIN_EXCEEDS_MAX)
    assert len(errs) == 1
    assert errs[0].path == "boilers[0].q_min_mw_th"


# --------------------------------------------------------------------------- #
# Collect-all-errors mode (the Task 5 contract)
# --------------------------------------------------------------------------- #


def test_maximally_broken_payload_reports_every_error_at_once():
    """Submit a payload with errors in independent places and verify
    ``validate_plant_payload`` returns *all* of them, not just the first."""
    payload = _legacy_payload()
    payload["dt_h"] = 0.5                          # DT_H_NOT_QUARTERHOUR
    payload["gas_price_eur_mwh_hs"] = -1.0         # NEGATIVE
    payload["heat_pumps"][0]["p_el_min_mw"] = 99.0  # MIN_EXCEEDS_MAX on HP
    payload["boilers"][0]["eff"] = 2.0             # EFF_OUT_OF_RANGE
    # Make CHP id collide with HP id to trigger DUPLICATE_ASSET_ID.
    payload["chps"][0]["id"] = "hp"

    result = validate_plant_payload(payload, CONFIG_SCHEMA_VERSION)

    assert not result.ok
    codes = {i.code for i in result.errors}
    assert {
        Codes.DT_H_NOT_QUARTERHOUR,
        Codes.NEGATIVE,
        Codes.MIN_EXCEEDS_MAX,
        Codes.EFF_OUT_OF_RANGE,
        Codes.DUPLICATE_ASSET_ID,
    } <= codes, f"missing codes; got: {codes}"


def test_payload_with_only_warnings_is_ok():
    """A config with warnings but no errors must validate as ``ok``."""
    payload = _legacy_payload()
    # Crank CHP eff_th up so eff_el + eff_th > 1 (legal individually).
    payload["chps"][0]["eff_el"] = 0.6
    payload["chps"][0]["eff_th"] = 0.6
    result = validate_plant_payload(payload, CONFIG_SCHEMA_VERSION)
    assert result.ok
    assert _has_code(result.warnings, Codes.CHP_EFF_SUM_OVER_ONE)


def test_validate_plant_config_on_legacy_default_has_no_errors():
    """``validate_plant_config`` on a constructed config never returns
    errors (the dataclass __post_init__ already enforced them)."""
    cfg = PlantConfig.legacy_default()
    result = validate_plant_config(cfg)
    assert result.errors == ()


# --------------------------------------------------------------------------- #
# dt_h promoted to hard error
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad_dt", [0.5, 1.0, 0.1, 0.0])
def test_dt_h_other_than_quarterhour_is_hard_error(bad_dt):
    """The MILP grid is hardcoded to 15-min; any other dt_h is now an error
    (previously a silent miscompute risk)."""
    payload = _legacy_payload()
    payload["dt_h"] = bad_dt
    result = validate_plant_payload(payload, CONFIG_SCHEMA_VERSION)
    assert not result.ok
    assert _has_code(result.errors, Codes.DT_H_NOT_QUARTERHOUR)


def test_dt_h_quarterhour_passes():
    payload = _legacy_payload()
    payload["dt_h"] = DT_H_REQUIRED
    result = validate_plant_payload(payload, CONFIG_SCHEMA_VERSION)
    assert result.ok


def test_constructing_plantconfig_with_bad_dt_h_raises():
    """Direct construction must also reject non-15-min dt_h."""
    base = PlantConfig.legacy_default()
    with pytest.raises(ValueError, match="0.25"):
        replace(base, dt_h=0.5)


# --------------------------------------------------------------------------- #
# Warning rules
# --------------------------------------------------------------------------- #


def test_chp_eff_sum_over_one_warns():
    """CHP with eff_el + eff_th > 1 is thermodynamically suspect — each is
    individually <= 1 so the hard checks don't catch it."""
    payload = _legacy_payload()
    payload["chps"][0]["eff_el"] = 0.55
    payload["chps"][0]["eff_th"] = 0.55  # sum = 1.10
    result = validate_plant_payload(payload, CONFIG_SCHEMA_VERSION)
    assert result.ok
    matching = _issues_with_code(result.warnings, Codes.CHP_EFF_SUM_OVER_ONE)
    assert len(matching) == 1
    assert matching[0].path == "chps[0].eff_th"


def test_chp_eff_sum_at_or_below_one_does_not_warn():
    payload = _legacy_payload()
    # legacy_default already has 0.40 + 0.48 = 0.88, no warning expected.
    result = validate_plant_payload(payload, CONFIG_SCHEMA_VERSION)
    assert not _has_code(result.warnings, Codes.CHP_EFF_SUM_OVER_ONE)


def test_storage_usable_range_tiny_warns():
    """When floor is within ~10% of capacity the storage is functionally
    useless — almost certainly a typo."""
    payload = _legacy_payload()
    payload["storages"][0]["capacity_mwh_th"] = 100.0
    payload["storages"][0]["floor_mwh_th"] = 95.0  # usable = 5 < 10% * 100
    payload["storages"][0]["soc_init_mwh_th"] = 98.0  # stay legal
    result = validate_plant_payload(payload, CONFIG_SCHEMA_VERSION)
    assert result.ok
    assert _has_code(result.warnings, Codes.STORAGE_USABLE_RANGE_TINY)


def test_storage_usable_range_just_above_threshold_does_not_warn():
    payload = _legacy_payload()
    payload["storages"][0]["capacity_mwh_th"] = 100.0
    # usable = 11 > 10% of 100, no warning.
    payload["storages"][0]["floor_mwh_th"] = 89.0
    payload["storages"][0]["soc_init_mwh_th"] = 95.0
    result = validate_plant_payload(payload, CONFIG_SCHEMA_VERSION)
    assert not _has_code(result.warnings, Codes.STORAGE_USABLE_RANGE_TINY)


def test_storage_loss_exceeds_charge_warns():
    """Storage that loses more energy per step than it can charge in one step
    drains itself faster than it fills."""
    payload = _legacy_payload()
    payload["storages"][0]["charge_max_mw_th"] = 1.0   # 1 MW * 0.25 h = 0.25 MWh / step
    payload["storages"][0]["loss_mwh_per_step"] = 1.0  # 1 MWh / step >> 0.25
    result = validate_plant_payload(payload, CONFIG_SCHEMA_VERSION)
    assert result.ok
    assert _has_code(result.warnings, Codes.STORAGE_LOSS_EXCEEDS_CHARGE)


def test_storage_loss_below_charge_does_not_warn():
    payload = _legacy_payload()
    # legacy_default has charge_max=15, loss=0.000125 → no warning expected.
    result = validate_plant_payload(payload, CONFIG_SCHEMA_VERSION)
    assert not _has_code(result.warnings, Codes.STORAGE_LOSS_EXCEEDS_CHARGE)


def test_storage_discharge_disabled_warns():
    payload = _legacy_payload()
    payload["storages"][0]["discharge_max_mw_th"] = 0.0
    result = validate_plant_payload(payload, CONFIG_SCHEMA_VERSION)
    assert result.ok
    assert _has_code(result.warnings, Codes.STORAGE_DISCHARGE_DISABLED)


def test_storage_charge_disabled_warns():
    payload = _legacy_payload()
    payload["storages"][0]["charge_max_mw_th"] = 0.0
    result = validate_plant_payload(payload, CONFIG_SCHEMA_VERSION)
    assert result.ok
    assert _has_code(result.warnings, Codes.STORAGE_CHARGE_DISABLED)


def test_min_up_down_very_long_warns():
    """min_up_steps or min_down_steps > 96 (24h) dominates any plausible
    MPC horizon (default target 35h, 140 steps)."""
    payload = _legacy_payload()
    payload["boilers"][0]["min_up_steps"] = MIN_UP_DOWN_WARN_STEPS + 1
    result = validate_plant_payload(payload, CONFIG_SCHEMA_VERSION)
    assert result.ok
    assert _has_code(result.warnings, Codes.MIN_UP_OR_DOWN_VERY_LONG)


def test_no_storages_warns():
    """A plant with no storages has zero inter-cycle thermal flexibility —
    legal but worth flagging."""
    payload = _legacy_payload()
    payload["storages"] = []
    result = validate_plant_payload(payload, CONFIG_SCHEMA_VERSION)
    assert result.ok
    assert _has_code(result.warnings, Codes.NO_STORAGES)


# --------------------------------------------------------------------------- #
# Storage usable-range threshold edge case
# --------------------------------------------------------------------------- #


def test_storage_usable_range_threshold_value():
    """Document the exact threshold via the public constant so consumers can
    rely on it (and a future change is a deliberate API change)."""
    assert STORAGE_USABLE_RANGE_WARN_RATIO == 0.10


# --------------------------------------------------------------------------- #
# Payload-level errors (preserve existing from_dict behavior)
# --------------------------------------------------------------------------- #


def test_missing_required_top_level_scalar_is_error():
    payload = _legacy_payload()
    del payload["dt_h"]
    result = validate_plant_payload(payload, CONFIG_SCHEMA_VERSION)
    assert not result.ok
    assert any("missing required field" in i.message for i in result.errors)


def test_non_numeric_scalar_is_error():
    payload = _legacy_payload()
    payload["gas_price_eur_mwh_hs"] = "not a number"
    result = validate_plant_payload(payload, CONFIG_SCHEMA_VERSION)
    assert not result.ok
    assert _has_code(result.errors, Codes.NON_NUMERIC)


def test_payload_not_a_dict_is_error():
    result = validate_plant_payload([1, 2, 3], CONFIG_SCHEMA_VERSION)
    assert not result.ok
    assert _has_code(result.errors, Codes.PAYLOAD_NOT_DICT)


# --------------------------------------------------------------------------- #
# Direct dataclass construction still raises (preserves existing tests)
# --------------------------------------------------------------------------- #


def test_direct_heat_pump_min_exceeds_max_raises():
    with pytest.raises(ValueError, match="p_el_min_mw"):
        HeatPumpParams(id="hp", p_el_min_mw=10.0, p_el_max_mw=5.0, cop=3.0)


def test_direct_boiler_bad_eff_raises():
    with pytest.raises(ValueError, match="eff"):
        BoilerParams(
            id="b", q_min_mw_th=1.0, q_max_mw_th=5.0, eff=2.0,
            min_up_steps=4, min_down_steps=4,
        )


def test_direct_chp_eff_sum_warning_does_not_raise():
    """Warnings must never raise from the dataclass — only errors do.

    This is the contract that lets the operator API surface warnings without
    refusing to construct the config.
    """
    c = CHPParams(
        id="c", p_el_min_mw=1.0, p_el_max_mw=5.0,
        eff_el=0.6, eff_th=0.6,  # sum > 1 → warning only
        min_up_steps=4, min_down_steps=4, startup_cost_eur=100.0,
    )
    assert c.eff_el + c.eff_th == pytest.approx(1.2)


def test_direct_storage_bad_floor_raises():
    with pytest.raises(ValueError, match="floor_mwh_th"):
        StorageParams(
            id="s", capacity_mwh_th=100.0, floor_mwh_th=150.0,
            charge_max_mw_th=10.0, discharge_max_mw_th=10.0,
            loss_mwh_per_step=0.0, soc_init_mwh_th=50.0,
        )


# --------------------------------------------------------------------------- #
# Schema-version handling
# --------------------------------------------------------------------------- #


def test_missing_schema_version_is_error():
    payload = _legacy_payload()
    del payload["schema_version"]
    result = validate_plant_payload(payload, CONFIG_SCHEMA_VERSION)
    assert not result.ok
    assert _has_code(result.errors, Codes.SCHEMA_VERSION_MISSING)


def test_schema_version_mismatch_is_error():
    payload = _legacy_payload()
    payload["schema_version"] = CONFIG_SCHEMA_VERSION + 99
    result = validate_plant_payload(payload, CONFIG_SCHEMA_VERSION)
    assert not result.ok
    assert _has_code(result.errors, Codes.SCHEMA_VERSION_MISMATCH)


def test_bool_schema_version_is_rejected():
    """``True == 1`` in Python — a boolean must NOT masquerade as schema 1."""
    payload = _legacy_payload()
    payload["schema_version"] = True
    result = validate_plant_payload(payload, CONFIG_SCHEMA_VERSION)
    assert not result.ok
    assert _has_code(result.errors, Codes.SCHEMA_VERSION_MISMATCH)


# --------------------------------------------------------------------------- #
# Payload shape errors
# --------------------------------------------------------------------------- #


def test_unknown_top_level_keys_is_error():
    payload = _legacy_payload()
    payload["whats_this"] = 42
    result = validate_plant_payload(payload, CONFIG_SCHEMA_VERSION)
    assert not result.ok
    assert _has_code(result.errors, Codes.UNKNOWN_TOP_LEVEL_KEYS)


def test_missing_required_field_has_dedicated_code():
    """Missing top-level scalars should use MISSING_REQUIRED_FIELD, not
    ASSET_FIELDS_WRONG (which promises asset-level scope)."""
    payload = _legacy_payload()
    del payload["dt_h"]
    result = validate_plant_payload(payload, CONFIG_SCHEMA_VERSION)
    assert not result.ok
    assert _has_code(result.errors, Codes.MISSING_REQUIRED_FIELD)
    assert not _has_code(result.errors, Codes.ASSET_FIELDS_WRONG)


def test_asset_entry_not_a_dict_is_error():
    payload = _legacy_payload()
    payload["boilers"] = ["not a dict"]
    result = validate_plant_payload(payload, CONFIG_SCHEMA_VERSION)
    assert not result.ok
    assert _has_code(result.errors, Codes.ASSET_NOT_A_DICT)


def test_asset_family_not_a_list_is_error():
    payload = _legacy_payload()
    payload["boilers"] = {"id": "b"}  # wrong container
    result = validate_plant_payload(payload, CONFIG_SCHEMA_VERSION)
    assert not result.ok
    assert _has_code(result.errors, Codes.ASSET_FIELDS_WRONG)


def test_numeric_string_is_rejected_as_non_numeric():
    """``"3.14"`` would silently coerce via float(); JSON typing should be
    correct so this is a typo signal we surface explicitly."""
    payload = _legacy_payload()
    payload["gas_price_eur_mwh_hs"] = "3.14"
    result = validate_plant_payload(payload, CONFIG_SCHEMA_VERSION)
    assert not result.ok
    assert _has_code(result.errors, Codes.NON_NUMERIC)


def test_asset_field_wrong_type_is_non_numeric():
    """A stringy number inside an asset must be reported as NON_NUMERIC,
    not crash on float()."""
    payload = _legacy_payload()
    payload["boilers"][0]["q_max_mw_th"] = "twelve"
    result = validate_plant_payload(payload, CONFIG_SCHEMA_VERSION)
    assert not result.ok
    matching = _issues_with_code(result.errors, Codes.NON_NUMERIC)
    assert len(matching) >= 1
    assert any(i.path == "boilers[0].q_max_mw_th" for i in matching)


def test_boiler_min_steps_float_is_non_numeric():
    """min_up_steps must be int, not float — ``4.0`` is fine in JSON but
    ``4.5`` should not silently round."""
    payload = _legacy_payload()
    payload["boilers"][0]["min_up_steps"] = 4.5
    result = validate_plant_payload(payload, CONFIG_SCHEMA_VERSION)
    assert not result.ok
    assert _has_code(result.errors, Codes.NON_NUMERIC)


# --------------------------------------------------------------------------- #
# Cross-asset rules
# --------------------------------------------------------------------------- #


def test_bad_asset_id_whitespace_is_error():
    payload = _legacy_payload()
    payload["boilers"][0]["id"] = "  padded  "
    result = validate_plant_payload(payload, CONFIG_SCHEMA_VERSION)
    assert not result.ok
    assert _has_code(result.errors, Codes.BAD_ASSET_ID)


def test_bad_asset_id_empty_string_is_error():
    payload = _legacy_payload()
    payload["boilers"][0]["id"] = ""
    result = validate_plant_payload(payload, CONFIG_SCHEMA_VERSION)
    assert not result.ok
    assert _has_code(result.errors, Codes.BAD_ASSET_ID)


def test_zero_heat_producers_is_error():
    payload = _legacy_payload()
    payload["heat_pumps"] = []
    payload["boilers"] = []
    payload["chps"] = []
    result = validate_plant_payload(payload, CONFIG_SCHEMA_VERSION)
    assert not result.ok
    assert _has_code(result.errors, Codes.ZERO_HEAT_PRODUCERS)


def test_duplicate_id_within_same_family_is_error():
    payload = _legacy_payload()
    payload["boilers"].append(dict(payload["boilers"][0]))  # same id "boiler"
    result = validate_plant_payload(payload, CONFIG_SCHEMA_VERSION)
    assert not result.ok
    assert _has_code(result.errors, Codes.DUPLICATE_ASSET_ID)


# --------------------------------------------------------------------------- #
# Storage collect-all + threshold boundary
# --------------------------------------------------------------------------- #


def test_storage_negativity_reported_alongside_bad_capacity():
    """Per the collect-all contract, a storage with capacity<=0 must still
    surface negativity errors on charge/discharge/loss."""
    payload = _legacy_payload()
    payload["storages"][0]["capacity_mwh_th"] = 0.0
    payload["storages"][0]["charge_max_mw_th"] = -1.0
    payload["storages"][0]["loss_mwh_per_step"] = -1.0
    result = validate_plant_payload(payload, CONFIG_SCHEMA_VERSION)
    assert not result.ok
    codes = {i.code for i in result.errors}
    assert Codes.NEGATIVE_OR_ZERO in codes  # capacity
    # Two distinct NEGATIVE issues: charge/discharge and loss.
    negatives = _issues_with_code(result.errors, Codes.NEGATIVE)
    assert len(negatives) >= 2


def test_storage_usable_range_at_exact_threshold_does_not_warn():
    """Threshold is < 10% of capacity. Exactly 10% does not trigger."""
    payload = _legacy_payload()
    payload["storages"][0]["capacity_mwh_th"] = 100.0
    payload["storages"][0]["floor_mwh_th"] = 90.0  # usable = 10 == 10% * 100
    payload["storages"][0]["soc_init_mwh_th"] = 95.0
    result = validate_plant_payload(payload, CONFIG_SCHEMA_VERSION)
    assert not _has_code(result.warnings, Codes.STORAGE_USABLE_RANGE_TINY)


def test_storage_usable_range_just_below_threshold_warns():
    """Just under 10% of capacity must trigger the warning."""
    payload = _legacy_payload()
    payload["storages"][0]["capacity_mwh_th"] = 100.0
    payload["storages"][0]["floor_mwh_th"] = 90.1  # usable = 9.9 < 10
    payload["storages"][0]["soc_init_mwh_th"] = 95.0
    result = validate_plant_payload(payload, CONFIG_SCHEMA_VERSION)
    assert result.ok
    assert _has_code(result.warnings, Codes.STORAGE_USABLE_RANGE_TINY)


# --------------------------------------------------------------------------- #
# min-vs-max split: negative-min is NEGATIVE, not MIN_EXCEEDS_MAX
# --------------------------------------------------------------------------- #


def test_negative_min_reported_as_negative_not_min_exceeds_max():
    """A negative min with a valid max is a NEGATIVE error, not MIN_EXCEEDS_MAX
    (the prior code conflated the two)."""
    payload = _legacy_payload()
    payload["heat_pumps"][0]["p_el_min_mw"] = -1.0
    result = validate_plant_payload(payload, CONFIG_SCHEMA_VERSION)
    assert not result.ok
    assert _has_code(result.errors, Codes.NEGATIVE)
    assert not any(
        i.code == Codes.MIN_EXCEEDS_MAX and i.path == "heat_pumps[0].p_el_min_mw"
        for i in result.errors
    )


# --------------------------------------------------------------------------- #
# validate_plant_config (constructed) surfaces warnings
# --------------------------------------------------------------------------- #


def test_validate_plant_config_surfaces_warnings_on_constructed_config():
    """The Task 3-5 path: load a config, then re-validate to get warnings.

    Build a PlantConfig with eff_el + eff_th > 1 (legal individually but
    suspect together) via from_dict and assert ``validate_plant_config``
    returns the warning.
    """
    payload = _legacy_payload()
    payload["chps"][0]["eff_el"] = 0.55
    payload["chps"][0]["eff_th"] = 0.55  # sum = 1.10
    cfg = PlantConfig.from_dict(payload)
    result = validate_plant_config(cfg)
    assert result.errors == ()
    assert _has_code(result.warnings, Codes.CHP_EFF_SUM_OVER_ONE)


# --------------------------------------------------------------------------- #
# Round-trip: to_dict -> validate -> from_dict on a non-legacy config
# --------------------------------------------------------------------------- #


def test_from_dict_raises_config_validation_error_carrying_full_result():
    """``from_dict`` must raise ``ConfigValidationError`` (still a
    ``ValueError`` for backwards compat) that carries the *full*
    ``ValidationResult`` — callers can't be forced to use the
    payload entrypoint to get collect-all behavior.
    """
    payload = _legacy_payload()
    payload["dt_h"] = 0.5                          # DT_H_NOT_QUARTERHOUR
    payload["gas_price_eur_mwh_hs"] = -1.0         # NEGATIVE
    payload["boilers"][0]["eff"] = 2.0             # EFF_OUT_OF_RANGE

    with pytest.raises(ConfigValidationError) as excinfo:
        PlantConfig.from_dict(payload)

    # Still a ValueError so legacy ``except ValueError`` keeps working.
    assert isinstance(excinfo.value, ValueError)
    codes = {i.code for i in excinfo.value.result.errors}
    assert {
        Codes.DT_H_NOT_QUARTERHOUR,
        Codes.NEGATIVE,
        Codes.EFF_OUT_OF_RANGE,
    } <= codes, f"missing codes; got {codes}"


def test_direct_plantconfig_construction_with_errors_raises_config_validation_error():
    """Direct ``PlantConfig(...)`` (not via ``from_dict``) must raise
    ``ConfigValidationError`` carrying the full ``ValidationResult`` — same
    contract as ``from_dict``. Without this, callers that use
    ``dataclasses.replace`` (Task 3's live-reload path) would get a plain
    ``ValueError`` with only the first message and lose the full diff.
    """
    base = PlantConfig.legacy_default()
    # dt_h=0.5 is one error; replace doesn't let us inject more without
    # constructing fresh asset tuples, but one error is enough to assert the
    # exception type and that the result carries the issue.
    with pytest.raises(ConfigValidationError) as excinfo:
        replace(base, dt_h=0.5)
    assert isinstance(excinfo.value, ValueError)  # backwards compat
    assert _has_code(excinfo.value.result.errors, Codes.DT_H_NOT_QUARTERHOUR)


def test_payload_min_up_steps_zero_is_min_up_down_too_small():
    """Stable code for min_up_steps/min_down_steps < 1 — Task 5's API will
    key off this code, not the message string."""
    payload = _legacy_payload()
    payload["boilers"][0]["min_up_steps"] = 0
    result = validate_plant_payload(payload, CONFIG_SCHEMA_VERSION)
    assert not result.ok
    assert _has_code(result.errors, Codes.MIN_UP_DOWN_TOO_SMALL)


def test_payload_floor_above_capacity_is_floor_out_of_range():
    payload = _legacy_payload()
    cap = payload["storages"][0]["capacity_mwh_th"]
    payload["storages"][0]["floor_mwh_th"] = cap + 1.0
    # keep soc_init in [0, capacity] so the FLOOR error isn't masked by
    # SOC_INIT bouncing first.
    payload["storages"][0]["soc_init_mwh_th"] = cap / 2
    result = validate_plant_payload(payload, CONFIG_SCHEMA_VERSION)
    assert not result.ok
    assert _has_code(result.errors, Codes.FLOOR_OUT_OF_RANGE)


def test_payload_soc_init_above_capacity_is_soc_init_out_of_range():
    payload = _legacy_payload()
    cap = payload["storages"][0]["capacity_mwh_th"]
    payload["storages"][0]["soc_init_mwh_th"] = cap + 1.0
    result = validate_plant_payload(payload, CONFIG_SCHEMA_VERSION)
    assert not result.ok
    assert _has_code(result.errors, Codes.SOC_INIT_OUT_OF_RANGE)


def test_null_asset_list_is_reported_not_silently_ignored():
    """``"heat_pumps": null`` (a JSON null) must be reported as a wrong-type
    error, not coerced to an empty list. Mirrors the unknown-key strictness:
    the previous ``payload.get(..., []) or []`` would have silently dropped
    a typo'd null and then complained about zero heat producers.
    """
    payload = _legacy_payload()
    payload["heat_pumps"] = None
    result = validate_plant_payload(payload, CONFIG_SCHEMA_VERSION)
    assert not result.ok
    matching = _issues_with_code(result.errors, Codes.ASSET_FIELDS_WRONG)
    assert any(i.path == "heat_pumps" for i in matching), result.errors


def test_non_list_asset_family_does_not_crash_cross_asset_check():
    """Cross-asset checks must run from in-loop counts, not from
    ``len(payload.get(...))``. If a family payload is something like
    ``5`` instead of a list, the family is errored *and* the count goes to
    zero — no ``TypeError: object of type 'int' has no len()`` ever leaks.
    """
    payload = _legacy_payload()
    payload["heat_pumps"] = 5  # not a list
    result = validate_plant_payload(payload, CONFIG_SCHEMA_VERSION)
    assert not result.ok
    assert _has_code(result.errors, Codes.ASSET_FIELDS_WRONG)
    # With heat_pumps unusable but boilers + chps still present, the plant
    # should NOT additionally report ZERO_HEAT_PRODUCERS.
    assert not _has_code(result.errors, Codes.ZERO_HEAT_PRODUCERS)


def test_to_dict_validate_from_dict_round_trip_non_legacy():
    """A multi-asset, non-default config must round-trip cleanly."""
    cfg = PlantConfig(
        dt_h=0.25,
        gas_price_eur_mwh_hs=40.0,
        co2_factor_t_per_mwh_hs=0.2,
        co2_price_eur_per_t=80.0,
        heat_pumps=(
            HeatPumpParams(id="hp-a", p_el_min_mw=0.5, p_el_max_mw=4.0, cop=3.2),
            HeatPumpParams(id="hp-b", p_el_min_mw=0.5, p_el_max_mw=4.0, cop=3.6),
        ),
        boilers=(
            BoilerParams(id="b-a", q_min_mw_th=1.0, q_max_mw_th=8.0,
                         eff=0.95, min_up_steps=4, min_down_steps=4),
        ),
        chps=(),
        storages=(
            StorageParams(id="s-a", capacity_mwh_th=150.0, floor_mwh_th=20.0,
                          charge_max_mw_th=10.0, discharge_max_mw_th=10.0,
                          loss_mwh_per_step=0.0001, soc_init_mwh_th=120.0),
        ),
    )
    payload = cfg.to_dict()
    validation = validate_plant_payload(payload, CONFIG_SCHEMA_VERSION)
    assert validation.ok, validation.errors
    cfg2 = PlantConfig.from_dict(payload)
    assert cfg2 == cfg


# --------------------------------------------------------------------------- #
# PlantConfig.same_asset_set — the gate used by the daemon to decide whether
# a live config reload is a parameter-only change (allowed) or an asset-set
# change (deferred to the live add/remove task).
# --------------------------------------------------------------------------- #


def test_same_asset_set_identical():
    cfg = PlantConfig.legacy_default()
    assert cfg.same_asset_set(cfg) is True


def test_same_asset_set_true_when_only_parameters_differ():
    """The whole point of the gate: param tweaks must pass."""
    base = PlantConfig.legacy_default()
    bumped = replace(
        base,
        gas_price_eur_mwh_hs=base.gas_price_eur_mwh_hs + 5.0,
        heat_pumps=(
            replace(base.heat_pumps[0], p_el_max_mw=base.heat_pumps[0].p_el_max_mw + 1.0),
        ),
        storages=(
            replace(base.storages[0], capacity_mwh_th=base.storages[0].capacity_mwh_th + 50.0),
        ),
    )
    assert base.same_asset_set(bumped) is True


def test_same_asset_set_false_when_unit_id_renamed():
    base = PlantConfig.legacy_default()
    renamed = replace(
        base,
        heat_pumps=(replace(base.heat_pumps[0], id="hp_renamed"),),
    )
    assert base.same_asset_set(renamed) is False


def test_same_asset_set_false_when_storage_id_renamed():
    base = PlantConfig.legacy_default()
    renamed = replace(
        base,
        storages=(replace(base.storages[0], id="storage_renamed"),),
    )
    assert base.same_asset_set(renamed) is False


def test_same_asset_set_false_when_asset_added():
    base = PlantConfig.legacy_default()
    grown = replace(
        base,
        heat_pumps=base.heat_pumps + (
            HeatPumpParams(id="hp_new", p_el_min_mw=1.0, p_el_max_mw=4.0, cop=3.5),
        ),
    )
    assert base.same_asset_set(grown) is False
    assert grown.same_asset_set(base) is False  # symmetric


def test_same_asset_set_false_when_id_moves_between_families():
    """Same string id used in a different family is not the same asset.

    Per-family state schemas differ (UnitState vs StorageState), so the daemon
    must treat 'storage' as a storage and refuse to silently rehome it as a
    boiler with the same id.
    """
    base = PlantConfig.legacy_default()
    # Rebuild with the boiler's id reused as a (new) CHP id, with the original
    # boiler dropped — same set of strings, different family membership.
    moved = replace(
        base,
        boilers=(),
        chps=base.chps + (
            CHPParams(
                id=base.boilers[0].id,
                p_el_min_mw=1.0, p_el_max_mw=4.0,
                eff_el=0.4, eff_th=0.4,
                min_up_steps=4, min_down_steps=4,
                startup_cost_eur=100.0,
            ),
        ),
    )
    assert base.same_asset_set(moved) is False


def test_same_asset_set_handles_empty_families():
    """A customer with no CHP should still compare cleanly."""
    cfg = PlantConfig(
        dt_h=0.25,
        gas_price_eur_mwh_hs=35.0,
        co2_factor_t_per_mwh_hs=0.201,
        co2_price_eur_per_t=60.0,
        heat_pumps=(HeatPumpParams(id="hp", p_el_min_mw=1.0, p_el_max_mw=8.0, cop=3.5),),
        boilers=(),
        chps=(),
        storages=(
            StorageParams(id="s", capacity_mwh_th=100.0, floor_mwh_th=0.0,
                          charge_max_mw_th=10.0, discharge_max_mw_th=10.0,
                          loss_mwh_per_step=0.0, soc_init_mwh_th=50.0),
        ),
    )
    same = replace(cfg, gas_price_eur_mwh_hs=40.0)
    assert cfg.same_asset_set(same) is True


# --------------------------------------------------------------------------- #
# disabled_asset_ids — operator-forced off-switch. The asset stays registered
# and keeps its state; the optimizer is constrained to zero output. Validator
# rejects unknown / duplicate / shape-broken entries before they ever reach
# the MILP and rejects an all-producers-off plant as infeasible by intent.
# --------------------------------------------------------------------------- #


def test_disabled_asset_ids_default_is_empty():
    """A config without disabled_asset_ids constructs cleanly and yields ()."""
    cfg = PlantConfig.legacy_default()
    assert cfg.disabled_asset_ids == ()
    assert cfg.is_enabled("hp") is True


def test_is_enabled_reflects_membership():
    cfg = replace(PlantConfig.legacy_default(), disabled_asset_ids=("hp",))
    assert cfg.is_enabled("hp") is False
    assert cfg.is_enabled("boiler") is True


def test_disabled_asset_ids_round_trips_via_to_dict_from_dict():
    """to_dict emits the list, from_dict restores it: required for the live
    reload path to ever see a disable change."""
    cfg = replace(PlantConfig.legacy_default(), disabled_asset_ids=("hp", "storage"))
    payload = cfg.to_dict()
    assert payload["disabled_asset_ids"] == ["hp", "storage"]
    restored = PlantConfig.from_dict(payload)
    assert restored.disabled_asset_ids == ("hp", "storage")
    assert restored == cfg


def test_disabled_asset_ids_input_list_is_coerced_to_tuple():
    """Frozen dataclass invariant — like the four asset family fields."""
    cfg = replace(PlantConfig.legacy_default(), disabled_asset_ids=["hp"])
    assert cfg.disabled_asset_ids == ("hp",)


def test_unknown_disabled_id_is_error():
    payload = _legacy_payload()
    payload["disabled_asset_ids"] = ["does-not-exist"]
    result = validate_plant_payload(payload, CONFIG_SCHEMA_VERSION)
    codes = {i.code for i in result.errors}
    assert Codes.UNKNOWN_DISABLED_ID in codes
    # And from_dict raises with the same code on .result
    with pytest.raises(ConfigValidationError) as ei:
        PlantConfig.from_dict(payload)
    assert any(i.code == Codes.UNKNOWN_DISABLED_ID for i in ei.value.result.errors)


def test_duplicate_disabled_id_is_error():
    payload = _legacy_payload()
    payload["disabled_asset_ids"] = ["hp", "hp"]
    result = validate_plant_payload(payload, CONFIG_SCHEMA_VERSION)
    codes = [i.code for i in result.errors]
    assert Codes.DUPLICATE_DISABLED_ID in codes


@pytest.mark.parametrize("bad", ["", "  ", " hp", "hp ", 0, None, True])
def test_bad_disabled_id_is_error(bad):
    payload = _legacy_payload()
    payload["disabled_asset_ids"] = [bad]
    result = validate_plant_payload(payload, CONFIG_SCHEMA_VERSION)
    assert any(i.code == Codes.BAD_DISABLED_ID for i in result.errors), result.errors


def test_disabled_asset_ids_not_a_list_is_shape_error():
    """Mistyping the field (e.g. ``"hp"`` instead of ``["hp"]``) should
    produce one clear shape error, not a cascade of per-character complaints."""
    payload = _legacy_payload()
    payload["disabled_asset_ids"] = "hp"
    result = validate_plant_payload(payload, CONFIG_SCHEMA_VERSION)
    shape_errors = [i for i in result.errors
                    if i.code == Codes.ASSET_FIELDS_WRONG and i.path == "disabled_asset_ids"]
    assert len(shape_errors) == 1
    # Per-item disable checks must be suppressed once the shape is wrong —
    # otherwise the operator gets BAD_DISABLED_ID for every character.
    assert not any(i.code == Codes.BAD_DISABLED_ID for i in result.errors)


def test_all_heat_producers_disabled_is_error():
    """Plant has registered producers but every one is in the disabled list:
    demand cannot be met, refuse the config."""
    payload = _legacy_payload()
    payload["disabled_asset_ids"] = ["hp", "boiler", "chp"]
    result = validate_plant_payload(payload, CONFIG_SCHEMA_VERSION)
    assert any(i.code == Codes.ALL_HEAT_PRODUCERS_DISABLED for i in result.errors)


def test_all_heat_producers_disabled_not_fired_when_zero_producers():
    """Empty plant fires ZERO_HEAT_PRODUCERS, not the disabled variant —
    the conditions are disjoint and the operator gets the relevant message."""
    payload = _legacy_payload()
    payload["heat_pumps"] = []
    payload["boilers"] = []
    payload["chps"] = []
    payload["disabled_asset_ids"] = []
    result = validate_plant_payload(payload, CONFIG_SCHEMA_VERSION)
    codes = {i.code for i in result.errors}
    assert Codes.ZERO_HEAT_PRODUCERS in codes
    assert Codes.ALL_HEAT_PRODUCERS_DISABLED not in codes


def test_storage_only_disabled_is_allowed():
    """Disabling just a storage is legal: the producers still meet demand."""
    payload = _legacy_payload()
    payload["disabled_asset_ids"] = ["storage"]
    result = validate_plant_payload(payload, CONFIG_SCHEMA_VERSION)
    assert result.ok, result.errors


def test_some_producers_disabled_is_allowed():
    """Disabling a subset of producers is fine — operator can shed capacity."""
    payload = _legacy_payload()
    payload["disabled_asset_ids"] = ["hp", "boiler"]  # CHP remains
    result = validate_plant_payload(payload, CONFIG_SCHEMA_VERSION)
    assert result.ok, result.errors


def test_same_asset_set_unaffected_by_disable_toggle():
    """Disable is a parameter-level change, not an asset-set change — the
    daemon's live-reload gate must let a disable edit through."""
    base = PlantConfig.legacy_default()
    toggled = replace(base, disabled_asset_ids=("hp",))
    assert base.same_asset_set(toggled) is True
    assert toggled.same_asset_set(base) is True
