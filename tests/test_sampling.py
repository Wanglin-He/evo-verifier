from __future__ import annotations

from conftest import write_annotations

from evo_verifier.__main__ import _stratified
from evo_verifier.items import BY_ID
from evo_verifier.labels import Label, load_annotations

B7 = BY_ID["B7"].column


def cases(tmp_path, failures: int, passes: int):
    rows = [{B7: "不满足"} for _ in range(failures)] + [{B7: "满足"} for _ in range(passes)]
    return load_annotations(write_annotations(tmp_path / "a.csv", rows))


def test_every_failure_survives_the_sample(tmp_path):
    """Positives are the scarce half; recall has to be measured on all of them."""
    sample = _stratified(cases(tmp_path, failures=10, passes=200), {"B7"}, 60)
    assert sum(1 for case in sample if case.labels["B7"] is Label.FAIL) == 10
    assert len(sample) == 60


def test_the_rest_are_sampled_at_a_known_rate(tmp_path):
    sample = _stratified(cases(tmp_path, failures=10, passes=200), {"B7"}, 60)
    kept = [case for case in sample if case.labels["B7"] is Label.PASS]
    assert len(kept) == 50, "50 of 200 passes: a rate precision can be corrected for"
    assert len({case.record_id for case in kept}) == 50, "no duplicates"


def test_the_sample_is_the_same_every_time(tmp_path):
    built = cases(tmp_path, failures=5, passes=100)
    first = [case.record_id for case in _stratified(built, {"B7"}, 40)]
    second = [case.record_id for case in _stratified(built, {"B7"}, 40)]
    assert first == second


def test_asking_for_more_than_exists_returns_everything(tmp_path):
    built = cases(tmp_path, failures=3, passes=7)
    assert len(_stratified(built, {"B7"}, 500)) == 10


def test_a_sample_smaller_than_the_failures_still_keeps_them_all(tmp_path):
    """Never drop a positive to hit a size. The size gives way instead."""
    sample = _stratified(cases(tmp_path, failures=12, passes=50), {"B7"}, 5)
    assert len(sample) == 12
