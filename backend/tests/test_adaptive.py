import zlib

import numpy as np
import pytest

from app.adaptive import (
    AdaptiveConfig, DiagnosticQuery, InMemorySearcher, Mode, Router, VehicleProfile, encode,
    extract_dtcs, load_dtc_table, run_adaptive,
)
from app.adaptive.context import metadata_matches, sensor_tokens
from app.adaptive.controller import rrf_fuse
from app.adaptive.router import count_symptoms, escalate
from app.adaptive.sufficiency import coverage, margin, rel_fused, rel_sparse
from app.adaptive.types import Hit
from app.adaptive.llm import Cause, Diagnosis, citation_precision, safety_notice

DIM = 64


def fake_embed(text: str) -> np.ndarray:
    """Deterministic hashed bag-of-words embedding (no model needed)."""
    v = np.zeros(DIM, dtype="float32")
    for w in text.lower().split():
        v[zlib.crc32(w.strip(".,?!").encode()) % DIM] += 1.0
    return v


DOCS = [
    ("Tire pressure monitor warning: check tire inflation pressure on the door pillar label.", 10),
    ("Jump starting: connect the positive terminal of the good battery to the starting aid terminal.", 20),
    ("Engine oil level: check the dipstick with the engine off and the car on level ground.", 30),
    ("Brake warning light: if it stays on, stop safely and check the brake fluid reservoir level.", 40),
    ("Fuel cap: turn the fuel cap counterclockwise to open the filler flap and refuel.", 50),
    ("P0301 cylinder 1 misfire detected: inspect spark plug and ignition coil for cylinder 1.", 60),
]


@pytest.fixture
def searcher():
    texts, pages = zip(*DOCS)
    return InMemorySearcher(
        list(texts), list(pages), embed_query=fake_embed,
        embed_documents=lambda ts: [fake_embed(t) for t in ts],
    )


@pytest.fixture
def dtc():
    return load_dtc_table()


# ---------------------------------------------------------------- context encoder

def test_extract_dtcs_dedupes_and_uppercases():
    assert extract_dtcs("codes p0301 and P0171, again P0301") == ["P0301", "P0171"]


def test_sensor_tokens_need_margin():
    # LTFT range is -10..10, margin 10% of 20 = 2 -> 11 is inside the slack, 13 is outside
    assert sensor_tokens({"ltft_pct": 11.0}) == []
    assert "high long-term fuel trim" in sensor_tokens({"ltft_pct": 13.0})[0]
    assert sensor_tokens({"unknown_sensor": 999}) == []


def test_encode_builds_filter_and_appends_tokens():
    q = DiagnosticQuery("rough idle", ("p0171",), VehicleProfile(make="BMW", year=2019), {"ltft_pct": 15})
    ctx = encode(q)
    assert ctx.dtcs == ["P0171"] and ctx.systems == {"powertrain"}
    assert ctx.filter["make"] == "BMW" and ctx.filter["system"] == {"powertrain"}
    assert ctx.text.startswith("rough idle") and "fuel trim" in ctx.text
    assert ctx.sensor_tokens  # surfaced for user confirmation


def test_metadata_filter_keeps_sections_without_metadata():
    assert metadata_matches({}, {"make": "BMW", "year": 2019, "system": {"powertrain"}})
    assert not metadata_matches({"make": "Suzuki"}, {"make": "BMW"})
    assert not metadata_matches({"year_from": 2010, "year_to": 2015}, {"year": 2019})
    assert metadata_matches({"year_from": 2010, "year_to": 2020}, {"year": 2019})
    assert not metadata_matches({"system": "body"}, {"system": {"powertrain"}})


# ---------------------------------------------------------------- router (Table I)

def route(text="", dtcs=(), table=None, classifier=None, **kw):
    q = DiagnosticQuery(text, tuple(dtcs))
    return Router(table or {}, classifier, **kw).route(q, encode(q))


def test_m0_single_known_code_without_text(dtc):
    assert route("P0301", table=dtc).mode is Mode.M0
    assert route("", dtcs=["P0301"], table=dtc).mode is Mode.M0


def test_m0_requires_table_hit_and_no_complaint(dtc):
    assert route("P9999", table=dtc).mode is not Mode.M0
    assert route("P0301 rough idle at startup", table=dtc).mode is not Mode.M0


def test_m1_short_part_query():
    assert route("spark plug replacement").mode is Mode.M1
    assert route("fuse F23 rating").mode is Mode.M1


def test_m1_not_for_long_or_multi_system_queries():
    long_q = "my spark plug looks worn and I am wondering what to do about it before the long trip next week"
    assert route(long_q).mode is not Mode.M1
    assert route("brake fluid and tire pressure").mode is not Mode.M1  # two systems


def test_m3_multiple_symptoms_or_systems(dtc):
    assert route("engine shakes and there is a fuel smell after startup").mode is Mode.M3
    assert route("check lights", dtcs=["P0301", "U0100"], table=dtc).mode is Mode.M3


def test_m2_default():
    assert route("why does the engine take longer to warm up in winter").mode is Mode.M2


def test_classifier_low_confidence_goes_to_m3_and_counts_a_call():
    d = route("why is my car sluggish today?", classifier=lambda t: (Mode.M2, 0.2), theta_r=0.5)
    assert d.mode is Mode.M3 and d.llm_calls == 1
    d = route("why is my car sluggish today?", classifier=lambda t: (Mode.M2, 0.9))
    assert d.mode is Mode.M2 and d.llm_calls == 1


def test_count_symptoms_distinct_groups():
    assert count_symptoms("rattle noise and squeal") == 1  # both are the "noise" group
    assert count_symptoms("rattle noise and fuel smell") == 2


def test_escalation_chain():
    assert [escalate(m) for m in (Mode.M0, Mode.M1, Mode.M2, Mode.M3)] == [Mode.M2, Mode.M2, Mode.M3, Mode.M3]


# ---------------------------------------------------------------- fusion and sufficiency

def h(i, text="x", score=0.0, page=1):
    return Hit(i, page, text, score)


def test_rrf_rewards_agreement():
    fused = rrf_fuse([[h(1), h(2), h(3)], [h(2), h(1), h(4)]], c=60)
    assert [x.id for x in fused[:2]] == [1, 2] or [x.id for x in fused[:2]] == [2, 1]
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert {x.id for x in fused} == {1, 2, 3, 4}


def test_rel_fused_is_one_when_every_retriever_agrees_on_rank_one():
    assert rel_fused(2 / 61, 2, 60) == pytest.approx(1.0)
    assert rel_fused(1 / 61, 2, 60) == pytest.approx(0.5)


def test_rel_sparse_is_term_coverage():
    assert rel_sparse("oil level dipstick", h(1, "check the oil level")) == pytest.approx(2 / 3)
    assert rel_sparse("", h(1)) == 0.0 and rel_sparse("oil", None) == 0.0


def test_coverage_and_margin():
    assert coverage([], []) == 1.0
    assert coverage(["P0301", "P0171"], [h(1, "p0301 misfire")]) == 0.5
    assert margin([1.0, 0.5, 0.25]) == pytest.approx(0.75)
    assert margin([1.0]) == 0.0 and margin([0.0, 0.0]) == 0.0


# ---------------------------------------------------------------- controller

def test_exact_dtc_lookup_skips_retrieval_and_has_full_sufficiency(searcher, dtc):
    out = run_adaptive(DiagnosticQuery("", ("P0301",)), searcher, dtc_table=dtc)
    assert out.mode is Mode.M0 and out.sufficiency == 1.0 and not out.abstained
    assert out.evidence[0].text.startswith("P0301") and out.llm_calls == 0


def test_m0_miss_falls_through_to_hybrid_in_the_same_round(searcher, dtc):
    # router believes the code is in its table, but the controller's table lacks it
    out = run_adaptive(DiagnosticQuery("P0301"), searcher, router=Router(dtc), dtc_table={})
    assert out.route.mode is Mode.M0
    assert out.steps[0].mode is Mode.M2 and not out.evidence[0].text.startswith("P0301:")


def test_unknown_code_is_routed_as_identifier_query(dtc):
    assert route("P9999", table=dtc).mode is Mode.M1


def test_answerable_query_is_accepted_and_evidence_is_relevant(searcher):
    cfg = AdaptiveConfig(tau=0.3)
    out = run_adaptive(DiagnosticQuery("how do I jump start with the starting aid terminal"), searcher, cfg=cfg)
    assert not out.abstained and out.evidence[0].page == 20


def test_unrelated_query_abstains_when_threshold_is_high(searcher):
    cfg = AdaptiveConfig(tau=0.95)
    out = run_adaptive(DiagnosticQuery("recipe for chocolate cake"), searcher, cfg=cfg)
    assert out.abstained and out.sufficiency < 0.95


def test_abstention_can_be_ablated(searcher):
    cfg = AdaptiveConfig(tau=0.99, allow_abstain=False)
    out = run_adaptive(DiagnosticQuery("recipe for chocolate cake"), searcher, cfg=cfg)
    assert not out.abstained and out.evidence


def test_retry_budget_is_bounded_and_llm_calls_are_counted(searcher):
    calls = []

    def rewriter(text, evidence, original):
        calls.append(text)
        return text + " manual"

    cfg = AdaptiveConfig(tau=0.999, t_max=3)  # never satisfied
    out = run_adaptive(DiagnosticQuery("recipe for chocolate cake"), searcher, cfg=cfg, rewriter=rewriter)
    assert len(out.steps) == 3 and len(calls) == 2  # no rewrite after the final round
    assert out.llm_calls == 2 and out.abstained
    assert [s.mode for s in out.steps][-1] is Mode.M3  # escalated to the iterative mode


def test_t_max_one_disables_the_loop(searcher):
    out = run_adaptive(DiagnosticQuery("recipe for chocolate cake"), searcher,
                       cfg=AdaptiveConfig(tau=0.999, t_max=1), rewriter=lambda *a: "x")
    assert len(out.steps) == 1 and out.llm_calls == 0


def test_router_ablation_forces_m2(searcher):
    out = run_adaptive(DiagnosticQuery("spark plug"), searcher, cfg=AdaptiveConfig(use_router=False))
    assert out.route.mode is Mode.M2 and out.steps[0].mode is Mode.M2


def test_encoder_ablation_drops_sensor_tokens(searcher):
    q = DiagnosticQuery("rough idle", sensors={"ltft_pct": 20.0})
    assert run_adaptive(q, searcher, cfg=AdaptiveConfig(use_encoder=True)).sensor_tokens
    assert run_adaptive(q, searcher, cfg=AdaptiveConfig(use_encoder=False)).sensor_tokens == []


def test_system_filter_restricts_candidates():
    texts = ["engine misfire spark plug", "door lock actuator misfire noise"]
    s = InMemorySearcher(texts, [1, 2], [{"system": "powertrain"}, {"system": "body"}],
                         embed_query=fake_embed, embed_documents=lambda ts: [fake_embed(t) for t in ts])
    hits = s.bm25("misfire", 5, {"system": {"powertrain"}})
    assert [x.page for x in hits] == [1]


# ---------------------------------------------------------------- generation helpers

def test_citation_precision_and_safety_notice():
    ev = [h(1, "brake fluid", page=40)]
    d = Diagnosis(causes=[Cause(cause="low fluid", citations=[40, 99])])
    assert citation_precision(d, ev) == 0.5
    assert "braking system" in safety_notice("brake pedal feels spongy", ev)
    assert safety_notice("how to change cabin air filter", [h(2, "cabin filter")]) is None
    # incidental words in retrieved text must not trigger a notice
    assert safety_notice("how often should tires be rotated", [h(3, "steering wheel and brake pedal")]) is None


def test_stemming_matches_inflections():
    from app.adaptive.searchers import stem, tokenize

    assert stem("starting") == stem("start") == stem("starts")
    assert stem("tires") == stem("tire") and stem("changing") == stem("change")
    assert tokenize("How do I jump start") != []
    s = InMemorySearcher(["Jump Starting Instructions", "Washer fluid level"], [1, 2],
                         embed_query=fake_embed, embed_documents=lambda ts: [fake_embed(t) for t in ts])
    assert s.bm25("jump start", 1, {})[0].page == 1


def test_per_mode_threshold_applies_to_sparse_mode_only(searcher):
    # "spark plug" routes to M1; its own threshold is unreachable while the default tau is trivial
    q = DiagnosticQuery("spark plug ignition coil")
    strict_m1 = AdaptiveConfig(tau=0.0, tau_m1=1.01, t_max=1)
    assert run_adaptive(q, searcher, cfg=strict_m1).abstained
    assert not run_adaptive(q, searcher, cfg=AdaptiveConfig(tau=0.0, tau_m1=None, t_max=1)).abstained
    assert AdaptiveConfig(tau=0.4, tau_m1=0.7).tau_for(Mode.M2) == 0.4
    assert AdaptiveConfig(tau=0.4, tau_m1=0.7).tau_for(Mode.M1) == 0.7
