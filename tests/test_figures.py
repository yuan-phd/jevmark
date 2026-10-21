"""scripts/make_figures.py on the committed metrics files (figures task)."""

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_figures", REPO / "scripts" / "make_figures.py")
make_figures = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = make_figures
spec.loader.exec_module(make_figures)


def test_every_figure_is_drawn_from_the_committed_metrics(tmp_path):
    assert make_figures.main(["--out", str(tmp_path)]) == 0
    for name in make_figures.EXPECTED:
        path = tmp_path / name
        assert path.is_file() and path.stat().st_size > 10_000, name
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", name
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(make_figures.EXPECTED)


def test_the_script_reads_no_per_question_files():
    source = (REPO / "scripts" / "make_figures.py").read_text()
    code = source.split('"""', 2)[2]  # everything after the module docstring
    assert "results.jsonl" not in code and "replies.jsonl" not in code and "read_results" not in code and "log.jsonl" not in code


def test_phase_4_figures_are_listed_and_the_pending_ones_documented():
    for name in ("v4_offpolicy.png", "v4_noise_sweep.png", "v4_scale.png", "v4_emotion_n_curve.png"):
        assert name in make_figures.EXPECTED
    doc = make_figures.__doc__
    for name, (source, waits_for) in make_figures.PENDING.items():
        assert name.split(",")[0] not in make_figures.EXPECTED or "seed ranges" in name
        assert source.startswith("runs/") and waits_for
    assert "emotion panel" in doc and "1.7B seed ranges" in doc


def test_scale_values_draw_a_seed_range_only_when_the_stage_file_holds_more_than_one_seed():
    def stage(n_seeds, deltas):
        block = {m: {"mean": 0.9, "range": [0.89, 0.91], "n_seeds": n_seeds} for m in ("accuracy", "ece")}
        return {"seeds": {"training_seeds": block, "full_sft": {"training_seeds": block}},
                "n_curve": {"5000": {"direct_brier": {"accuracy": 0.95, "ece": 0.02}}},
                "forgetting": {"test_indomain": {"minus_zero_shot": {f"direct_brier_n5000_s{k}": {"accuracy": {"delta": d}} for k, d in enumerate(deltas)}}}}

    one = stage(1, [-0.04])
    assert make_figures.scale_values(one, "direct_brier", "accuracy") == (0.95, None)
    assert make_figures.forgetting_values(one, "direct_brier") == (-0.04, None)
    three = stage(3, [-0.04, -0.02, -0.03])
    assert make_figures.scale_values(three, "direct_brier", "accuracy") == (0.9, [0.89, 0.91])
    value, rng = make_figures.forgetting_values(three, "direct_brier")
    assert abs(value + 0.03) < 1e-12 and rng == [-0.04, -0.02]



def test_paper_figures_use_the_paper_names_and_carry_no_source_line(tmp_path, monkeypatch):
    import re

    from matplotlib.text import Text

    texts: dict[str, list[str]] = {}
    real_save = make_figures.save

    def save(fig, out, name, written):
        texts[name] = [t.get_text() for t in fig.findobj(Text) if t.get_text()]
        real_save(fig, out, name, written)

    monkeypatch.setattr(make_figures, "save", save)
    assert make_figures.main(["--paper", "--out", str(tmp_path)]) == 0
    assert sorted(texts) == sorted(f.__name__ + ".png" for f in make_figures.PAPER_FIGURES)
    for name, shown in texts.items():
        joined = "\n".join(shown)
        assert "Source" not in joined and "runs/" not in joined, name
        assert not re.search(r"direct_|positive_sft|full_sft|zero_shot|ips_|outcome_minus|sft_cont|sft_06b|RLCD|v3_|\bv[1-4]\b|[Pp]hase", joined), (name, joined)
    assert any("CAPS-Brier" in t for t in texts["v3_n_curve.png"]) and any("CAPS-log" in t for t in texts["v4_offpolicy.png"])
