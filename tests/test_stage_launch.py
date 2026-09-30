"""Тесты: раздельный запуск этапов A и B, проверка чекпойнта инициализации, ворота,
кандидаты этапа A, пробный запуск этапа B и отчёты для ручного решения."""
import copy
import csv
import dataclasses
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mayak.data import store as S
from mayak.data.splits import ROLE_TEST, ROLE_TRAIN, ROLE_VAL
from mayak.leakage import file_digest
from mayak.protocol import Protocol, ProtocolError, Stage, read_journal, run_protocol
from mayak.stages import (STAGE_KEY, GateError, InitCheckpointError, Launch, init_mismatches,
                          launch_from_config, plan_stages)

REPO = Path(__file__).resolve().parents[1]
N_HOURS = 12_000
STATIONS = [("t0", ROLE_TRAIN), ("t1", ROLE_TRAIN), ("t2", ROLE_TRAIN),
            ("v0", ROLE_VAL), ("v1", ROLE_VAL), ("x0", ROLE_TEST)]
TINY = Protocol(stages=(Stage("A", "L0", 2), Stage("B", "full", 2)),
                batch_size=2, windows_per_epoch=8, num_workers=0, seed=3, precision="32",
                val_every=1)
TINY_DATA = dict(val_windows_per_station=3)


def _write_station(root, sid, seed, n=N_HOURS):
    rng = np.random.default_rng(seed)
    h = np.arange(n)
    T = 10 + 6 * np.sin(2 * np.pi * (h - 8) / 24) + rng.standard_normal(n)
    P = 1000 + 2 * np.sin(2 * np.pi * h / 100) + 0.2 * rng.standard_normal(n)
    RH = 60 + 10 * np.cos(2 * np.pi * h / 24) + rng.standard_normal(n)
    np.savez(root / "stations" / f"{sid}.npz", T=T.astype(np.float32), P=P.astype(np.float32),
             RH=RH.astype(np.float32), valid=np.ones((n, 3), np.uint8), t0_utc_h=np.int64(0))


@pytest.fixture(scope="module")
def manifest(tmp_path_factory):
    root = tmp_path_factory.mktemp("stages_data")
    (root / "stations").mkdir()
    rows = []
    for i, (sid, role) in enumerate(STATIONS):
        _write_station(root, sid, seed=i)
        rows.append(dict(id=sid, lat=40.0 + i, lon=5.0 * i, elev=100.0, koppen="Cfb", split=role))
    path = root / "manifest.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    S._STORES.clear()
    return str(path)


def _run(arch, manifest, out, tag, launch=None, protocol=TINY, data_config=None, **kw):
    return run_protocol(arch, manifest, protocol, out_root=str(out), accelerator="cpu", tag=tag,
                        enable_progress_bar=False, data_config=data_config or TINY_DATA,
                        launch=launch, **kw)


def _weights(path):
    return torch.load(path, map_location="cpu", weights_only=False)["state_dict"]


def _same_weights(a, b):
    wa, wb = _weights(a), _weights(b)
    return wa.keys() == wb.keys() and all(torch.equal(wa[k], wb[k]) for k in wa)


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stage(journal, name):
    return next(s for s in journal["stages"] if s["name"] == name)


# Выбор этапов и настройки запуска.

def test_stage_selection_rules():
    assert [s.name for _i, s in plan_stages(TINY, Launch())] == ["A", "B"]
    assert [i for i, _s in plan_stages(TINY, Launch(stages="B", init_from="x"))] == [1]
    bad = [dict(stages="B"), dict(stages="A", init_from="x"), dict(stages="B,A", init_from="x"),
           dict(stages="C"), dict(stages="B", init_from="x", probe_steps=2),
           dict(stages="A", probe_steps=1), dict(candidates=["Z"])]
    for kw in bad:
        with pytest.raises(ProtocolError):
            plan_stages(TINY, Launch(**kw))
    for kw in (dict(require_gate=0), dict(probe_steps=0), dict(stages="")):
        with pytest.raises(ValueError):
            Launch(**kw)


def test_launch_is_not_part_of_the_protocol():
    """Выбор этапов не меняет протокол: чекпойнт B сравним с прогоном одной командой."""
    tr = _load_script("train")
    ap = tr.make_parser()
    full = ap.parse_args([])
    split = ap.parse_args(["--stages", "B", "--init-from", "a.ckpt", "--require-gate", "1.05",
                           "--probe-steps", "5", "--candidates"])
    from mayak.protocol import protocol_from_args
    assert protocol_from_args(full) == protocol_from_args(split)
    assert tr.launch_from_args(split) == Launch(stages=("B",), init_from="a.ckpt",
                                                require_gate=1.05, probe_steps=5, candidates=())
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(REPO / "conf"), version_base="1.3"):
        cfg = compose("config", overrides=["run.stages=[B]", "run.init_from=a.ckpt",
                                           "run.require_gate=1.05", "run.probe_steps=5",
                                           "run.candidates=[]"])
        default = compose("config")
    assert launch_from_config(OmegaConf.to_container(cfg.run)) == tr.launch_from_args(split)
    assert launch_from_config(OmegaConf.to_container(default.run)) == Launch()
    assert _load_script("run").to_launch(cfg) == tr.launch_from_args(split)


def _fake_checkpoint():
    a = TINY.stages[0]
    ck = {"hyper_parameters": dict(arch="dlinear", stage="A", protocol=TINY.to_dict(),
                                   model_config={"arch": "dlinear", "input_len": 336},
                                   data_config={"manifest": "a.csv", "val_seed": 0}),
          STAGE_KEY: dict(stage="A", data_key="K", probe_steps=None),
          "mayak_selection": dict(windows_digest="V", history=dict(curriculum="L0"))}
    kw = dict(arch="dlinear", protocol=TINY.to_dict(),
              model_config={"arch": "dlinear", "input_len": 336},
              data_config={"manifest": "/elsewhere/a.csv", "val_seed": 0}, stage=a,
              data_key="K", val_digest="V")
    return ck, kw


def test_init_mismatches_lists_every_difference():
    ck, kw = _fake_checkpoint()
    assert init_mismatches(ck, **kw) == [], "путь к манифесту сам по себе не отличие"
    bad = copy.deepcopy(ck)
    bad["hyper_parameters"]["model_config"]["input_len"] = 168
    bad["hyper_parameters"]["protocol"]["seed"] = 9
    bad["hyper_parameters"]["data_config"]["val_seed"] = 1
    bad[STAGE_KEY].update(probe_steps=1, data_key="K0")
    bad["mayak_selection"] = dict(windows_digest="W", history=dict(curriculum="full"))
    text = "\n".join(init_mismatches(bad, **kw))
    for part in ("модель.input_len", "протокол.seed", "данные.val_seed", "пробного запуска",
                 "кэше", "набор валидации", "куррикулумом"):
        assert part in text, part
    old = copy.deepcopy(ck)
    del old[STAGE_KEY]
    old["hyper_parameters"]["stage"] = "B"
    text = "\n".join(init_mismatches(old, **kw))
    assert "нет записи об этапе" in text and "этап" in text


# Раздельный прогон МАЯК против прогона одной командой.

@pytest.fixture(scope="module")
def split_runs(manifest, tmp_path_factory):
    out = tmp_path_factory.mktemp("stage_runs")
    single = _run("mayak", manifest, out / "single", "m", launch=Launch(require_gate=1e9))
    only_a = _run("mayak", manifest, out / "split", "m", launch=Launch(stages="A"))
    a_best = _stage(only_a, "A")["best_ckpt"]
    only_b = _run("mayak", manifest, out / "split", "m",
                  launch=Launch(stages="B", init_from=a_best))
    return dict(out=out, single=single, only_a=only_a, only_b=only_b, a_best=a_best)


def test_split_run_reproduces_single_run(split_runs):
    r = split_runs
    assert _same_weights(_stage(r["single"], "A")["best_ckpt"], r["a_best"])
    assert _same_weights(r["single"]["final_ckpt"], r["only_b"]["final_ckpt"]), \
        "этап B, запущенный отдельно, дал другой чекпойнт"


def test_split_journal_links_stage_b_to_stage_a(split_runs):
    r = split_runs
    assert r["only_a"]["final_ckpt"] is None, "после одного этапа A итогового чекпойнта нет"
    j = read_journal(r["out"] / "split" / "m")
    assert j == r["only_b"]
    assert [s["name"] for s in j["stages"]] == ["A", "B"]
    init = _stage(j, "B")["init_from"]
    assert init["digest"] == file_digest(r["a_best"]) and init["stage"] == "A"
    assert init["is_best"] is True and init["journal"].endswith("protocol.json")
    assert init["report"]["mse_ratio"] is not None
    rec = torch.load(j["final_ckpt"], map_location="cpu", weights_only=False)[STAGE_KEY]
    assert rec["stage"] == "B" and rec["index"] == 1 and rec["init_from"]["digest"] == \
        init["digest"]
    single_init = _stage(r["single"], "B")["init_from"]
    assert single_init["gate"]["passed"] is True and single_init["step"] == init["step"]


def test_stage_a_report_covers_every_candidate(split_runs):
    from mayak.stage_report import PLOT_FILES, read_report
    a = _stage(split_runs["only_a"], "A")
    report = read_report(a["report"])
    assert [e["step"] for e in report["candidates"]] == [c["step"] for c in a["candidates"]]
    assert [c["step"] for c in a["candidates"]] == [1, 2]
    assert sum(e["is_best"] for e in report["candidates"]) == 1
    assert report["val_set"]["fingerprint"] == a["val_set"]
    assert report["val_set"]["history"]["curriculum"] == "L0"
    for e in report["candidates"]:
        assert e["mse_ratio"] > 0 and 0.0 <= e["picp90"] <= 1.0
        lo, hi = e["mse_ratio_ci"]
        assert lo <= hi and [row["lead"] for row in e["by_lead"]] == report["leads"]
    assert not report.get("plots_error")
    assert sorted(Path(p).name for p in a["plots"]) == sorted(PLOT_FILES)
    assert all(Path(p).exists() for p in a["plots"])
    for c in a["candidates"]:
        rec = torch.load(c["ckpt"], map_location="cpu", weights_only=False)
        assert rec[STAGE_KEY]["step"] == c["step"] == rec["global_step"]


# Проверки и ворота на DLinear: быстро и без случайности в прямом проходе.

@pytest.fixture(scope="module")
def dl_a(manifest, tmp_path_factory):
    out = tmp_path_factory.mktemp("dl")
    journal = _run("dlinear", manifest, out, "dl", launch=Launch(stages="A"))
    return out, journal


@pytest.mark.parametrize("case", ["model", "arch", "protocol", "data", "missing"])
def test_incompatible_init_checkpoint_is_rejected(case, manifest, dl_a, tmp_path):
    _out, journal = dl_a
    a_best = _stage(journal, "A")["best_ckpt"]
    kw, expect = {}, None
    if case == "model":
        kw, expect = dict(model_config={"input_len": 168}), "модель.input_len"
    elif case == "protocol":
        kw, expect = dict(protocol=dataclasses.replace(TINY, seed=4)), "протокол.seed"
    elif case == "data":
        kw, expect = dict(data_config=dict(val_windows_per_station=2)), \
            "данные.val_windows_per_station"
    elif case == "missing":
        a_best, expect = str(tmp_path / "none.ckpt"), "не найден"
    arch = "gru" if case == "arch" else "dlinear"
    if case == "arch":
        expect = "архитектура"
    with pytest.raises(InitCheckpointError, match=expect):
        _run(arch, manifest, tmp_path, "b", launch=Launch(stages="B", init_from=a_best), **kw)
    assert not (tmp_path / "b").exists(), "этап B начался или оставил следы"


def test_gate_blocks_stage_b_on_bad_field(manifest, dl_a, tmp_path):
    _out, journal = dl_a
    ck = torch.load(_stage(journal, "A")["best_ckpt"], map_location="cpu", weights_only=False)
    ck["state_dict"]["model.lin_trend.bias"] += 100.0
    bad = tmp_path / "bad.ckpt"
    torch.save(ck, bad)
    with pytest.raises(GateError, match="не начат"):
        _run("dlinear", manifest, tmp_path, "b",
             launch=Launch(stages="B", init_from=str(bad), require_gate=1.05))
    assert not (tmp_path / "b" / "stageB").exists()
    ok = _run("dlinear", manifest, tmp_path, "ok",
              launch=Launch(stages="B", init_from=_stage(journal, "A")["best_ckpt"],
                            require_gate=1e9))
    assert _stage(ok, "B")["init_from"]["gate"]["passed"] is True


def test_gate_in_single_command_stops_before_stage_b(manifest, tmp_path):
    with pytest.raises(GateError):
        _run("dlinear", manifest, tmp_path, "g", launch=Launch(require_gate=1e-9))
    j = read_journal(tmp_path / "g")
    assert [s["name"] for s in j["stages"]] == ["A"] and j["final_ckpt"] is None
    assert j["stages"][0]["report_best"]["gate"]["passed"] is False
    assert not (tmp_path / "g" / "stageB").exists()


def test_stage_b_can_start_from_any_candidate(manifest, dl_a, tmp_path):
    _out, journal = dl_a
    a = _stage(journal, "A")
    other = next(c for c in a["candidates"] if c["step"] != a["best_step"])
    j = _run("dlinear", manifest, tmp_path, "c",
             launch=Launch(stages="B", init_from=other["ckpt"]))
    init = _stage(j, "B")["init_from"]
    assert init["step"] == other["step"] and init["is_best"] is False
    assert init["digest"] == other["digest"] and init["report"]["step"] == other["step"]
    assert j["final_ckpt"] is not None


def test_probe_is_the_start_of_the_full_stage(manifest, dl_a, tmp_path):
    _out, journal = dl_a
    a_best = _stage(journal, "A")["best_ckpt"]
    full = _run("dlinear", manifest, tmp_path, "full",
                launch=Launch(stages="B", init_from=a_best, candidates=["B"]))
    probe = _run("dlinear", manifest, tmp_path, "probe",
                 launch=Launch(stages="B", init_from=a_best, probe_steps=1))
    b = _stage(probe, "B")
    assert probe["final_ckpt"] is None and b["probe_steps"] == 1 and b["steps_done"] == 1
    step1 = next(c for c in _stage(full, "B")["candidates"] if c["step"] == 1)
    assert _same_weights(b["best_ckpt"], step1["ckpt"])
    rec = torch.load(b["best_ckpt"], map_location="cpu", weights_only=False)[STAGE_KEY]
    assert rec["probe_steps"] == 1

    rep = _load_script("stage_report")
    rep.main(["b", "--runs", str(tmp_path / "full"), str(tmp_path / "probe"),
              "--out", str(tmp_path / "cmp.png")])
    from mayak.stage_report import stage_runs
    rows, warnings = stage_runs([str(tmp_path / "full"), str(tmp_path / "probe")])
    assert warnings == [] and rows[0]["common_step"] is not None
    assert all(r["loss_at_common"] is not None for r in rows)
    assert (tmp_path / "cmp.png").exists()


def test_offline_report_matches_report_written_by_training(dl_a):
    from mayak.stage_report import read_report
    out, journal = dl_a
    path = _stage(journal, "A")["report"]
    before = read_report(path)
    _load_script("stage_report").main(["a", "--run", str(out / "dl"), "--examples", "2"])
    after = read_report(path)
    for x, y in zip(before["candidates"], after["candidates"]):
        assert x["digest"] == y["digest"]
        assert y["mse_ratio"] == pytest.approx(x["mse_ratio"])
        assert y["mse_ratio_ci"] == pytest.approx(x["mse_ratio_ci"])


def test_train_script_stops_with_gate_exit_code(manifest, tmp_path, capsys):
    tr = _load_script("train")
    common = ["--arch", "dlinear", "--manifest", manifest, "--accelerator", "cpu",
              "--out-root", str(tmp_path), "--steps-a", "2", "--steps-b", "2", "--batch", "2",
              "--windows", "8", "--workers", "0", "--precision", "32", "--val-every", "1",
              "--seed", "3"]
    tr.main(common + ["--stages", "A"])
    shown = capsys.readouterr().out
    assert "--stages B --init-from" in shown and "Отчёт этапа A" in shown
    best = _stage(read_journal(tmp_path / "dlinear"), "A")["best_ckpt"]
    with pytest.raises(SystemExit) as e:
        tr.main(common + ["--stages", "B", "--init-from", best, "--require-gate", "1e-9"])
    assert e.value.code == 3
    assert not (tmp_path / "dlinear" / "stageB").exists()
    assert json.loads((tmp_path / "dlinear" / "protocol.json").read_text())["final_ckpt"] is None
