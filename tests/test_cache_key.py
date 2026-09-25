"""Тесты: ключ кэша зависит от кода правил, а не от ручных версий."""
import csv
import importlib.util
import json
import runpy
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from mayak import codehash as C
from mayak import leakage as LK
from mayak.data import store as S
from mayak.data.qc import DEFAULT_QC, QCConfig
from mayak.data.splits import LAYOUT_CODE, layout_fingerprint
from mayak.timeaxis import to_utc_hour

REPO = Path(__file__).resolve().parents[1]
N_HOURS = 12_000
T0 = int(to_utc_hour(datetime(2019, 12, 30, 5)))
QC_SOURCE = Path(C.source_path("mayak.data.qc"))
SPLITS_SOURCE = Path(C.source_path("mayak.data.splits"))

SAMPLE = '''"""Модуль."""
import numpy as np

LIMIT = 3


def f(x, y):
    """Сумма с порогом."""
    # комментарий
    return np.minimum(x + y, LIMIT)


class A:
    """Класс."""

    def g(self):
        """Метод."""
        return 1
'''


def _write_station(root, sid, seed, n=N_HOURS):
    rng = np.random.default_rng(seed)
    h = np.arange(n)
    T = 10 + 6 * np.sin(2 * np.pi * h / 24) + 0.3 * rng.standard_normal(n)
    P = 1000 + 2 * np.sin(2 * np.pi * h / 100) + 0.2 * rng.standard_normal(n)
    RH = 60 + 10 * np.cos(2 * np.pi * h / 24) + rng.standard_normal(n)
    np.savez(Path(root) / "stations" / f"{sid}.npz", T=T.astype(np.float32),
             P=P.astype(np.float32), RH=RH.astype(np.float32),
             valid=np.ones((n, 3), np.uint8), t0_utc_h=np.int64(T0))


def _make_dataset(root, n_stations=2):
    root = Path(root)
    (root / "stations").mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n_stations):
        _write_station(root, f"s{i}", seed=i)
        rows.append(dict(id=f"s{i}", lat=40.0 + i, lon=10.0 * i, elev=100.0, koppen="Cfb",
                         split="train"))
    with open(root / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return str(root / "manifest.csv")


@pytest.fixture
def manifest(tmp_path):
    return _make_dataset(tmp_path / "data")


@pytest.fixture(autouse=True)
def _fresh_process_memo():
    S._STORES.clear()
    yield
    S._STORES.clear()


def _key(manifest, qc_cfg=DEFAULT_QC):
    return S.cache_key(S.key_payload(manifest, qc_cfg=qc_cfg))


def _redirect(monkeypatch, unit, path):
    """Подменяет файл, из которого читается единица кода."""
    real = C.source_path
    monkeypatch.setattr(C, "source_path", lambda u: str(path) if u == unit else real(u))


def _qc_copy(tmp_path, name, edit=None):
    src = QC_SOURCE.read_text(encoding="utf-8")
    text = src if edit is None else edit(src)
    assert edit is None or text != src, "правка не применилась"
    path = tmp_path / f"{name}.py"
    path.write_text(text, encoding="utf-8")
    return path


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _meta(path):
    with open(Path(path) / "meta.json", encoding="utf-8") as f:
        return json.load(f)


def test_digest_ignores_comments_docstrings_and_layout():
    base = C.source_digest(SAMPLE)
    same = [
        SAMPLE.replace("# комментарий", "# другой комментарий\n    # и ещё один"),
        SAMPLE.replace('"""Сумма с порогом."""', '"""Совсем другое описание.\n\n    Args:\n'
                                                '        x: число.\n    """'),
        SAMPLE.replace('"""Модуль."""', '"""Другой модуль."""'),
        SAMPLE.replace('        """Метод."""\n', ""),
        SAMPLE.replace("return np.minimum(x + y, LIMIT)", "return np.minimum(\n        x+y,\n"
                                                           "        LIMIT,\n    )"),
        SAMPLE.replace("\n\ndef f", "\n\n\n\ndef f"),
    ]
    for text in same:
        assert text != SAMPLE
        assert C.source_digest(text) == base


def test_digest_changes_on_semantic_edit():
    base = C.source_digest(SAMPLE)
    changed = [SAMPLE.replace("LIMIT = 3", "LIMIT = 4"),
               SAMPLE.replace("x + y", "x - y"),
               SAMPLE.replace("np.minimum", "np.maximum"),
               SAMPLE.replace("return 1", "return 2"),
               SAMPLE.replace("def f(x, y):", "def f(x, y=0):")]
    digests = {C.source_digest(t) for t in changed}
    assert base not in digests and len(digests) == len(changed)


def test_digest_of_selected_names():
    only_f = C.source_digest(SAMPLE, ("f",))
    assert only_f == C.source_digest(SAMPLE.replace("return 1", "return 2"), ("f",))
    assert only_f != C.source_digest(SAMPLE.replace("x + y", "x - y"), ("f",))
    assert C.source_digest(SAMPLE, ("f", "LIMIT")) == C.source_digest(SAMPLE, ("LIMIT", "f"))
    with pytest.raises(ValueError, match="нет определений"):
        C.source_digest(SAMPLE, ("f", "missing"))


def test_write_into_object_belongs_to_its_name():
    text = 'KEYS = {"a": 1}\nKEYS["b"] = 2\nOTHER = 0\n'
    assert C.source_digest(text, ("KEYS",)) != \
        C.source_digest(text.replace('KEYS["b"] = 2', 'KEYS["b"] = 3'), ("KEYS",))
    assert C.source_digest(text, ("OTHER",)) == \
        C.source_digest(text.replace('KEYS["b"] = 2', 'KEYS["b"] = 3'), ("OTHER",))


def test_semantic_edit_in_qc_copy_changes_cache_key(manifest, tmp_path, monkeypatch):
    k0 = _key(manifest)
    _redirect(monkeypatch, "mayak.data.qc", _qc_copy(tmp_path, "same"))
    assert _key(manifest) == k0

    cosmetic = {
        "comment": lambda s: s.replace("MAD_TO_SD = 1.4826",
                                       "MAD_TO_SD = 1.4826  # переход от MAD к разбросу"),
        "docstring": lambda s: s.replace('"""Коды причин отбраковки."""',
                                         '"""Коды причин, по которым значение отбраковано."""'),
        "module_docstring": lambda s: s.replace("Единый контроль качества.",
                                                "Контроль качества для всех источников."),
        "layout": lambda s: s.replace("MAD_TO_SD = 1.4826", "MAD_TO_SD = (\n    1.4826\n)"),
    }
    for name, edit in cosmetic.items():
        _redirect(monkeypatch, "mayak.data.qc", _qc_copy(tmp_path, name, edit))
        assert _key(manifest) == k0, name

    semantic = {
        "constant": lambda s: s.replace("MAD_TO_SD = 1.4826", "MAD_TO_SD = 1.5"),
        "default": lambda s: s.replace("jump_thresh: float = 8.0", "jump_thresh: float = 9.0"),
    }
    keys = set()
    for name, edit in semantic.items():
        _redirect(monkeypatch, "mayak.data.qc", _qc_copy(tmp_path, name, edit))
        keys.add(_key(manifest))
    assert k0 not in keys and len(keys) == len(semantic)


def test_repeat_build_only_hashes(manifest, monkeypatch):
    path, built = S.build_cache(manifest)
    assert built

    def boom(*args, **kwargs):
        raise AssertionError("станции не должны обрабатываться повторно")

    monkeypatch.setattr(S, "process_station", boom)
    assert S.build_cache(manifest) == (path, False)
    S._STORES.clear()
    assert S.get_store(manifest).path == path


def test_rebuild_reasons_name_changed_parts(manifest, tmp_path, monkeypatch):
    path, _ = S.build_cache(manifest)
    assert _meta(path)["rebuild"] == dict(
        previous_key=None, reasons=["первая сборка: готовых сборок этого манифеста нет"])

    def rebuild(**kwargs):
        new_path, built = S.build_cache(manifest, **kwargs)
        assert built
        meta = _meta(new_path)
        return meta["rebuild"]

    _write_station(Path(manifest).parent, "s0", seed=99)
    rec = rebuild()
    assert rec["previous_key"] == Path(path).name
    assert rec["reasons"] == ["источники: изменено содержимое 1 (s0)"]

    rec = rebuild(qc_cfg=QCConfig(spike_thresh=7.0))
    assert rec["reasons"] == ["конфиг QC: изменились spike_thresh"]

    edited = _qc_copy(tmp_path, "semantic",
                      lambda s: s.replace("MAD_TO_SD = 1.4826", "MAD_TO_SD = 1.5"))
    _redirect(monkeypatch, "mayak.data.qc", edited)
    rec = rebuild(qc_cfg=QCConfig(spike_thresh=7.0))
    assert rec["reasons"] == ["код правил: изменились mayak.data.qc"]

    monkeypatch.setattr(S, "TIME_LAYOUT", dict(S.TIME_LAYOUT, n_blocks=6))
    rec = rebuild(qc_cfg=QCConfig(spike_thresh=7.0))
    assert rec["reasons"] == ["раскладка сплитов: изменились params.n_blocks"]

    final, _ = S.build_cache(manifest, qc_cfg=QCConfig(spike_thresh=7.0), force=True)
    assert _meta(final)["rebuild"]["reasons"] == ["пересборка по требованию при том же ключе"]


def test_rebuild_reasons_for_payload_of_other_shape():
    old = {"format": "3", "stations": []}
    assert S.rebuild_reasons(old, {"sources": []}) == [
        "прошлая сборка сделана с ключом другого состава, части не сравнить"]
    new = {"sources": [["a", "1", []], ["b", "2", [["lon", 1.0]]]]}
    old = {"sources": [["b", "2", [["lon", 2.0]]], ["c", "3", []]]}
    assert S.rebuild_reasons(old, new) == [
        "источники: добавлены 1 (a); удалены 1 (c); изменены метаданные 1 (b)"]


def test_build_cache_script_prints_reason(manifest, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["build_cache", "--manifest", manifest, "--jobs", "1"])
    runpy.run_path(str(REPO / "scripts" / "build_cache.py"), run_name="__main__")
    out = capsys.readouterr().out
    assert "причина сборки:" in out and "первая сборка" in out
    assert "отпечаток кода правил:" in out and "mayak.data.qc" in out

    runpy.run_path(str(REPO / "scripts" / "build_cache.py"), run_name="__main__")
    out = capsys.readouterr().out
    assert "уже актуален" in out and "причина сборки" not in out


def test_code_sets_are_closed():
    assert C.missing_dependencies(S.CACHE_CODE, S.CACHE_CODE_IGNORED) == []
    assert C.missing_dependencies(LAYOUT_CODE) == []
    bookkeeping = ("mayak.codehash", "command_line", "write_source_build")
    for name in ("make_era5", "make_ghcnh"):
        spec = _load_script(name).BUILD_CODE
        assert C.missing_dependencies(spec, bookkeeping) == [], name


def test_closure_check_catches_dropped_dependency():
    spec = {k: v for k, v in S.CACHE_CODE.items() if k != "mayak.data.masking"}
    assert "mayak.data.qc: нет модуля mayak.data.masking" in \
        C.missing_dependencies(spec, S.CACHE_CODE_IGNORED)
    spec = dict(S.CACHE_CODE)
    spec["mayak.data.store"] = tuple(n for n in spec["mayak.data.store"] if n != "qc_elev")
    assert "mayak.data.store: нет определения qc_elev" in \
        C.missing_dependencies(spec, S.CACHE_CODE_IGNORED)
    spec = dict(S.CACHE_CODE, **{"mayak.constants": ("H",)})
    assert "mayak.data.splits: из mayak.constants нужно L_MAX" in \
        C.missing_dependencies(spec, S.CACHE_CODE_IGNORED)


def test_stale_source_build_is_rejected(manifest, tmp_path):
    builder = tmp_path / "builder.py"
    builder.write_text('"""Сборщик."""\nSCALE = 1.0\n', encoding="utf-8")
    folder = Path(manifest).parent
    S.write_source_build(str(folder), "builder", "python builder.py --out data",
                         {str(builder): None, "mayak.timeaxis": None})
    rec = json.loads((folder / S.SOURCE_BUILD_NAME).read_text(encoding="utf-8"))
    assert rec["builder"] == "builder" and set(rec["code"]) == {str(builder), "mayak.timeaxis"}
    S.check_sources(manifest)

    builder.write_text('"""Сборщик файлов."""\nSCALE = 1.0  # множитель\n', encoding="utf-8")
    path, built = S.build_cache(manifest)
    assert built

    builder.write_text('"""Сборщик."""\nSCALE = 2.0\n', encoding="utf-8")
    for call in (lambda: S.build_cache(manifest), lambda: S.get_store(manifest)):
        S._STORES.clear()
        with pytest.raises(RuntimeError, match="python builder.py --out data") as e:
            call()
        assert str(builder) in str(e.value) and "mayak.timeaxis" not in str(e.value)

    builder.unlink()
    with pytest.raises(RuntimeError, match="собраны другим кодом"):
        S.check_sources(manifest)


def test_dataset_without_source_build_is_not_checked(manifest):
    assert not (Path(manifest).parent / S.SOURCE_BUILD_NAME).exists()
    S.check_sources(manifest)


def test_split_records_are_checked_by_layout_code(tmp_path, monkeypatch):
    rec = LK._split_state()
    assert rec["layout_code"] == layout_fingerprint()
    LK._check_split_state(dict(rec), "запись")

    with pytest.raises(LK.LeakageError, match="код раскладки"):
        LK._check_split_state(dict(rec, layout_code="0" * 16), "запись")
    with pytest.raises(LK.LeakageError, match="код раскладки"):
        LK._check_split_state({k: v for k, v in rec.items() if k != "layout_code"}, "запись")

    src = SPLITS_SOURCE
    comment = tmp_path / "comment.py"
    comment.write_text(src.read_text(encoding="utf-8").replace(
        "MIN_BLOCK_HOURS = H + 1", "MIN_BLOCK_HOURS = H + 1  # хотя бы одно окно"),
        encoding="utf-8")
    _redirect(monkeypatch, "mayak.data.splits", comment)
    LK._check_split_state(rec, "запись")

    semantic = tmp_path / "semantic.py"
    shutil.copy(src, semantic)
    semantic.write_text(src.read_text(encoding="utf-8").replace(
        "MIN_BLOCK_HOURS = H + 1", "MIN_BLOCK_HOURS = H + 2"), encoding="utf-8")
    _redirect(monkeypatch, "mayak.data.splits", semantic)
    with pytest.raises(LK.LeakageError, match="построено при других правилах сплитов"):
        LK._check_split_state(rec, "запись")
