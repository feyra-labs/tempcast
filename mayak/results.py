"""Таблицы результатов в JSON: по файлу на таблицу, с записью о прогоне."""
import json
import os

from mayak.provenance import provenance
from mayak.stages import jsonable

SCHEMA = 1


def save_json(obj, path):
    """Записывает значение в файл строгим JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(jsonable(obj), f, ensure_ascii=False, indent=1, allow_nan=False)
    return path


def run_record(**info):
    """Запись о прогоне для заголовка каждого файла таблицы."""
    return dict(schema=SCHEMA, **provenance(), **info)


def set_record(record, dataset):
    """Запись о прогоне для таблиц одного набора окон."""
    return dict(record, station_roles=list(dataset.station_splits),
                time_key=dataset.time_key, n_windows=len(dataset))


def reliability_table(ev):
    """Надёжность одной модели по сырым выходам."""
    return dict(pit=ev.pit_histogram(), pit_by_lead_bin=ev.pit_by_lead_bin(),
                reliability=ev.reliability(), sharpness=ev.sharpness_coverage())


def lead_tables(res):
    """Метрики по лидам и по всему горизонту из результата оценки набора."""
    return {"metrics": dict(main=res["main"], leads=res["leads"], overall=res["overall"])}


def evaluation_tables(res):
    """Таблицы одного набора окон из результата оценки."""
    bench = res["bench"]
    tables = {
        **lead_tables(res),
        "history": dict(grid=list(bench.grid), models=res["history"]),
        "breakdowns": res["breakdowns"],
        "reliability": dict(model=res["main"], **reliability_table(res["reliability"]),
                            sharpness_all=res["sharpness"], coverage=res["coverage"]),
    }
    if res.get("calibrated") is not None:
        tables["calibrated"] = dict(model=res["main"], **res["calibrated"])
    return tables


def transfer_tables(transfer):
    """Сопоставление внешнего и внутреннего теста в виде, пригодном для JSON."""
    return [dict(model=name, internal=tag, table=tbl) for (name, tag), tbl in transfer.items()]


def write_tables(tables, out_dir, record):
    """Пишет каждую таблицу в свой файл вместе с записью о прогоне."""
    return [save_json(dict(run=record, table=table), os.path.join(out_dir, f"{name}.json"))
            for name, table in tables.items()]


__all__ = ["SCHEMA", "evaluation_tables", "lead_tables", "reliability_table", "run_record",
           "set_record", "transfer_tables", "write_tables"]
