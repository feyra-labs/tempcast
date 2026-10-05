"""Таблицы результатов в JSON: по файлу на таблицу, с записью о прогоне.

Каталог результатов хранит то, из чего автор заполняет таблицы описания проекта. Каждый
файл содержит запись о прогоне (формат, время, коммит, версии библиотек, чекпойнты, сид
оценки, роли станций оценённых наборов) и саму таблицу. Таблицы одного набора окон
пишутся в его каталог; запись о прогоне в них несёт роли станций, временное окно и число
окон этого набора. Файлы пишутся строгим JSON: пропуски и бесконечности становятся null.
Таблицы берутся из уже посчитанного результата оценки, ничего не пересчитывается.
"""
import os

from mayak.calibration import save_json
from mayak.provenance import provenance

SCHEMA = 1


def run_record(**info):
    """Запись о прогоне для заголовка каждого файла таблицы.

    Args:
        **info: сведения о прогоне: чекпойнты, манифест, сид оценки и прочее.

    Returns:
        Словарь с версией формата, происхождением прогона и переданными сведениями.
    """
    return dict(schema=SCHEMA, **provenance(), **info)


def set_record(record, dataset):
    """Запись о прогоне для таблиц одного набора окон.

    Args:
        record: общая запись о прогоне.
        dataset: набор окон, по которому посчитаны таблицы.

    Returns:
        Копия записи с ролями станций, временным окном и числом окон набора.
    """
    return dict(record, station_roles=list(dataset.station_splits),
                time_key=dataset.time_key, n_windows=len(dataset))


def reliability_table(ev):
    """Надёжность одной модели по сырым выходам.

    Args:
        ev: оценка модели на наборе окон.

    Returns:
        Словарь: гистограмма PIT по всему горизонту и по бинам лидов, диаграмма
        надёжности и острота против покрытия.
    """
    return dict(pit=ev.pit_histogram(), pit_by_lead_bin=ev.pit_by_lead_bin(),
                reliability=ev.reliability(), sharpness=ev.sharpness_coverage())


def lead_tables(res):
    """Метрики по лидам и по всему горизонту из результата оценки набора.

    Args:
        res: результат оценки набора: основная модель, сводки по лидам и по всему
            горизонту.

    Returns:
        Словарь из одной таблицы metrics.
    """
    return {"metrics": dict(main=res["main"], leads=res["leads"], overall=res["overall"])}


def evaluation_tables(res):
    """Таблицы одного набора окон из результата оценки.

    Args:
        res: результат оценки набора, как его возвращает стенд оценки.

    Returns:
        Словарь из имени таблицы в её содержимое. Раздел после калибровки есть только
        тогда, когда оценка шла с конформной таблицей.
    """
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
    """Сопоставление внешнего и внутреннего теста в виде, пригодном для JSON.

    Args:
        transfer: словарь из пары «модель, набор станций внутреннего теста» в таблицу.

    Returns:
        Список строк с именем модели, набором станций и таблицей.
    """
    return [dict(model=name, internal=tag, table=tbl) for (name, tag), tbl in transfer.items()]


def write_tables(tables, out_dir, record):
    """Пишет каждую таблицу в свой файл вместе с записью о прогоне.

    Args:
        tables: словарь из имени таблицы в содержимое; имя становится именем файла.
        out_dir: каталог; создаётся при необходимости.
        record: запись о прогоне.

    Returns:
        Пути к записанным файлам в порядке таблиц.
    """
    return [save_json(dict(run=record, table=table), os.path.join(out_dir, f"{name}.json"))
            for name, table in tables.items()]


__all__ = ["SCHEMA", "evaluation_tables", "lead_tables", "reliability_table", "run_record",
           "set_record", "transfer_tables", "write_tables"]
