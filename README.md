<h1 align="center">МАЯК</h1>

<p align="center"><b>Вероятностный прогноз температуры воздуха на 168 часов по одной метеостанции,
с рантаймом для слабого устройства.</b></p>

МАЯК выдаёт 7 квантилей (5, 10, 25, 50, 75, 90 и 95 %) на каждый из 168 часов вперёд по
почасовой истории одной точки, её координатам и высоте. Обучение — на реанализе ERA5,
внешний тест — на наблюдениях GHCNh, устройство — Python с `numpy` и `onnxruntime`.

- модели и их источники — [`MODELS.md`](MODELS.md);
- область применения, данные, обучение, оценка, калибровка, устройство и формулы —
  [`METHODS.md`](METHODS.md);
- JSON таблиц результатов — [`results/`](results/README.md).

## Установка

Нужны Python 3.11+ и [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/feyra-labs/tempcast.git
cd tempcast
uv sync --extra train --group dev   # PyTorch из индекса CUDA 12.8, работает и на CPU
uv run pytest -q                    # тесты на синтетике, данные не нужны
```

Команды ниже запускаются из корня репозитория. Справка: `--help` у скриптов на argparse,
`uv run python scripts/run.py --help` у обучения.

## Данные

Нужна карта зон Кёппена — Гейгера с кодами 1..30, например
`Beck_KG_V1_present_0p0083.tif` (Beck et al., 2018).

```bash
# обучающий набор: точки на суше, ERA5 из архивного API Open-Meteo, станции и манифест
uv run python scripts/make_points.py --koppen Beck_KG_V1_present_0p0083.tif --out data/points.csv
uv run python scripts/fetch_era5.py --points data/points.csv --out data     # 2016–2025, повторять до «осталось 0»
uv run python scripts/make_era5.py --data data
uv run python scripts/make_splits.py --manifest data/manifest.csv          # роли станций
uv run python scripts/build_cache.py --manifest data/manifest.csv --jobs 8 # QC и климатология

# внешний тест: наблюдения GHCNh, только почасовые станции
uv run python scripts/fetch_ghcnh.py --out data/ghcnh/raw --years 2016-2025 \
    --bbox 35 60 -10 40 --max-stations 300 --jobs 8
uv run python scripts/make_ghcnh.py --raw data/ghcnh/raw --out data/ghcnh \
    --train-manifest data/manifest.csv --koppen Beck_KG_V1_present_0p0083.tif --dem open-meteo
uv run python scripts/build_cache.py --manifest data/ghcnh/manifest.csv

# доля ложных срабатываний QC на реанализе
uv run python scripts/qc_false_alarms.py --manifest data/manifest.csv

# синтетика вместо скачивания: проверка конвейера
uv run python scripts/make_synth.py --out data
uv run python scripts/make_splits.py --manifest data/manifest.csv
uv run python scripts/build_cache.py --manifest data/manifest.csv
```

| путь | что это |
|---|---|
| `data/points.csv` | `id, lat, lon, koppen` — выбранные точки |
| `data/era5/<id>.json.gz` | ответ API по точке вместе с параметрами запроса |
| `data/fetch_meta.json` | эндпоинт, модель, переменные, период, коммит скачивания |
| `data/stations/<id>.npz` | почасовые `T`, `P`, `RH`, маска наличия `valid`, начало ряда `t0_utc_h` |
| `data/manifest.csv` | `id, lat, lon, elev, koppen, cell_lat, cell_lon, split` |
| `data/splits_report.json` | число станций по ролям, страты, раскладка времени |
| `data/cache/<ключ>/` | кэш после QC, `qc_report.csv`, `meta.json` |
| `data/ghcnh/manifest.csv` | станции внешнего теста, роль `external_test` |
| `data/ghcnh/selection_report.csv` | причина исключения каждой станции GHCNh |

Роли станций и временные блоки — [`METHODS.md`](METHODS.md), «Данные и сплиты»; источники
и лицензии данных — там же, «Источники».

## Обучение

Единственный вход — `scripts/run.py` (Hydra, конфиги в `conf/`). Одна команда на модель.
Правила подбора и перехода между этапами — [`METHODS.md`](METHODS.md), «Подбор скорости
обучения».

**1. Подбор скорости обучения и этап B** — МАЯК и бейзлайны:

```bash
# порядок: подбор и этап B всех моделей → абляции, augment=none (когда в runs/mayak/protocol.json есть запись о подборе) → калибровка → оценка
uv run python scripts/run.py run.lr_search=true
uv run python scripts/run.py -m model=gru,dlinear,lru,patchtst run.lr_search=true
```

**2. Абляции, без аугментаций** — lr основного МАЯК, этапы A и B одной командой:

```bash
uv run python scripts/run.py -m run.lr_from=runs/mayak \
    ablation=no_compression,no_passport,no_solar,no_mode_groups,no_offset_aug,no_correction,no_persistent
uv run python scripts/run.py run.lr_from=runs/mayak augment=none
```

**3. Этап 2, необязательный** — дополнительная настройка МАЯК по валидации, в таблицы
«Результаты» не входит ([`METHODS.md`](METHODS.md), «Подбор скорости обучения»):

```bash
uv run python scripts/run.py run.extra_tuning=true run.lr_search=true model.encoder_width=96
```

**Отладка на CPU:**

```bash
uv run python scripts/run.py train=debug run.accelerator=cpu run.out_root=runs/debug
uv run python scripts/run.py train=debug run.accelerator=cpu run.out_root=runs/debug \
    run.lr_search=true
```

| каталог прогона | команда |
|---|---|
| `runs/<arch>` | `model=<arch>` |
| `runs/mayak-<абляция>` | `ablation=<абляция>` |
| `runs/mayak-aug_none` | `augment=none` |

| путь в каталоге прогона | что это |
|---|---|
| `protocol.json` | журнал: протокол, сиды, запись подбора, этапы |
| `config.json` | полностью разрешённый конфиг прогона |
| `lr_search/lr<X>/stageA/` | этап A прогона сетки: `best.ckpt`, `report.json`, `report/` |
| `lr_search/lr<X>/stageB/` | короткий этап B прогона сетки |
| `stageA/` | этап A прогона с `run.lr_from` или без подбора: `best.ckpt`, `report.json`, `report/` |
| `stageB/best.ckpt` | итоговый чекпойнт |

Эталонные окна аугментаций и их действие на QC: `uv run python scripts/aug_reference.py`.

## Оценка, калибровка, робастность

```bash
# конформная таблица: runs/conformal.npy, .meta.json, .report.json
uv run python scripts/calibrate.py --ckpt runs/mayak/stageB/best.ckpt --out runs/conformal.npy
mkdir -p results/calibration
cp runs/conformal.report.json results/calibration/conformal.report.json

# таблицы 1–6, 8, 11; прогон этапа 2, если он есть, — ещё --tuned-ckpt runs/mayak-tuned/stageB/best.ckpt
uv run python -m mayak.evaluate --ckpt runs/mayak/stageB/best.ckpt \
    --gru-ckpt runs/gru/stageB/best.ckpt --dlinear-ckpt runs/dlinear/stageB/best.ckpt \
    --lru-ckpt runs/lru/stageB/best.ckpt --patchtst-ckpt runs/patchtst/stageB/best.ckpt \
    --conformal runs/conformal.npy --external-manifest data/ghcnh/manifest.csv \
    --bootstrap 1000 --results-dir results/evaluate

# таблица 7: абляции, ERA5 и GHCNh
ABLATIONS="no_compression no_passport no_solar no_mode_groups no_offset_aug no_correction no_persistent"
uv run python -m mayak.evaluate \
    --ckpt runs/mayak/stageB/best.ckpt \
    --ablation-ckpt $(for a in $ABLATIONS; do echo runs/mayak-$a/stageB/best.ckpt; done) \
    --external-manifest data/ghcnh/manifest.csv \
    --results-dir results/ablations --out-dir runs/plots/ablations

# таблица 12: профиль аугментаций none
uv run python -m mayak.evaluate --ckpt runs/mayak-aug_none/stageB/best.ckpt \
    --external-manifest data/ghcnh/manifest.csv \
    --results-dir results/augment_none --out-dir runs/plots/augment_none

# таблица 9: робастность
uv run python -m mayak.robustness --ckpt runs/mayak/stageB/best.ckpt \
    --external-manifest data/ghcnh/manifest.csv --out-dir results/robustness
```

| конфиг | что задаёт |
|---|---|
| `conf/calibration/default.yaml` | калибровка и офлайн-прогон адаптивной калибровки |
| `conf/robustness/default.yaml` | сценарии и уровни робастности |

## Результаты

Числа перенесены из JSON в [`results/`](results/README.md), прогон стенда оценки —
2026-10-16 14:59 UTC. В ячейке «пул / макро», интервалы бутстрапа — в JSON. Одиночное
значение скилла — пул. Скилл, CRPS, MAE и PICP90 — по сырым выходам моделей при полной
истории 672 ч; конформная поправка — только в таблице 6. Наборы и метрики —
[`METHODS.md`](METHODS.md), «Оценка». Прочерк — прогона ещё нет или числа нет в JSON.

**1. Метрики по лидам, новые станции** — `results/evaluate/internal/metrics.json`

Скилл:

| модель | 1 ч | 6 ч | 24 ч | 72 ч | 168 ч |
|---|:-:|:-:|:-:|:-:|:-:|
| МАЯК | 0,93 / 0,87 | 0,81 / 0,72 | 0,48 / 0,44 | 0,12 / 0,10 | −0,05 / −0,05 |
| LRU | 0,82 / 0,69 | 0,72 / 0,54 | 0,41 / 0,30 | 0,06 / −0,03 | −0,10 / −0,20 |
| GRU seq2seq | 0,90 / 0,80 | 0,80 / 0,68 | 0,46 / 0,39 | 0,07 / 0,01 | −0,10 / −0,17 |
| DLinear | 0,92 / 0,86 | 0,66 / 0,56 | 0,32 / 0,28 | −0,15 / −0,16 | −0,54 / −0,50 |
| PatchTST | 0,93 / 0,88 | 0,75 / 0,67 | 0,40 / 0,38 | −0,05 / −0,04 | −0,41 / −0,34 |
| Damped persistence | 0,70 / 0,61 | 0,60 / 0,54 | 0,32 / 0,28 | 0,08 / 0,07 | 0,00 / 0,00 |
| Seasonal-naive 24ч | 0,36 / 0,30 | 0,36 / 0,30 | 0,33 / 0,26 | −0,35 / −0,37 | −0,88 / −0,86 |
| Климатология | 0,00 / 0,00 | 0,00 / 0,00 | 0,00 / 0,00 | 0,00 / 0,00 | 0,00 / 0,00 |

<details>
<summary>MAE и CRPS (°C), PICP90 по лидам, ч — тот же файл</summary>

| модель | MAE 1 | MAE 6 | MAE 24 | MAE 72 | MAE 168 | CRPS 1 | CRPS 6 | CRPS 24 | CRPS 72 | CRPS 168 | PICP90 1 | PICP90 6 | PICP90 24 | PICP90 72 | PICP90 168 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| МАЯК | 0,82 / 0,82 | 1,27 / 1,27 | 2,03 / 2,03 | 2,69 / 2,69 | 2,94 / 2,94 | 0,48 / 0,48 | 0,73 / 0,73 | 1,17 / 1,17 | 1,54 / 1,54 | 1,66 / 1,66 | 0,935 / 0,935 | 0,895 / 0,895 | 0,897 / 0,897 | 0,886 / 0,886 | 0,889 / 0,889 |
| LRU | 1,30 / 1,30 | 1,58 / 1,58 | 2,22 / 2,22 | 2,84 / 2,84 | 3,06 / 3,06 | 0,78 / 0,78 | 0,92 / 0,92 | 1,29 / 1,29 | 1,62 / 1,62 | 1,74 / 1,74 | 0,947 / 0,947 | 0,925 / 0,925 | 0,903 / 0,903 | 0,888 / 0,888 | 0,884 / 0,884 |
| GRU seq2seq | 0,98 / 0,98 | 1,33 / 1,33 | 2,09 / 2,09 | 2,79 / 2,79 | 3,03 / 3,03 | 0,58 / 0,58 | 0,77 / 0,77 | 1,21 / 1,21 | 1,59 / 1,59 | 1,72 / 1,72 | 0,933 / 0,933 | 0,908 / 0,908 | 0,882 / 0,882 | 0,879 / 0,879 | 0,878 / 0,878 |
| DLinear | 0,86 / 0,86 | 1,66 / 1,66 | 2,31 / 2,31 | 3,07 / 3,07 | 3,51 / 3,51 | 0,66 / 0,66 | 1,11 / 1,11 | 1,50 / 1,50 | 1,97 / 1,97 | 2,26 / 2,26 | 0,992 / 0,992 | 0,954 / 0,954 | 0,920 / 0,920 | 0,906 / 0,906 | 0,893 / 0,893 |
| PatchTST | 0,82 / 0,82 | 1,43 / 1,43 | 2,19 / 2,19 | 2,92 / 2,92 | 3,35 / 3,35 | 0,52 / 0,52 | 0,83 / 0,83 | 1,28 / 1,28 | 1,71 / 1,71 | 1,97 / 1,97 | 0,968 / 0,968 | 0,935 / 0,935 | 0,894 / 0,894 | 0,875 / 0,875 | 0,873 / 0,873 |
| Damped persistence | 1,61 / 1,61 | 1,78 / 1,78 | 2,35 / 2,35 | 2,77 / 2,77 | 2,87 / 2,87 | 1,00 / 1,00 | 1,05 / 1,05 | 1,35 / 1,35 | 1,57 / 1,57 | 1,62 / 1,62 | 0,709 / 0,709 | 0,808 / 0,808 | 0,882 / 0,882 | 0,903 / 0,903 | 0,905 / 0,905 |
| Seasonal-naive 24ч | 2,21 / 2,21 | 2,12 / 2,12 | 2,23 / 2,23 | 3,27 / 3,27 | 3,84 / 3,84 | 1,36 / 1,36 | 1,32 / 1,32 | 1,38 / 1,38 | 1,92 / 1,92 | 2,29 / 2,29 | 0,947 / 0,947 | 0,947 / 0,947 | 0,940 / 0,940 | 0,853 / 0,853 | 0,794 / 0,794 |
| Климатология | 2,89 / 2,89 | 2,80 / 2,80 | 2,87 / 2,87 | 2,91 / 2,91 | 2,88 / 2,88 | 1,63 / 1,63 | 1,58 / 1,58 | 1,62 / 1,62 | 1,64 / 1,64 | 1,63 / 1,63 | 0,905 / 0,905 | 0,903 / 0,903 | 0,909 / 0,909 | 0,907 / 0,907 | 0,908 / 0,908 |

</details>

<details>
<summary>2. Скилл по длине истории L, ч — <code>results/evaluate/internal/history.json</code></summary>

В ячейке «лид 24 / 72 / 168 ч», пул.

| модель | L=0 | 6 | 24 | 72 | 168 | 336 | 672 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| МАЯК | −0,57 / −0,52 / −0,56 | 0,45 / 0,05 / −0,15 | 0,46 / 0,06 / −0,12 | 0,46 / 0,08 / −0,09 | 0,47 / 0,10 / −0,06 | 0,48 / 0,11 / −0,05 | 0,48 / 0,12 / −0,05 |
| LRU | −0,26 / −0,23 / −0,25 | 0,34 / 0,00 / −0,15 | 0,38 / 0,02 / −0,14 | 0,41 / 0,04 / −0,12 | 0,42 / 0,05 / −0,11 | 0,42 / 0,06 / −0,10 | 0,41 / 0,06 / −0,10 |
| GRU seq2seq | −0,38 / −0,37 / −0,37 | 0,40 / −0,02 / −0,22 | 0,42 / 0,00 / −0,19 | 0,45 / 0,03 / −0,16 | 0,45 / 0,05 / −0,13 | 0,46 / 0,06 / −0,12 | 0,46 / 0,07 / −0,10 |
| DLinear | −29,34 / −28,15 / −28,18 | 0,21 / −0,47 / −1,03 | 0,18 / −0,49 / −1,02 | 0,28 / −0,30 / −0,74 | 0,32 / −0,19 / −0,57 | 0,32 / −0,15 / −0,54 | 0,32 / −0,15 / −0,54 |
| PatchTST | −22,57 / −21,84 / −22,20 | 0,32 / −0,36 / −0,90 | 0,33 / −0,28 / −0,78 | 0,35 / −0,22 / −0,62 | 0,39 / −0,11 / −0,46 | 0,40 / −0,06 / −0,41 | 0,40 / −0,05 / −0,41 |
| Damped persistence | 0,00 / 0,00 / 0,00 | 0,41 / 0,11 / 0,01 | 0,32 / 0,08 / 0,00 | 0,32 / 0,08 / 0,00 | 0,32 / 0,08 / 0,00 | 0,32 / 0,08 / 0,00 | 0,32 / 0,08 / 0,00 |
| Seasonal-naive 24ч | 0,00 / 0,00 / 0,00 | 0,34 / −0,34 / −0,88 | 0,34 / −0,34 / −0,88 | 0,33 / −0,35 / −0,88 | 0,33 / −0,35 / −0,88 | 0,33 / −0,35 / −0,88 | 0,33 / −0,35 / −0,88 |
| Климатология | 0,00 / 0,00 / 0,00 | 0,00 / 0,00 / 0,00 | 0,00 / 0,00 / 0,00 | 0,00 / 0,00 / 0,00 | 0,00 / 0,00 / 0,00 | 0,00 / 0,00 / 0,00 | 0,00 / 0,00 / 0,00 |

</details>

<details>
<summary>3–4. Роли станций, сезоны, зоны Кёппена, лид 24 ч, МАЯК — <code>results/evaluate/internal/breakdowns.json</code>, <code>results/evaluate/train_stations/metrics.json</code></summary>

| срез | Skill пул | Skill макро | MAE | CRPS | PICP90 | станций |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| обучающие (train) | 0,49 | 0,43 | 1,97 | 1,14 | 0,893 | 260 |
| новые (unseen_test) | 0,48 | 0,44 | 2,03 | 1,17 | 0,897 | 51 |
| зима | 0,49 | 0,45 | 2,36 | 1,37 | 0,897 | 51 |
| весна | 0,47 | 0,42 | 2,21 | 1,28 | 0,895 | 51 |
| лето | 0,39 | 0,32 | 1,62 | 0,93 | 0,908 | 51 |
| осень | 0,53 | 0,44 | 1,93 | 1,12 | 0,890 | 51 |
| зона Af | 0,07 | 0,19 | 1,00 | 0,59 | 0,906 | 2 |
| зона Aw | 0,42 | 0,42 | 1,09 | 0,63 | 0,893 | 6 |
| зона BSh | 0,57 | 0,50 | 1,24 | 0,72 | 0,923 | 3 |
| зона BSk | 0,55 | 0,56 | 2,16 | 1,24 | 0,900 | 3 |
| зона BWh | 0,51 | 0,51 | 1,38 | 0,79 | 0,937 | 6 |
| зона BWk | 0,52 | 0,53 | 2,26 | 1,29 | 0,915 | 2 |
| зона Cfa | 0,36 | 0,35 | 2,14 | 1,21 | 0,889 | 2 |
| зона Cwa | 0,35 | 0,35 | 1,27 | 0,75 | 0,872 | 2 |
| зона Dfb | 0,58 | 0,58 | 2,77 | 1,56 | 0,927 | 2 |
| зона Dfc | 0,52 | 0,51 | 3,37 | 1,97 | 0,875 | 5 |
| зона ET | 0,44 | 0,44 | 2,65 | 1,55 | 0,885 | 2 |

</details>

<details>
<summary>5. Надёжность МАЯК — <code>results/evaluate/internal/reliability.json</code></summary>

| PIT | <q05 | q05–q10 | q10–q25 | q25–q50 | q50–q75 | q75–q90 | q90–q95 | >q95 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| МАЯК, весь горизонт | 0,051 | 0,051 | 0,148 | 0,244 | 0,245 | 0,151 | 0,051 | 0,058 |

| P(факт ≤ квантиль) | 5% | 10% | 25% | 50% | 75% | 90% | 95% |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| МАЯК | 0,051 | 0,102 | 0,250 | 0,494 | 0,739 | 0,890 | 0,942 |

Острота против покрытия, весь горизонт:

| модель | фактическое покрытие 90% интервала | ширина, °C | ширина для фактических 90 %, °C |
|---|:-:|:-:|:-:|
| МАЯК | 0,891 | 10,08 | 10,39 |
| LRU | 0,892 | 10,86 | 11,15 |
| GRU seq2seq | 0,885 | 10,23 | 10,72 |
| DLinear | 0,909 | 15,73 | 15,05 |
| PatchTST | 0,885 | 11,28 | 11,87 |
| Damped persistence | 0,889 | 10,74 | 11,13 |
| Seasonal-naive 24ч | 0,845 | 11,83 | 14,01 |
| Климатология | 0,903 | 11,83 | 11,74 |

</details>

<details>
<summary>6. МАЯК после калибровки — <code>results/evaluate/internal/calibrated.json</code>, <code>results/evaluate/internal/reliability.json</code></summary>

По бинам лидов, полная история 672 ч (`calibrated.json`, ключ `effect`):

| бин лидов | PICP90 до | PICP90 после | ширина 90 до, °C | ширина 90 после, °C | CRPS до | CRPS после |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1–6 ч | 0,908 | 0,893 | 4,68 | 4,50 | 0,626 | 0,628 |
| 7–24 ч | 0,898 | 0,900 | 7,16 | 7,23 | 0,996 | 0,997 |
| 25–72 ч | 0,892 | 0,895 | 9,79 | 9,87 | 1,391 | 1,391 |
| 73–168 ч | 0,888 | 0,896 | 11,11 | 11,30 | 1,599 | 1,598 |

По запрошенной длине истории, весь горизонт: «до» — `reliability.json` (`coverage.report`,
разрез «длина истории»), «после» — `calibrated.json` (`report`, тот же разрез).

| длина истории | PICP90 до | PICP90 после | ширина 90 до, °C | ширина 90 после, °C |
|---|:-:|:-:|:-:|:-:|
| 0 ч | 0,864 | 0,869 | 12,74 | 12,87 |
| 6 ч | 0,887 | 0,899 | 11,02 | 11,37 |
| 24 ч | 0,890 | 0,901 | 10,59 | 10,94 |
| 72 ч | 0,890 | 0,891 | 10,36 | 10,39 |
| 168 ч | 0,888 | 0,889 | 10,15 | 10,17 |
| 336 ч | 0,889 | 0,895 | 10,07 | 10,20 |
| 672 ч | 0,891 | 0,896 | 10,08 | 10,21 |

Офлайн-прогон адаптивной калибровки устройства (`calibrated.json`, ключ `aci`; 8 станций,
5 760 ежечасных выпусков):

| бин лидов | PICP90 без ACI | PICP90 с ACI | ширина 90 без, °C | ширина 90 с, °C | обратных связей | θ в конце, медиана |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1–6 ч | 0,892 | 0,894 | 5,31 | 5,34 | 5 760 | 0,020 |
| 7–24 ч | 0,887 | 0,885 | 8,86 | 8,75 | 5 760 | −0,025 |
| 25–72 ч | 0,889 | 0,896 | 12,69 | 12,77 | 5 760 | 0,023 |
| 73–168 ч | 0,899 | 0,901 | 14,48 | 14,49 | 5 760 | −0,010 |
| весь поток | 0,895 | 0,898 | 13,04 | 13,06 | 23 040 | — |

Обновлений, упёршихся в границу множителя: 0.

Подгонка таблицы на калибровочных окнах, в выборке —
`results/calibration/conformal.report.json`:

| бин валидных часов истории | окон | строка таблицы | PICP90 до | PICP90 после |
|---|:-:|:-:|:-:|:-:|
| L=0 | 179 | маргинальная | 0,898 | 0,902 |
| 1–24 ч | 354 | своя | 0,883 | 0,900 |
| 25–168 ч | 1 048 | своя | 0,899 | 0,900 |
| 169–672 ч | 3 019 | своя | 0,895 | 0,900 |

Окна и строка таблицы — по валидным часам (`record.history_fit`), покрытие — по
запрошенной длине истории (`report`), поэтому в бинах 1–24 и 25–168 ч наборы окон
расходятся на пару окон. Весь набор: PICP90 0,895 до и 0,900 после, окон 4 600, станций
`unseen_val` 32.

</details>

<details>
<summary>7. Абляции — <code>results/ablations/{internal,external}/metrics.json</code></summary>

Абляции сравниваются с основным МАЯК — [`METHODS.md`](METHODS.md), раздел «Абляции».

ERA5, станции `unseen_test` (`internal/metrics.json`):

| вариант | Skill@24 | Skill@72 | Skill@168 | CRPS@24 | PICP90@24 |
|---|:-:|:-:|:-:|:-:|:-:|
| МАЯК (эталон абляций, `none`) | — | — | — | — | — |
| МАЯК [mayak-no_compression] | — | — | — | — | — |
| МАЯК [mayak-no_passport] | — | — | — | — | — |
| МАЯК [mayak-no_solar] | — | — | — | — | — |
| МАЯК [mayak-no_mode_groups] | — | — | — | — | — |
| МАЯК [mayak-no_offset_aug] | — | — | — | — | — |
| МАЯК [mayak-no_correction] | — | — | — | — | — |
| МАЯК [mayak-no_persistent] | — | — | — | — | — |

GHCNh, станции `external_test` (`external/metrics.json`):

| вариант | Skill@24 | Skill@72 | Skill@168 | CRPS@24 | PICP90@24 |
|---|:-:|:-:|:-:|:-:|:-:|
| МАЯК (эталон абляций, `none`) | — | — | — | — | — |
| МАЯК [mayak-no_compression] | — | — | — | — | — |
| МАЯК [mayak-no_passport] | — | — | — | — | — |
| МАЯК [mayak-no_solar] | — | — | — | — | — |
| МАЯК [mayak-no_mode_groups] | — | — | — | — | — |
| МАЯК [mayak-no_offset_aug] | — | — | — | — | — |
| МАЯК [mayak-no_correction] | — | — | — | — | — |
| МАЯК [mayak-no_persistent] | — | — | — | — | — |

</details>

<details>
<summary>8. Внешний тест GHCNh — <code>results/evaluate/external/*.json</code></summary>

Скилл по лидам, «пул / макро» (`metrics.json`):

| модель | 1 ч | 6 ч | 24 ч | 72 ч | 168 ч |
|---|:-:|:-:|:-:|:-:|:-:|
| МАЯК | 0,90 / 0,89 | 0,67 / 0,67 | 0,48 / 0,47 | 0,09 / 0,09 | −0,06 / −0,06 |
| LRU | 0,67 / 0,63 | 0,48 / 0,45 | 0,20 / 0,13 | −0,18 / −0,26 | −0,36 / −0,44 |
| GRU seq2seq | 0,88 / 0,87 | 0,66 / 0,65 | 0,46 / 0,45 | 0,07 / 0,05 | −0,10 / −0,12 |
| DLinear | 0,90 / 0,90 | 0,55 / 0,53 | 0,38 / 0,37 | −0,10 / −0,10 | −0,38 / −0,38 |
| PatchTST | 0,90 / 0,89 | 0,61 / 0,61 | 0,41 / 0,40 | −0,06 / −0,06 | −0,34 / −0,34 |
| Damped persistence | 0,66 / 0,66 | 0,54 / 0,53 | 0,26 / 0,27 | 0,06 / 0,06 | 0,00 / 0,00 |
| Seasonal-naive 24ч | 0,30 / 0,29 | 0,16 / 0,15 | 0,37 / 0,33 | −0,36 / −0,37 | −0,90 / −0,90 |
| Климатология | 0,00 / 0,00 | 0,00 / 0,00 | 0,00 / 0,00 | 0,00 / 0,00 | 0,00 / 0,00 |

Разрезы, лид 24 ч, МАЯК (`breakdowns.json`):

| разрез | Skill@24 | MAE@24 | CRPS@24 | PICP90@24 | станций |
|---|:-:|:-:|:-:|:-:|:-:|
| валидных часов истории <50% | — | — | — | — | — |
| 50–80% | 0,31 | 2,15 | 1,24 | 0,849 | 12 |
| 80–95% | 0,39 | 2,23 | 1,28 | 0,892 | 54 |
| 95–100% | 0,50 | 2,13 | 1,23 | 0,908 | 65 |
| до обучающей точки <25 км | — | — | — | — | — |
| 25–100 км | 0,48 | 2,21 | 1,25 | 0,875 | 4 |
| 100–300 км | 0,49 | 2,24 | 1,28 | 0,911 | 21 |
| ≥300 км | 0,48 | 2,10 | 1,21 | 0,901 | 43 |
| давление есть | 0,50 | 2,12 | 1,22 | 0,901 | 45 |
| давления нет | 0,46 | 2,20 | 1,26 | 0,906 | 30 |
| Δ высоты станция−ЦМР <50 м | 0,48 | 2,14 | 1,23 | 0,904 | 59 |
| Δ высоты 50–150 м | 0,51 | 2,15 | 1,23 | 0,906 | 5 |
| Δ высоты 150–300 м | 0,49 | 2,28 | 1,31 | 0,870 | 4 |

Внутренний против внешнего на общих группах зон B, C, D (`transfer.json`, Δ = внешн. −
внутр.):

| модель | Skill@24 внутр. | Skill@24 внешн. | Δ | валидных часов истории внутр. | валидных часов истории внешн. |
|---|:-:|:-:|:-:|:-:|:-:|
| МАЯК | 0,49 | 0,49 | −0,001 | — | — |
| Климатология | 0,00 | 0,00 | 0,000 | — | — |

</details>

<details>
<summary>9. Робастность — <code>results/robustness/robustness.json</code></summary>

Уровни сценариев — `conf/robustness/default.yaml`, описание — [`METHODS.md`](METHODS.md),
раздел «Робастность».

| сценарий | наихудший Skill@24 | наихудший Skill@168 | проверка скилла |
|---|:-:|:-:|:-:|
| dropout | — | — | — |
| gap | — | — | — |
| noise | — | — | — |
| spikes | — | — | — |
| freeze | — | — | — |
| drop_channel | — | — | — |
| history | — | — | — |
| coords | — | — | — |
| elev | — | — | — |
| offset | — | — | — |
| offset_input | — | — | — |
| drift | — | — | — |
| drift_input | — | — | — |
| scale | — | — | — |

</details>

<details>
<summary>10. Устройство — <code>results/device/bench.json</code></summary>

```bash
uv run python scripts/bench_device.py --ckpt runs/mayak/stageB/best.ckpt \
    --conformal runs/conformal.npy --lat 52.37 --lon 4.90 --elev -2 --out-dir runs/bench_device
mkdir -p results/device
cp runs/bench_device/results.json results/device/bench.json
```

| реализация | час p50, мкс | час p95, мкс | час p99, мкс | выпуск p50, мкс | выпуск p95, мкс | выпуск p99, мкс | пиковая память, Б | расхождение с эталоном, °C |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Python + ONNX Runtime | — | — | — | — | — | — | — | — |
| PyTorch (эталон) | — | — | — | — | — | — | — | — |

| величина | значение |
|---|:-:|
| графы экспорта, Б | — |
| состояние на диске, Б | — |
| рабочая память устройства, Б | — |
| расхождение выпуска устройства с окном оценки, °C | — |

</details>

**11. Число параметров** — `results/evaluate/params.json`

| модель | параметров | доля от МАЯК |
|---|:-:|:-:|
| МАЯК | 110 214 | 1,00 |
| LRU | 103 986 | 0,94 |
| GRU seq2seq | 120 754 | 1,10 |
| DLinear | 114 408 | 1,04 |
| PatchTST | 138 944 | 1,26 |

**12. Профили аугментаций, внешний тест** — `results/evaluate/external/metrics.json`,
`results/augment_none/external/metrics.json`

| профиль | Skill@24 | Skill@72 | Skill@168 | CRPS@24 | PICP90@24 |
|---|:-:|:-:|:-:|:-:|:-:|
| МАЯК, `default` | 0,48 | 0,09 | −0,06 | 1,23 | 0,903 |
| МАЯК, `none` | — | — | — | — | — |

## Устройство

Нужны 64-битная ОС (aarch64 или x86_64, например Raspberry Pi 3/4/5 с 64-битной Raspberry
Pi OS) и Python 3.11+. Работа устройства — [`METHODS.md`](METHODS.md), «Устройство» и
«Калибровка».

**1. Экспорт** (на рабочей машине): два графа ONNX, манифест и конформная таблица.

```bash
uv run python scripts/export_runtime.py --ckpt runs/mayak/stageB/best.ckpt \
    --conformal runs/conformal.npy --aci --out runtime/model
```

**2. Установка** колеса в виртуальное окружение на устройстве, только `numpy` и `onnxruntime`.

```bash
ssh pi@device 'sudo mkdir -p /opt/mayak /var/lib/mayak && sudo chown $USER /opt/mayak /var/lib/mayak'
ssh pi@device 'python3 -m venv /opt/mayak/venv'

# колесо с рабочей машины
uv build --wheel                                     # dist/mayak-0.1.0-py3-none-any.whl
scp dist/mayak-0.1.0-py3-none-any.whl pi@device:/opt/mayak/
ssh pi@device '/opt/mayak/venv/bin/pip install /opt/mayak/mayak-0.1.0-py3-none-any.whl'

scp -r runtime/model pi@device:/opt/mayak/model
ssh pi@device '/opt/mayak/venv/bin/python -m mayak.runtime.run_inference --help'
```

**3. Запуск.** Команды — со stdin по одной на строку, ответ — строка JSON. `--lat`,
`--lon`, `--elev` обязательны.

```bash
printf 'obs 1767225600 11 1012.4 81\nforecast\nstatus\n' | \
    /opt/mayak/venv/bin/python -m mayak.runtime.run_inference --model /opt/mayak/model \
    --lat 52.37 --lon 4.90 --elev -2 --state-dir /var/lib/mayak --aci

# на рабочей машине после экспорта
uv run python -m mayak.runtime.run_inference --model runtime/model \
    --lat 52.37 --lon 4.90 --elev -2
```

| команда | ответ |
|---|---|
| `obs <секунды UTC> <T> <P> <RH>` | `{"ok": true, "codes": [...]}` — коды QC часа; значение — число, `-` или `nan` |
| `forecast [<секунды UTC>]` | `after_unix_hour`, `fallback`, `theta`, медиана `mu` и квантили `q` по 168 лидам |
| `status` | сводка, поля ниже |
| ошибка любой команды | `{"error": "..."}`, хост продолжает работу |

Выбор строки конформной таблицы, состояние и память устройства — [`METHODS.md`](METHODS.md),
«Устройство».

| поле `status` | что это |
|---|---|
| `filled` | часов в окне после холодного старта |
| `valid_hours` | часов с валидной температурой во входе модели |
| `theta`, `aci_lead_bins` | θ по бинам лидов и сами бины, ч |
| `aci_updates`, `aci_misses` | обратных связей и промахов по бинам лидов с последнего сброса |
| `conformal` | применяется ли конформная таблица |
| `state_bytes` | размер состояния на диске, Б |
| `memory_bytes` | рабочая память устройства, Б |
| `site`, `loaded_site`, `site_change` | точка прибора, точка загруженного состояния и исход сравнения: `same`, `refined`, `moved` |
| `idle_hours`, `fallbacks`, `last_unix_hour` | простой, откаты к климатологии, последний час |
| `rss_bytes`, `peak_rss_bytes` | память процесса |

**4. Служба systemd**, датчик пишет строки в именованный канал:

```ini
# /etc/systemd/system/mayak.service
[Unit]
Description=MAYAK forecast host

[Service]
User=pi
RuntimeDirectory=mayak
ExecStartPre=/usr/bin/mkfifo -m 600 /run/mayak/in
ExecStart=/bin/sh -c 'exec /opt/mayak/venv/bin/python -m mayak.runtime.run_inference \
    --model /opt/mayak/model --lat 52.37 --lon 4.90 --elev -2 --state-dir /var/lib/mayak \
    --aci 0<>/run/mayak/in >>/var/lib/mayak/out.jsonl'
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now mayak
echo "obs $(date -u +%s -d "$(date -u +%Y-%m-%dT%H:00:00)") 11 1012.4 81" > /run/mayak/in
echo forecast > /run/mayak/in && tail -n 1 /var/lib/mayak/out.jsonl
```

## Скрипты

| скрипт | назначение |
|---|---|
| `scripts/make_points.py` | выбор обучающих точек на суше по решётке Фибоначчи с покрытием зон Кёппена |
| `scripts/fetch_era5.py` | скачивание рядов ERA5 для точек из архивного API Open-Meteo |
| `scripts/make_era5.py` | сборка станций обучающего набора и манифеста из скачанных рядов |
| `scripts/make_synth.py` | синтетический набор для проверки конвейера без скачивания |
| `scripts/make_splits.py` | стратифицированное разбиение станций на роли |
| `scripts/build_cache.py` | кэш: QC, станционные проверки, отбор, климатология |
| `scripts/fetch_ghcnh.py` | скачивание наблюдений GHCNh для внешнего теста |
| `scripts/make_ghcnh.py` | сборка набора внешнего теста из наблюдений GHCNh |
| `scripts/qc_false_alarms.py` | доля ложных срабатываний QC на реанализе |
| `scripts/make_qc_golden.py` | перегенерация регрессионного вектора QC, только при смене правил |
| `scripts/aug_reference.py` | эталонные окна аугментаций и их действие на QC |
| `scripts/run.py` | обучение, единственный вход: подбор lr, этапы, абляции, профили аугментаций, этап 2 |
| `scripts/calibrate.py` | подгонка конформной таблицы МАЯК |
| `scripts/export_runtime.py` | экспорт графов, манифеста и таблицы для устройства |
| `scripts/bench_device.py` | замеры устройства: час, выпуск, память, расхождение с оценкой |

| модуль | назначение |
|---|---|
| `python -m mayak.evaluate` | стенд оценки: таблицы 1–8, 11, 12 |
| `python -m mayak.robustness` | сценарии робастности обученной модели |
| `python -m mayak.runtime.run_inference` | хост устройства |

| конфиг | назначение |
|---|---|
| `conf/config.yaml` | корневой конфиг Hydra: секция `run` (подбор, `lr_from`, этап 2), каталоги прогонов |
| `conf/model/` | архитектуры: `mayak`, `gru`, `dlinear`, `lru`, `patchtst` |
| `conf/ablation/` | `none` и семь абляций МАЯК |
| `conf/augment/` | профили аугментаций `default` и `none` |
| `conf/data/` | данные, раскладка времени, набор валидации, параметры аугментаций |
| `conf/train/` | протокол обучения: `default`, `debug` — отладка на CPU |
| `conf/calibration/`, `conf/robustness/`, `conf/runtime/` | калибровка, робастность, пороги смены точки устройства |

## Структура

```
mayak/           модель (modules/), бейзлайны (baselines/), данные и QC (data/), протокол
                 обучения, подбор lr, метрики, оценка, калибровка, робастность, экспорт
mayak/runtime/   устройство: часовой цикл, выпуск, состояние, хост, исполнитель ONNX
scripts/         данные, обучение, калибровка, экспорт, замеры
conf/            конфиги Hydra, калибровки, робастности, рантайма
tests/           тесты на синтетике и регрессионный вектор QC
results/         JSON таблиц результатов
```

## Лицензия

MIT — см. [`LICENSE`](LICENSE). Тесты и документация к коду написаны с помощью LLM.
