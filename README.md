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
uv run python scripts/fetch_era5.py --points data/points.csv --out data     # 2016–2025, докачка
uv run python scripts/make_era5.py --data data
uv run python scripts/make_splits.py --manifest data/manifest.csv          # роли станций
uv run python scripts/build_cache.py --manifest data/manifest.csv --jobs 8 # QC и климатология

# внешний тест: наблюдения GHCNh, только почасовые станции
uv run python scripts/fetch_ghcnh.py --out data/ghcnh/raw --years 2016-2025 \
    --bbox 35 60 -10 40 --max-stations 300 --jobs 8
uv run python scripts/make_ghcnh.py --raw data/ghcnh/raw --out data/ghcnh \
    --koppen Beck_KG_V1_present_0p0083.tif --dem open-meteo
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
uv run python scripts/run.py run.lr_search=true
uv run python scripts/run.py -m model=gru,dlinear,lru,patchtst run.lr_search=true
```

**2. Абляции, сиды, без аугментаций** — lr основного МАЯК, этапы A и B одной командой:

```bash
uv run python scripts/run.py -m run.lr_from=runs/mayak \
    ablation=no_compression,no_passport,no_solar,no_mode_groups,no_offset_aug,no_correction,no_persistent
uv run python scripts/run.py -m run.lr_from=runs/mayak train.seed=1,2
uv run python scripts/run.py run.lr_from=runs/mayak augment=none
```

**3. Этап 2, необязательный:**

```bash
uv run python scripts/run.py run.extra_tuning=true run.lr_search=true model.encoder_width=64
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
| `runs/mayak-s<сид>` | `train.seed=<сид>` |
| `runs/mayak-aug_none` | `augment=none` |
| `runs/mayak-tuned` | `run.extra_tuning=true` |

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
cp runs/conformal.report.json results/calibration/conformal.report.json

# таблицы 1–6, 8, 11; строка этапа 2 — с --tuned-ckpt runs/mayak-tuned/stageB/best.ckpt
uv run python -m mayak.evaluate --ckpt runs/mayak/stageB/best.ckpt \
    --gru-ckpt runs/gru/stageB/best.ckpt --dlinear-ckpt runs/dlinear/stageB/best.ckpt \
    --lru-ckpt runs/lru/stageB/best.ckpt --patchtst-ckpt runs/patchtst/stageB/best.ckpt \
    --conformal runs/conformal.npy --external-manifest data/ghcnh/manifest.csv \
    --bootstrap 1000 --results-dir results/evaluate

# таблица 7: абляции и сиды, ERA5 и GHCNh, значимость абляций
ABLATIONS="no_compression no_passport no_solar no_mode_groups no_offset_aug no_correction no_persistent"
uv run python -m mayak.evaluate \
    --ckpt runs/mayak/stageB/best.ckpt \
           runs/mayak-s1/stageB/best.ckpt runs/mayak-s2/stageB/best.ckpt \
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

Чисел пока нет: таблицы заполняются из JSON в [`results/`](results/README.md). В ячейке
«пул / макро», интервалы бутстрапа — в JSON. Наборы, метрики и строка «МАЯК (доп.
настройка)†» — [`METHODS.md`](METHODS.md), «Оценка».

**1. Метрики по лидам, новые станции** — `results/evaluate/internal/metrics.json`

Скилл:

| модель | 1 ч | 6 ч | 24 ч | 72 ч | 168 ч |
|---|:-:|:-:|:-:|:-:|:-:|
| МАЯК | — / — | — / — | — / — | — / — | — / — |
| МАЯК (доп. настройка)† | — / — | — / — | — / — | — / — | — / — |
| LRU | — / — | — / — | — / — | — / — | — / — |
| GRU seq2seq | — / — | — / — | — / — | — / — | — / — |
| DLinear | — / — | — / — | — / — | — / — | — / — |
| PatchTST | — / — | — / — | — / — | — / — | — / — |
| Damped persistence | — / — | — / — | — / — | — / — | — / — |
| Seasonal-naive 24ч | — / — | — / — | — / — | — / — | — / — |
| Климатология | — / — | — / — | — / — | — / — | — / — |

<details>
<summary>MAE и CRPS (°C), PICP90 по лидам, ч — тот же файл</summary>

| модель | MAE 1 | MAE 6 | MAE 24 | MAE 72 | MAE 168 | CRPS 1 | CRPS 6 | CRPS 24 | CRPS 72 | CRPS 168 | PICP90 1 | PICP90 6 | PICP90 24 | PICP90 72 | PICP90 168 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| МАЯК | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — |
| LRU | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — |
| GRU seq2seq | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — |
| DLinear | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — |
| PatchTST | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — |
| Damped persistence | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — |
| Seasonal-naive 24ч | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — |
| Климатология | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — |

</details>

<details>
<summary>2. Скилл по длине истории L, ч — <code>results/evaluate/internal/history.json</code></summary>

В ячейке «лид 24 / 72 / 168 ч», пул.

| модель | L=0 | 6 | 24 | 72 | 168 | 336 | 672 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| МАЯК | — / — / — | — / — / — | — / — / — | — / — / — | — / — / — | — / — / — | — / — / — |
| LRU | — / — / — | — / — / — | — / — / — | — / — / — | — / — / — | — / — / — | — / — / — |
| GRU seq2seq | — / — / — | — / — / — | — / — / — | — / — / — | — / — / — | — / — / — | — / — / — |
| DLinear | — / — / — | — / — / — | — / — / — | — / — / — | — / — / — | — / — / — | — / — / — |
| PatchTST | — / — / — | — / — / — | — / — / — | — / — / — | — / — / — | — / — / — | — / — / — |
| Damped persistence | — / — / — | — / — / — | — / — / — | — / — / — | — / — / — | — / — / — | — / — / — |
| Seasonal-naive 24ч | — / — / — | — / — / — | — / — / — | — / — / — | — / — / — | — / — / — | — / — / — |
| Климатология | — / — / — | — / — / — | — / — / — | — / — / — | — / — / — | — / — / — | — / — / — |

</details>

<details>
<summary>3–4. Роли станций, сезоны, зоны Кёппена, лид 24 ч, МАЯК — <code>results/evaluate/internal/breakdowns.json</code>, <code>results/evaluate/train_stations/metrics.json</code></summary>

| срез | Skill пул | Skill макро | MAE | CRPS | PICP90 | станций |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| обучающие (train) | — | — | — | — | — | — |
| новые (unseen_test) | — | — | — | — | — | — |
| зима | — | — | — | — | — | — |
| весна | — | — | — | — | — | — |
| лето | — | — | — | — | — | — |
| осень | — | — | — | — | — | — |
| каждая зона из JSON | — | — | — | — | — | — |

</details>

<details>
<summary>5. Надёжность МАЯК — <code>results/evaluate/internal/reliability.json</code></summary>

| PIT | <q05 | q05–q10 | q10–q25 | q25–q50 | q50–q75 | q75–q90 | q90–q95 | >q95 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| МАЯК, весь горизонт | — | — | — | — | — | — | — | — |

| P(факт ≤ квантиль) | 5% | 10% | 25% | 50% | 75% | 90% | 95% |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| МАЯК | — | — | — | — | — | — | — |

Острота против покрытия, весь горизонт:

| модель | фактическое покрытие 90% интервала | ширина, °C | ширина для фактических 90 %, °C |
|---|:-:|:-:|:-:|
| МАЯК | — | — | — |
| LRU | — | — | — |
| GRU seq2seq | — | — | — |
| DLinear | — | — | — |
| PatchTST | — | — | — |
| Damped persistence | — | — | — |
| Seasonal-naive 24ч | — | — | — |
| Климатология | — | — | — |

</details>

<details>
<summary>6. МАЯК после калибровки — <code>results/evaluate/internal/calibrated.json</code>, <code>results/evaluate/internal/reliability.json</code></summary>

По бинам лидов, полная история 672 ч (`calibrated.json`, ключ `effect`):

| бин лидов | PICP90 до | PICP90 после | ширина 90 до, °C | ширина 90 после, °C | CRPS до | CRPS после |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1–6 ч | — | — | — | — | — | — |
| 7–24 ч | — | — | — | — | — | — |
| 25–72 ч | — | — | — | — | — | — |
| 73–168 ч | — | — | — | — | — | — |

По запрошенной длине истории, весь горизонт: «до» — `reliability.json` (`coverage.report`,
разрез «длина истории»), «после» — `calibrated.json` (`report`, тот же разрез).

| длина истории | PICP90 до | PICP90 после | ширина 90 до, °C | ширина 90 после, °C |
|---|:-:|:-:|:-:|:-:|
| 0 ч | — | — | — | — |
| 6 ч | — | — | — | — |
| 24 ч | — | — | — | — |
| 72 ч | — | — | — | — |
| 168 ч | — | — | — | — |
| 336 ч | — | — | — | — |
| 672 ч | — | — | — | — |

Офлайн-прогон адаптивной калибровки устройства (`calibrated.json`, ключ `aci`):

| бин лидов | PICP90 без ACI | PICP90 с ACI | ширина 90 без, °C | ширина 90 с, °C | обратных связей | θ в конце, медиана |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1–6 ч | — | — | — | — | — | — |
| 7–24 ч | — | — | — | — | — | — |
| 25–72 ч | — | — | — | — | — | — |
| 73–168 ч | — | — | — | — | — | — |
| весь поток | — | — | — | — | — | — |

Обновлений, упёршихся в границу множителя: —.

Подгонка таблицы на калибровочных окнах, в выборке — `results/calibration/conformal.report.json`:

| бин валидных часов истории | окон | строка таблицы | PICP90 до | PICP90 после |
|---|:-:|:-:|:-:|:-:|
| L=0 | — | — | — | — |
| 1–24 ч | — | — | — | — |
| 25–168 ч | — | — | — | — |
| 169–672 ч | — | — | — | — |

</details>

<details>
<summary>7. Абляции и сиды — <code>results/ablations/{internal,external}/{metrics,seeds}.json</code>, <code>results/ablations/significance.json</code></summary>

Правило значимости — [`METHODS.md`](METHODS.md), раздел «Абляции и сиды».

ERA5, станции `unseen_test` (`internal/metrics.json`, `internal/seeds.json`):

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

| метрика, 3 сида | среднее | мин | макс | ст. откл. |
|---|:-:|:-:|:-:|:-:|
| Skill@24 | — | — | — | — |
| Skill@72 | — | — | — | — |
| Skill@168 | — | — | — | — |
| MAE@24 | — | — | — | — |
| CRPS@24 | — | — | — | — |
| PICP90@24 | — | — | — | — |

GHCNh, станции `external_test` (`external/metrics.json`, `external/seeds.json`):

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

| метрика, 3 сида | среднее | мин | макс | ст. откл. |
|---|:-:|:-:|:-:|:-:|
| Skill@24 | — | — | — | — |
| Skill@72 | — | — | — | — |
| Skill@168 | — | — | — | — |
| MAE@24 | — | — | — | — |
| CRPS@24 | — | — | — | — |
| PICP90@24 | — | — | — | — |

Значимость: выражена на ERA5 и GHCNh с одним знаком Δ (`significance.json`, ключ `verdict`):

| абляция | Skill@24 | Skill@72 | Skill@168 | CRPS@24 | PICP90@24 |
|---|:-:|:-:|:-:|:-:|:-:|
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
| МАЯК | — / — | — / — | — / — | — / — | — / — |
| LRU | — / — | — / — | — / — | — / — | — / — |
| GRU seq2seq | — / — | — / — | — / — | — / — | — / — |
| DLinear | — / — | — / — | — / — | — / — | — / — |
| PatchTST | — / — | — / — | — / — | — / — | — / — |
| Damped persistence | — / — | — / — | — / — | — / — | — / — |
| Seasonal-naive 24ч | — / — | — / — | — / — | — / — | — / — |
| Климатология | — / — | — / — | — / — | — / — | — / — |

Разрезы, лид 24 ч, МАЯК (`breakdowns.json`):

| разрез | Skill@24 | MAE@24 | CRPS@24 | PICP90@24 | станций |
|---|:-:|:-:|:-:|:-:|:-:|
| валидных часов истории <50% | — | — | — | — | — |
| 50–80% | — | — | — | — | — |
| 80–95% | — | — | — | — | — |
| 95–100% | — | — | — | — | — |
| до обучающей точки <25 км | — | — | — | — | — |
| 25–100 км | — | — | — | — | — |
| 100–300 км | — | — | — | — | — |
| ≥300 км | — | — | — | — | — |
| давление есть | — | — | — | — | — |
| давления нет | — | — | — | — | — |
| Δ высоты станция−ЦМР (строки из JSON) | — | — | — | — | — |

Внутренний против внешнего на общих зонах (`transfer.json`):

| модель | Skill@24 внутр. | Skill@24 внешн. | Δ | валидных часов истории внутр. | валидных часов истории внешн. |
|---|:-:|:-:|:-:|:-:|:-:|
| МАЯК | — | — | — | — | — |
| Климатология | — | — | — | — | — |

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
| МАЯК | — | — |
| LRU | — | — |
| GRU seq2seq | — | — |
| DLinear | — | — |
| PatchTST | — | — |

**12. Профили аугментаций, внешний тест** — `results/evaluate/external/metrics.json`,
`results/augment_none/external/metrics.json`

| профиль | Skill@24 | Skill@72 | Skill@168 | CRPS@24 | PICP90@24 |
|---|:-:|:-:|:-:|:-:|:-:|
| МАЯК, `default` | — | — | — | — | — |
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

**2. Установка** в виртуальное окружение на устройстве, только `numpy` и `onnxruntime`.

```bash
ssh pi@device 'sudo mkdir -p /opt/mayak /var/lib/mayak && sudo chown $USER /opt/mayak /var/lib/mayak'
ssh pi@device 'python3 -m venv /opt/mayak/venv'

# вариант A: колесо с рабочей машины
uv build --wheel                                     # dist/mayak-0.1.0-py3-none-any.whl
scp dist/mayak-0.1.0-py3-none-any.whl pi@device:/opt/mayak/
ssh pi@device '/opt/mayak/venv/bin/pip install /opt/mayak/mayak-0.1.0-py3-none-any.whl'
# вариант B: из git по тегу
ssh pi@device '/opt/mayak/venv/bin/pip install "mayak @ git+https://github.com/feyra-labs/tempcast.git@<тег>"'

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

| поле `status` | что это |
|---|---|
| `filled` | часов в окне после холодного старта |
| `valid_hours` | часов с валидной температурой во входе модели; по ним выбирается строка конформной таблицы (не больше 672) |
| `theta`, `aci_lead_bins` | θ по бинам лидов и сами бины, ч |
| `aci_updates`, `aci_misses` | обратных связей и промахов по бинам лидов с последнего сброса |
| `conformal` | применяется ли конформная таблица |
| `state_bytes` | размер состояния на диске: 3236 Б (`state_a.bin`, `state_b.bin` в `--state-dir`) |
| `memory_bytes` | окно, таблица климатологии и при `--aci` кольцо калибровки: 82 368 Б без `--aci`, 95 808 Б с ним |
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
| `scripts/qc_false_alarms.py` | доля ложных срабатываний QC на реанализе, по ней подбираются пороги |
| `scripts/make_qc_golden.py` | перегенерация регрессионного вектора QC, только при смене правил |
| `scripts/aug_reference.py` | эталонные окна аугментаций и их действие на QC |
| `scripts/run.py` | обучение, единственный вход: подбор lr, этапы, абляции, сиды, профили аугментаций, этап 2 |
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
| `conf/train/` | протокол обучения; `train/debug.yaml` — отладка на CPU |
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
