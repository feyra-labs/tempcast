<h1 align="center">МАЯК</h1>

<p align="center"><b>Вероятностный прогноз температуры воздуха на 168 часов по одной метеостанции,
с рантаймом для слабого устройства.</b></p>

МАЯК получает почасовую историю одной точки (температура, давление, влажность и маска
пропусков), её координаты и высоту и выдаёт 7 квантилей (5, 10, 25, 50, 75, 90 и 95 %) на
каждый из 168 часов вперёд. Обучение идёт только на реанализе ERA5; наблюдения реальной
сети GHCNh служат внешним тестом. На устройстве модель работает на Python с ONNX Runtime:
нужны только `numpy` и `onnxruntime`, память фиксирована, состояние переживает перезапуск.
Модели и их источники — [`MODELS.md`](MODELS.md); данные, обучение, оценка, калибровка,
устройство и формулы — [`METHODS.md`](METHODS.md).

**Ограничения.** Обоснования — в [`METHODS.md`](METHODS.md).

- Температура — целые градусы Цельсия везде: в данных, кэше, обучении, оценке и на
  устройстве. Влажность — целые проценты, давление — десятые гПа. Единицы не проверяются:
  что прибор присылает °C, обеспечивает владелец прибора.
- Датчик отчитывается раз в час. Пропуски и разрывы допустимы; регулярный шаг в
  несколько часов вне области проекта.
- Время — UTC, момент наблюдения лежит на целом часе. Нужны координаты и высота точки.
- История модели — не больше 672 ч (28 суток), горизонт — 168 ч. Прогноз не видит того,
  чего нет в истории одной точки, и на дальних лидах стремится к климатологии.
- Обучение только на реанализе; перенос на реальные приборы измеряется внешним тестом.
- Климат-поле учится на ~270 точках с шагом в сотни километров. Между ними оно
  интерполирует; локальные особенности (побережье, долины) передаёт только высота,
  остальное модель набирает из истории модой P. Запоминание точек ограничивают полоса
  частот координатных признаков и затухание матриц поля; его величину показывает отчёт
  этапа A.
- Смещение станции модель переносит на горизонт как константу плюс суточную
  составляющую с фиксированным периодом 24 ч.
- Дрейфа и ошибки масштаба прибора в обучении нет.
- Эталон скилла — климатология самой станции с базисом грубее базиса климат-поля.
- Основные таблицы считаются по станциям, которых модель не видела при обучении
  (`unseen_test`).

## Установка

Нужны Python 3.11+ и [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/feyra-labs/tempcast.git
cd tempcast
uv sync --extra train --group dev   # PyTorch берётся из индекса CUDA 12.8, работает и на CPU
uv run pytest -q                    # тесты на синтетике, данные не нужны
```

Все команды ниже запускаются из корня репозитория через `uv run`, у каждой есть `--help`.
Установка на устройство — в разделе «Устройство».

## Данные

Данные в репозиторий не входят и собираются скриптами. Нужна карта зон Кёппена — Гейгера
с кодами 1..30, например `Beck_KG_V1_present_0p0083.tif` из Beck et al. (2018).

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

# доля ложных срабатываний QC на реанализе (по ней подбираются пороги QC)
uv run python scripts/qc_false_alarms.py --manifest data/manifest.csv
```

Для проверки конвейера без скачивания: `uv run python scripts/make_synth.py`, затем
`make_splits.py` и `build_cache.py` как выше.

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

Роли станций (`train`, `unseen_val`, `unseen_test`, `external_test`) и временные блоки
описаны в [`METHODS.md`](METHODS.md), раздел «Данные и сплиты».

Лицензии данных: Open-Meteo — CC BY 4.0, бесплатный API только для некоммерческого
использования; ERA5 — Copernicus Climate Change Service; GHCNh — NOAA NCEI. Полные
ссылки — в [`METHODS.md`](METHODS.md).

## Обучение

Этап 1 сравнения: каждая нейросеть, включая МАЯК, проходит одну сетку скоростей обучения
(`--lr-search`), затем полный прогон с выбранным значением.

```bash
uv run python scripts/train.py --arch mayak --lr-search --accelerator gpu
uv run python scripts/train_neurobaselines.py --lr-search --accelerator gpu   # gru, dlinear, lru, patchtst
```

Каталог прогона — `runs/<arch>/`: журнал `protocol.json` с записью о подборе, прогоны
сетки `lr_search/<lr>/`, чекпойнты `stageA/best.ckpt` и `stageB/best.ckpt`. Отчёт о поле
этапа A — `stageA/report.json` и `stageA/report/*.png`.

Этап 2, необязательный: дополнительная настройка МАЯК любыми гиперпараметрами по
валидации. Каталог `runs/mayak-tuned/`, в таблицах — отдельная строка.

```bash
uv run python scripts/train.py --arch mayak --extra-tuning --lr 1e-3 --accelerator gpu
```

Этапы по отдельности, с решением человека между ними:

```bash
uv run python scripts/train.py --arch mayak --stages A --accelerator gpu   # кандидаты и отчёт о поле
uv run python scripts/stage_report.py a --run runs/mayak                   # пересчёт отчёта
uv run python scripts/diagnose_stage_a.py --ckpt runs/mayak/stageA/best.ckpt
uv run python scripts/train.py --arch mayak --stages B --accelerator gpu \
    --init-from runs/mayak/stageA/candidates/<шаг>.ckpt --require-gate 1.05
# пробный этап B с нескольких кандидатов и их сравнение
uv run python scripts/train.py --arch mayak --stages B --probe-steps 20000 --accelerator gpu \
    --init-from runs/mayak/stageA/candidates/<шаг>.ckpt --tag mayak-probe-<шаг>
uv run python scripts/stage_report.py b --runs runs/mayak-probe-<шаг1> runs/mayak-probe-<шаг2>
```

Отладочный прогон на CPU: `uv run python scripts/run.py train=debug run.accelerator=cpu`
или `model=mayak_small` (уменьшенный МАЯК).

### Абляции и сиды

Абляции и повторы с другими сидами берут скорость обучения основного МАЯК (`--lr-from`).
Абляции: `no_compression`, `no_passport`, `no_solar`, `no_mode_groups`, `no_offset_aug`,
`no_correction`, `no_persistent`; `none` — полная модель. Что снимает каждая —
[`MODELS.md`](MODELS.md), раздел «Абляции».

```bash
ABLATIONS="no_compression no_passport no_solar no_mode_groups no_offset_aug no_correction no_persistent"
for a in $ABLATIONS; do      # runs/mayak-<абляция>/
    uv run python scripts/train.py --arch mayak --ablate $a --lr-from runs/mayak --accelerator gpu
done
for s in 1 2; do             # runs/mayak-s<сид>/
    uv run python scripts/train.py --arch mayak --seed $s --tag mayak-s$s --lr-from runs/mayak \
        --accelerator gpu
done
```

То же через Hydra: `uv run python scripts/run.py -m ablation=no_compression,no_solar
run.lr_from=runs/mayak`; каталог — `runs/mayak-<абляция>-s<сид>/`, конфиги — `conf/`.

### Профили аугментаций

Профили — `default` (по умолчанию) и `none` (`conf/augment/`). Прогон без аугментаций
задаётся только через Hydra и получает свой каталог:

```bash
uv run python scripts/run.py augment=none run.lr_from=runs/mayak   # runs/mayak-none-aug_none-s0/
uv run python -m mayak.evaluate --ckpt runs/mayak-none-aug_none-s0/stageB/best.ckpt \
    --external-manifest data/ghcnh/manifest.csv --results-dir results/augment_none
```

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
    --bootstrap 1000 --save-preds runs/preds --results-dir results/evaluate

# таблица 7: абляции и сиды
uv run python -m mayak.evaluate --ckpt runs/mayak/stageB/best.ckpt \
    --ablation-ckpt $(for a in $ABLATIONS; do echo runs/mayak-$a/stageB/best.ckpt; done) \
    --results-dir results/ablations
uv run python -m mayak.evaluate --ckpt runs/mayak/stageB/best.ckpt \
    runs/mayak-s1/stageB/best.ckpt runs/mayak-s2/stageB/best.ckpt --results-dir results/seeds

# разрезы покрытия и офлайн-прогон адаптивной калибровки по сохранённым предсказаниям
uv run python -m mayak.calibration --preds runs/preds/internal.npz \
    --history-preds runs/preds/internal_history.npz --hourly-preds runs/preds/internal_hourly.npz \
    --external-preds runs/preds/external.npz --out-dir runs/calibration

# таблица 9: робастность
uv run python -m mayak.robustness --ckpt runs/mayak/stageB/best.ckpt \
    --external-manifest data/ghcnh/manifest.csv --out-dir results/robustness
```

Параметры калибровки — `conf/calibration/default.yaml`, робастности —
`conf/robustness/default.yaml`. Стенд оценки отказывает, если модели обучены по разным
протоколам или без записи о подборе скорости обучения по одной сетке.

## Результаты

Чисел пока нет: таблицы заполняются из JSON в [`results/`](results/README.md). Основной
внутренний набор — станции `unseen_test` в тестовом году; основные таблицы — при полной
истории 672 ч, по сырым выходам моделей. В ячейке «пул / макро»; в JSON у каждого числа
интервал бутстрапа по станциям (90 %). Как считаются метрики — [`METHODS.md`](METHODS.md),
раздел «Оценка». Строка «МАЯК (доп. настройка)†» появляется только с `--tuned-ckpt` и в
сравнение на равных не входит.

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

Строка «обучающие» — набор обучающих станций в тестовом году
(`results/evaluate/train_stations/`), остальные строки — новые станции.

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

По длине истории, весь горизонт: «до» — `reliability.json` (`coverage.report`, разрез
«длина истории»), «после» — `calibrated.json` (`report`, тот же разрез).

| бин таблицы | длина истории | PICP90 до | PICP90 после | ширина 90 до, °C | ширина 90 после, °C |
|---|:-:|:-:|:-:|:-:|:-:|
| L=0 | 0 ч | — | — | — | — |
| 1–24 ч | 6 ч | — | — | — | — |
| 1–24 ч | 24 ч | — | — | — | — |
| 25–168 ч | 72 ч | — | — | — | — |
| 25–168 ч | 168 ч | — | — | — | — |
| 169–672 ч | 336 ч | — | — | — | — |
| 169–672 ч | 672 ч | — | — | — | — |

Адаптивная калибровка устройства, офлайн-прогон с ежечасным выпуском: 8 станций × 720 ч
(`calibrated.json`, ключ `aci`).

| бин лидов | PICP90 без ACI | PICP90 с ACI | ширина 90 без, °C | ширина 90 с, °C | обратных связей | θ в конце, медиана |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1–6 ч | — | — | — | — | — | — |
| 7–24 ч | — | — | — | — | — | — |
| 25–72 ч | — | — | — | — | — | — |
| 73–168 ч | — | — | — | — | — | — |
| весь поток | — | — | — | — | — | — |

Обновлений, упёршихся в границу множителя: —.

Подгонка таблицы на калибровочных окнах, в выборке — `results/calibration/conformal.report.json`:

| бин длины истории | окон | строка таблицы | PICP90 до | PICP90 после |
|---|:-:|:-:|:-:|:-:|
| L=0 | — | — | — | — |
| 1–24 ч | — | — | — | — |
| 25–168 ч | — | — | — | — |
| 169–672 ч | — | — | — | — |

</details>

<details>
<summary>7. Абляции и сиды — <code>results/ablations/internal/metrics.json</code>, <code>results/seeds/internal/seeds.json</code></summary>

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
| MAE@24 | — | — | — | — |
| CRPS@24 | — | — | — | — |
| PICP90@24 | — | — | — | — |

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

На устройство копируются пакет `mayak` без дополнительных групп и каталог экспорта
модели. Нужны 64-битная ОС (aarch64 или x86_64, например Raspberry Pi 3/4/5 с 64-битной
Raspberry Pi OS) и Python 3.11+.

**1. Экспорт** (на рабочей машине): два графа ONNX, манифест и конформная таблица.

```bash
uv run python scripts/export_runtime.py --ckpt runs/mayak/stageB/best.ckpt \
    --conformal runs/conformal.npy --aci --out runtime/model
```

**2. Установка.** Пакет собирается на рабочей машине или ставится из git по тегу в
виртуальное окружение на устройстве; ставятся только `numpy` и `onnxruntime`.

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

**3. Запуск.** Хост читает команды со stdin по одной на строку и отвечает строкой JSON.
Координаты и высота точки над уровнем моря (`--lat`, `--lon`, `--elev`) обязательны.

```bash
printf 'obs 1767225600 11 1012.4 81\nforecast\nstatus\n' | \
    /opt/mayak/venv/bin/python -m mayak.runtime.run_inference --model /opt/mayak/model \
    --lat 52.37 --lon 4.90 --elev -2 --state-dir /var/lib/mayak --aci
```

| команда | ответ |
|---|---|
| `obs <секунды UTC> <T> <P> <RH>` | `{"ok": true, "codes": [...]}` — коды QC часа; значение — число, `-` или `nan` |
| `forecast [<секунды UTC>]` | `after_unix_hour`, `fallback`, `theta`, медиана `mu` и квантили `q` по 168 лидам |
| `status` | сводка, поля ниже |
| ошибка любой команды | `{"error": "..."}`, хост продолжает работу |

Состояние пишется в `--state-dir` после каждого `obs` (`state_a.bin`, `state_b.bin`).
Прибор рассчитан на ежечасный выпуск: `forecast` отправляется после каждого `obs`. С
`--aci` интервал подстраивается адаптивной калибровкой; правило и поведение при выпуске
реже раза в час — [`METHODS.md`](METHODS.md), раздел «Калибровка». Высоту нужно указать
сразу: смена точки больше порогов манифеста (0,5° по координатам, 100 м по высоте)
считается переносом прибора.

| поле `status` | что это |
|---|---|
| `filled`, `history_hours` | часов в окне после холодного старта; длина истории, по которой выбирается строка конформной таблицы (не больше 672) |
| `theta`, `aci_lead_bins` | θ по бинам лидов и сами бины, ч |
| `aci_updates`, `aci_misses` | обратных связей и промахов по бинам лидов с последнего сброса |
| `conformal` | применяется ли конформная таблица |
| `state_bytes` | размер состояния на диске: 3236 Б |
| `memory_bytes` | окно, таблица климатологии и при `--aci` кольцо калибровки: 82 368 Б без `--aci`, 95 808 Б с ним |
| `site`, `loaded_site`, `site_change` | точка прибора, точка загруженного состояния и исход сравнения: `same`, `refined`, `moved` |
| `idle_hours`, `fallbacks`, `last_unix_hour` | простой, откаты к климатологии, последний час |
| `rss_bytes`, `peak_rss_bytes` | память процесса |

**4. Служба.** Программа датчика пишет строки в именованный канал, хост работает службой
systemd на Python из окружения:

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

На рабочей машине тот же хост открывает чекпойнт напрямую:
`uv run python -m mayak.runtime.run_inference --ckpt runs/mayak/stageB/best.ckpt
--conformal runs/conformal.npy --lat 52.37 --lon 4.90 --elev -2`.

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
| `scripts/train.py` | обучение одной архитектуры по протоколу: подбор lr, этапы, абляции, этап 2 |
| `scripts/train_neurobaselines.py` | обучение всех нейробейзлайнов той же функцией |
| `scripts/run.py` | обучение через Hydra: композиция конфигов, групповые запуски, профили аугментаций |
| `scripts/stage_report.py` | отчёт о поле после этапа A и сравнение пробных запусков этапа B |
| `scripts/diagnose_stage_a.py` | диагностика поля после этапа A |
| `scripts/calibrate.py` | подгонка конформной таблицы МАЯК |
| `scripts/export_runtime.py` | экспорт графов, манифеста и таблицы для устройства |
| `scripts/bench_device.py` | замеры устройства: час, выпуск, память, расхождение с оценкой |

| модуль | назначение |
|---|---|
| `python -m mayak.evaluate` | стенд оценки: таблицы 1–8 и 11 |
| `python -m mayak.calibration` | разрезы покрытия и офлайн-прогон адаптивной калибровки по сохранённым предсказаниям |
| `python -m mayak.robustness` | сценарии робастности обученной модели |
| `python -m mayak.runtime.run_inference` | хост устройства |

| конфиг | назначение |
|---|---|
| `conf/model/` | архитектуры: `mayak`, `mayak_small` (уменьшенный МАЯК для отладки на CPU), `gru`, `dlinear`, `lru`, `patchtst` |
| `conf/ablation/` | `none` и семь абляций МАЯК |
| `conf/augment/` | профили аугментаций `default` и `none` |
| `conf/data/`, `conf/train/` | данные и протокол обучения; `train/debug.yaml` — отладка на CPU |
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
