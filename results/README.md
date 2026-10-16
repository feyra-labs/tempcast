# Результаты

Здесь лежат JSON всех таблиц раздела «Результаты» корневого README. Файлы пишут команды
из корневого README; в каждом файле два ключа: `run` — запись о прогоне (формат, время UTC,
коммит и признак незакоммиченных изменений, версии библиотек, чекпойнты, сид оценки,
параметры бутстрапа, роли станций, временное окно и число окон набора) и `table` — сама
таблица. Числа в README переносятся из этих файлов; число без файла в README не попадает.

Каталог коммитится целиком вместе с заполнением README. Git хранит здесь только JSON и этот
файл: графики, CSV и экспортированные модели, которые команды пишут рядом, игнорируются.

| поле | значение |
|---|---|
| дата прогона (UTC) | 2026-10-16 14:59:22 (стенд оценки, `evaluate/`) |
| коммит | `71a86bed60372e18180be9290868a19f4c4203cc`, с незакоммиченными изменениями (`dirty: true`) |
| набор данных (`data/fetch_meta.json`, ключ кэша) | — |
| чекпойнт МАЯК | `runs/mayak/stageB/best.ckpt`, отпечаток `c0e7824d79311cc2` |
| чекпойнты бейзлайнов | `runs/{gru,dlinear,lru,patchtst}/stageB/best.ckpt` |
| чекпойнты абляций (`runs/mayak-<абляция>`) | — |
| чекпойнт МАЯК без аугментаций (`augment=none`) | — |
| конформная таблица | `runs/conformal.npy`: 32 станции `unseen_val`, 4 600 окон, отпечаток окон `9881c588075daa53` |
| устройство замеров | — |

| файл | таблица README |
|---|---|
| `evaluate/internal/metrics.json` | 1. Метрики по лидам, новые станции |
| `evaluate/internal/history.json` | 2. Скилл по длине истории |
| `evaluate/internal/breakdowns.json` | 3–4. Роли станций, сезоны, зоны |
| `evaluate/train_stations/metrics.json` | 3–4. Обучающие станции в тестовом году, метрики по лидам |
| `evaluate/internal/zones.json` | 4. Зоны Кёппена на лидах 24 и 72 ч |
| `evaluate/internal/reliability.json` | 5. Надёжность |
| `evaluate/internal/calibrated.json`, `evaluate/internal/reliability.json` | 6. После калибровки: тестовые окна и офлайн-прогон адаптивной калибровки |
| `calibration/conformal.report.json` | 6. Подгонка таблицы: калибровочные окна, в выборке |
| `evaluate/internal/coldstart.json` | проверка поля при нулевой истории |
| `ablations/internal/metrics.json` | 7. Абляции, ERA5 (`unseen_test`) |
| `ablations/external/metrics.json` | 7. Абляции, GHCNh (`external_test`) |
| `evaluate/external/*.json` | 8. Внешний тест, в том числе `transfer.json` |
| `robustness/robustness.json` | 9. Робастность |
| `device/bench.json` | 10. Устройство |
| `evaluate/params.json` | 11. Число параметров |
| `evaluate/external/metrics.json`, `augment_none/external/metrics.json` | 12. Профили аугментаций, внешний тест |

Строка «МАЯК (доп. настройка)†» есть в файлах `evaluate/` только тогда, когда оценка шла с
`--tuned-ckpt`; запись о прогоне помечает её как неравную бейзлайнам, в таблицы README она
не переносится. Стенды оценки абляций и профиля `none` пишут в свои каталоги и остальные
таблицы набора; в README из них берутся только перечисленные выше.
