# Результаты

Здесь лежат JSON всех таблиц раздела «Результаты» корневого README. Файлы пишут команды,
указанные над каждой таблицей; в каждом файле два ключа: `run` — запись о прогоне (формат,
время UTC, коммит и признак незакоммиченных изменений, версии библиотек, чекпойнты, сид
оценки, параметры бутстрапа) и `table` — сама таблица. Числа в README переносятся из этих
файлов; число без файла в README не попадает.

Каталог коммитится целиком вместе с заполнением README. Git хранит здесь только JSON и этот
файл: графики, CSV и экспортированные модели, которые команды пишут рядом, игнорируются.

| поле | значение |
|---|---|
| дата прогона (UTC) | — |
| коммит | — |
| набор данных (`data/fetch_meta.json`, ключ кэша) | — |
| чекпойнт МАЯК | — |
| чекпойнты бейзлайнов | — |
| чекпойнты абляций и сидов | — |
| конформная таблица | — |
| устройство замеров | — |

| файл | таблица README |
|---|---|
| `evaluate/internal/metrics.json` | 1. Метрики по лидам |
| `evaluate/internal/history.json` | 2. Скилл по длине истории |
| `evaluate/internal/breakdowns.json` | 3–4. Роли станций, сезоны, зоны |
| `evaluate/internal/zones.json` | 4. Зоны Кёппена на лидах 24 и 72 ч |
| `evaluate/internal/reliability.json` | 5. Надёжность |
| `evaluate/internal/calibrated.json`, `evaluate/internal/reliability.json` | 6. После калибровки: тестовые окна |
| `calibration/conformal.report.json` | 6. Подгонка таблицы: калибровочные окна, в выборке |
| `evaluate/internal/coldstart.json` | проверка поля при нулевой истории |
| `ablations/internal/metrics.json`, `seeds/internal/seeds.json` | 7. Абляции и сиды |
| `evaluate/external/*.json` | 8. Внешний тест |
| `robustness/robustness.json` | 9. Робастность |
| `device/bench.json` | 10. Устройство |
| `evaluate/params.json` | 11. Число параметров |
