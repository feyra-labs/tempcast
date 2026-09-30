"""Хост устройства на Python: команды, состояние на диске, откат.

Хост говорит тем же построчным протоколом и пишет то же состояние, что хост на Rust.
Совпадение двух хостов закреплено общими эталонными сценариями, которые проходят оба.

Команды, по одной на строку:

* ``obs <секунды UTC> <T> <P> <RH>`` - наблюдение часа. Значение: число, ``-`` (нет
  данных) или ``nan``. Момент лежит на целом часе и позже последнего шага. Пропущенные
  часы заполняются пустыми шагами. После шага состояние записывается на диск. Ответ:
  ``{"ok": true, "codes": [...]}`` с кодами контроля качества часа.
* ``forecast [<секунды UTC>]`` - выпуск после последнего шага. До первого шага момент
  выпуска - текущий час по часам устройства; секунды в команде заменяют часы
  устройства. Ответ: момент выпуска, признак отката, множитель калибровки, медиана и
  квантили.
* ``status`` - сводка рантайма, в том числе текущая точка, точка загруженного состояния
  и исход их сравнения: та же точка, уточнение или перенос прибора.

Любая ошибка команды - ответ ``{"error": "..."}``, хост продолжает работу.

Состояние пишется атомарно в два чередующихся файла: временный файл, fsync, замена
файла очереди, fsync каталога. При старте свежий файл выбирается по содержимому: по
абсолютному часу последнего шага из заголовка. Файл, который не разбирается, идёт в
конец очереди. Следующая запись идёт в файл, который не был восстановлен.
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
import time

from mayak.runtime.streaming import parse_state

log = logging.getLogger(__name__)

STATE_FILES = ("state_a.bin", "state_b.bin")


def _mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


class StateStore:
    """Два чередующихся файла состояния в одном каталоге.

    Args:
        directory: каталог состояния; создаётся, если его нет.
    """

    def __init__(self, directory):
        self.dir = str(directory)
        os.makedirs(self.dir, exist_ok=True)
        self.files = [os.path.join(self.dir, f) for f in STATE_FILES]
        self.toggle = 0

    def candidates(self):
        """Существующие файлы состояния, новейший по времени изменения первым."""
        have = [(f, _mtime(f)) for f in self.files]
        have = [(f, t) for f, t in have if t is not None]
        have.sort(key=lambda a: -a[1])
        return [f for f, _ in have]

    def ordered(self, window):
        """Файлы состояния с содержимым, свежий первым.

        Свежее то состояние, у которого позже абсолютный час последнего шага. Состояние
        без шагов старше любого состояния с шагами. Файл, который не разбирается, идёт в
        конец. При равенстве раньше идёт файл, изменённый позже.

        Args:
            window: длина окна модели, часы.

        Returns:
            Список пар: путь и байты файла.
        """
        out = []
        for f in self.candidates():
            try:
                with open(f, "rb") as fh:
                    raw = fh.read()
            except OSError:
                continue
            try:
                last = parse_state(raw, window).last_hour
                key = (0, -last) if last is not None else (1, 0)
            except ValueError:
                key = (2, 0)
            out.append((key, f, raw))
        out.sort(key=lambda a: a[0])
        return [(f, raw) for _, f, raw in out]

    def mark_restored(self, path):
        """Следующая запись идёт в другой файл, чтобы восстановленный остался целым."""
        self.toggle = 1 if os.path.abspath(path) == os.path.abspath(self.files[0]) else 0

    def save(self, data):
        """Атомарная запись в очередной файл.

        Args:
            data: байты состояния.

        Returns:
            Путь записанного файла.
        """
        f = self.files[self.toggle % 2]
        tmp = f + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, f)
        try:
            fd = os.open(self.dir, os.O_RDONLY)
        except OSError:
            fd = None
        if fd is not None:
            try:
                os.fsync(fd)
            except OSError:
                pass
            finally:
                os.close(fd)
        self.toggle += 1
        return f


def parse_obs(text):
    """Значение наблюдения из команды.

    Args:
        text: число, ``-`` или пустая строка (нет данных), ``nan``.

    Returns:
        Число или None.

    Raises:
        ValueError: значение не число.
    """
    if text in ("-", ""):
        return None
    if text in ("nan", "NaN"):
        return math.nan
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"значение наблюдения не число: {text}") from None


def _status_kib(key):
    """Поле сводки процесса Linux в КиБ или None, если сводки нет."""
    try:
        with open("/proc/self/status", encoding="ascii") as fh:
            for line in fh:
                if line.startswith(key):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def rss_bytes():
    """Текущая резидентная память процесса, байт; вне Linux None."""
    kib = _status_kib("VmRSS:")
    return None if kib is None else kib * 1024


def peak_rss_bytes():
    """Пиковая резидентная память процесса, байт; вне Linux None."""
    kib = _status_kib("VmHWM:")
    return None if kib is None else kib * 1024


def _f32(v):
    return float(v)


class Host:
    """Хост устройства поверх потокового рантайма.

    Args:
        runtime: потоковый рантайм.
        store: файлы состояния или None, если состояние на диск не пишется.
        clock: функция без аргументов, текущее время UTC в секундах от эпохи.
    """

    def __init__(self, runtime, store=None, clock=time.time):
        self.rt = runtime
        self.store = store
        self.clock = clock

    def restore(self):
        """Восстановление из самого свежего годного файла состояния.

        Returns:
            Путь восстановленного файла или None - чистый старт.
        """
        if self.store is None:
            return None
        for f, raw in self.store.ordered(self.rt.window):
            try:
                self.rt.load_state(raw)
            except Exception as e:
                print(f"mayak-rt: состояние {f} не принято: {e}", file=sys.stderr)
                continue
            print(f"mayak-rt: состояние восстановлено из {f} ({len(raw)} Б)", file=sys.stderr)
            self.store.mark_restored(f)
            return f
        print("mayak-rt: чистый старт (история пуста)", file=sys.stderr)
        return None

    def _now_hour(self, arg=None):
        sec = int(arg) if arg is not None else int(math.floor(self.clock()))
        return sec // 3600

    def obs(self, ts, t, p, rh):
        """Команда наблюдения.

        Returns:
            Ответ: признак успеха и коды контроля качества часа.

        Raises:
            ValueError: момент не число, не на целом часе, не позже последнего шага или
                значение не число.
        """
        try:
            sec = int(ts)
        except ValueError:
            raise ValueError(f"время не число: {ts}") from None
        if sec % 3600:
            raise ValueError(f"момент {sec} не на целом часе UTC")
        codes = self.rt.step(parse_obs(t), parse_obs(p), parse_obs(rh), sec // 3600)
        if self.store is not None:
            self.store.save(self.rt.serialize())
        return {"ok": True, "codes": [int(c) for c in codes]}

    def forecast(self, ts=None):
        """Команда выпуска с откатом к климатологии.

        Args:
            ts: секунды UTC, которые заменяют часы устройства, или None.

        Returns:
            Ответ: момент выпуска, признак отката, множитель калибровки, медиана и
            квантили по лидам.
        """
        if ts is not None:
            try:
                int(ts)
            except ValueError:
                raise ValueError(f"время не число: {ts}") from None
        # часы устройства нужны только до первого шага
        now = self._now_hour(ts) if self.rt.last_hour is None else None
        theta = _f32(self.rt.theta)
        q, mu, fallback = self.rt.safe_forecast(now)
        return {"after_unix_hour": int(self.rt.issue_hour(now)), "fallback": bool(fallback),
                "theta": theta, "mu": [_f32(v) for v in mu],
                "q": [[_f32(v) for v in row] for row in q]}

    def status(self):
        """Команда сводки рантайма."""
        rt = self.rt
        return {"filled": int(rt.filled), "theta": _f32(rt.theta),
                "conformal": rt.conformal is not None,
                "aci_updates": int(rt.aci_updates), "aci_misses": int(rt.aci_misses),
                "idle_hours": int(rt.idle_hours), "fallbacks": int(rt.fallbacks),
                "state_bytes": int(rt.state_nbytes), "last_unix_hour": rt.last_hour,
                "memory_bytes": int(rt.memory_nbytes), "site": list(rt.site),
                "loaded_site": None if rt.loaded_site is None else list(rt.loaded_site),
                "site_change": rt.site_change,
                "rss_bytes": rss_bytes(), "peak_rss_bytes": peak_rss_bytes()}

    def handle(self, line):
        """Одна строка протокола.

        Args:
            line: строка команды.

        Returns:
            Ответ словарём или None для пустой строки.
        """
        parts = line.split()
        if not parts:
            return None
        try:
            if parts[0] == "obs" and len(parts) == 5:
                return self.obs(*parts[1:])
            if parts[0] == "forecast" and len(parts) <= 2:
                return self.forecast(parts[1] if len(parts) == 2 else None)
            if parts == ["status"]:
                return self.status()
            raise ValueError(f"неизвестная команда: {line.strip()}")
        except Exception as e:
            return {"error": str(e)}

    def serve(self, lines, out):
        """Цикл протокола: строка команды на входе, строка JSON на выходе.

        Args:
            lines: итератор строк команд.
            out: поток вывода с методами write и flush.
        """
        for line in lines:
            reply = self.handle(line)
            if reply is None:
                continue
            out.write(json.dumps(reply, ensure_ascii=False) + "\n")
            out.flush()


__all__ = ["STATE_FILES", "Host", "StateStore", "parse_obs", "peak_rss_bytes", "rss_bytes"]
