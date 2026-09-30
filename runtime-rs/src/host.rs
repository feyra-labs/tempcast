//! Хост устройства: построчный протокол, состояние на диске, откат.
//!
//! Команды, по одной на строку:
//!   obs <секунды UTC> <T> <P> <RH>   значение: число, "-" (нет данных) или "nan";
//!   forecast [<секунды UTC>]         выпуск после последнего шага;
//!   status                           сводка рантайма, в том числе текущая точка, точка
//!                                    загруженного состояния и исход их сравнения.
//! Ответ на каждую команду - одна строка JSON. Момент наблюдения лежит на целом часе и
//! позже последнего шага; пропущенные часы заполняются пустыми шагами, в том числе
//! простой между перезапусками. После каждого наблюдения состояние пишется на диск.
//! До первого шага момент выпуска - текущий час по часам устройства; секунды в команде
//! выпуска заменяют часы устройства. Выпуск, который не удался, заменяется
//! климатологией точки. Ошибка команды - ответ {"error": ...}, хост работает дальше.
//!
//! Совпадение с хостом на Python закреплено общими эталонными сценариями.
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

use crate::memory::{peak_rss_bytes, rss_bytes};
use crate::state::Snapshot;
use crate::store::StateStore;
use crate::{Error, Result, Runtime};

/// Текущее время UTC в секундах от эпохи по часам устройства.
pub fn system_clock() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Значение наблюдения из команды: число, "-" или пусто (нет данных), "nan".
pub fn parse_obs(s: &str) -> Result<Option<f64>> {
    match s {
        "-" | "" => Ok(None),
        "nan" | "NaN" => Ok(Some(f64::NAN)),
        v => v
            .parse()
            .map(Some)
            .map_err(|_| Error::new(format!("значение наблюдения не число: {v}"))),
    }
}

fn parse_sec(s: &str) -> Result<i64> {
    s.parse().map_err(|_| Error::new(format!("время не число: {s}")))
}

pub struct Host {
    pub rt: Runtime,
    store: Option<StateStore>,
    buf: Vec<u8>,
    clock: Box<dyn Fn() -> i64>,
}

impl Host {
    /// Хост поверх рантайма; без каталога состояния ничего не пишется на диск.
    pub fn new(rt: Runtime, store: Option<StateStore>) -> Self {
        Host {
            rt,
            store,
            buf: Vec::new(),
            clock: Box::new(system_clock),
        }
    }

    /// Другие часы устройства: функция возвращает секунды UTC от эпохи.
    pub fn with_clock(mut self, clock: impl Fn() -> i64 + 'static) -> Self {
        self.clock = Box::new(clock);
        self
    }

    /// Восстановление из самого свежего годного файла состояния; None - чистый старт.
    pub fn restore(&mut self) -> Option<PathBuf> {
        let store = self.store.as_mut()?;
        let (dims, bounds) = (self.rt.manifest.dims.clone(), self.rt.manifest.phys_bounds());
        let files = store.ordered(|b| Snapshot::parse(b, &dims, &bounds).map(|s| s.last_hour));
        for (f, raw) in files {
            match self.rt.load_state(&raw) {
                Ok(()) => {
                    eprintln!("mayak-rt: состояние восстановлено из {} ({} Б)", f.display(), raw.len());
                    store.mark_restored(&f);
                    return Some(f);
                }
                Err(e) => eprintln!("mayak-rt: состояние {} не принято: {e}", f.display()),
            }
        }
        eprintln!("mayak-rt: чистый старт (история пуста)");
        None
    }

    /// Одна строка протокола; None для пустой строки.
    pub fn handle(&mut self, line: &str) -> Option<Value> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        let reply = match parts.as_slice() {
            [] => return None,
            ["obs", ts, t, p, rh] => self.obs(ts, t, p, rh),
            ["forecast"] => self.forecast(None),
            ["forecast", ts] => self.forecast(Some(*ts)),
            ["status"] => Ok(self.status()),
            _ => Err(Error::new(format!("неизвестная команда: {}", line.trim()))),
        };
        Some(reply.unwrap_or_else(|e| json!({"error": e.to_string()})))
    }

    fn obs(&mut self, ts: &str, t: &str, p: &str, rh: &str) -> Result<Value> {
        let sec = parse_sec(ts)?;
        if sec.rem_euclid(3600) != 0 {
            return Err(Error::new(format!("момент {sec} не на целом часе UTC")));
        }
        let obs = [parse_obs(t)?, parse_obs(p)?, parse_obs(rh)?];
        let codes = self.rt.step(obs, sec.div_euclid(3600))?;
        if let Some(store) = self.store.as_mut() {
            self.rt.serialize(&mut self.buf);
            store.save(&self.buf)?;
        }
        Ok(json!({"ok": true, "codes": codes}))
    }

    fn forecast(&mut self, ts: Option<&str>) -> Result<Value> {
        let arg = ts.map(parse_sec).transpose()?;
        // часы устройства нужны только до первого шага
        let now = match self.rt.last_hour() {
            Some(_) => None,
            None => Some(arg.unwrap_or_else(|| (self.clock)()).div_euclid(3600)),
        };
        let nq = self.rt.n_quantiles();
        let theta = self.rt.theta();
        let f = self.rt.safe_forecast(now)?;
        let q: Vec<&[f32]> = f.q.chunks(nq).collect();
        Ok(
            json!({"after_unix_hour": f.after_hour, "fallback": f.fallback, "theta": theta,
                  "mu": f.mu, "q": q}),
        )
    }

    fn status(&self) -> Value {
        let rt = &self.rt;
        json!({"filled": rt.filled(), "theta": rt.theta(),
               "conformal": rt.conformal_applied(),
               "aci_updates": rt.aci_updates(), "aci_misses": rt.aci_misses(),
               "idle_hours": rt.idle_hours(), "fallbacks": rt.fallbacks,
               "state_bytes": rt.state_nbytes(), "last_unix_hour": rt.last_hour(),
               "memory_bytes": rt.memory_bytes(), "site": rt.site(),
               "loaded_site": rt.loaded_site(), "site_change": rt.site_change().map(|c| c.as_str()),
               "rss_bytes": rss_bytes(), "peak_rss_bytes": peak_rss_bytes()})
    }
}
