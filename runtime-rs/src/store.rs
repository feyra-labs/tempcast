//! Атомарная запись состояния в два чередующихся файла.
//!
//! Запись: временный файл → fsync → rename поверх файла очереди → fsync каталога.
//! Отключение питания в любой момент оставляет хотя бы один целый файл: второй файл
//! очереди содержит предыдущий час. Имена файлов - те же, что у Python-рантайма
//! (runtime/state_a.bin, runtime/state_b.bin), состояния взаимозаменяемы.
//!
//! Чтение: порядок задаёт само содержимое, а не время изменения файла. Две записи
//! подряд могут получить одинаковую отметку времени (запись занимает доли
//! миллисекунды), и тогда сортировка по mtime выбирает файл произвольно - на практике
//! это приводило к загрузке состояния на час старше. Сравниваются заполненность окна и
//! час года последней позиции: у соседних записей он отличается на час, поэтому
//! разность по модулю длины года меньше полугода указывает на более свежую запись.
//! Время изменения остаётся тай-брейком для случая, когда содержимое неразличимо.
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use std::cmp::Ordering;

use crate::state::Snapshot;
use crate::Result;

const HALF_YEAR_H: i64 = 183 * 24;

/// Порядок «свежести» двух состояний по их содержимому.
pub fn newer_first(a: &Snapshot, b: &Snapshot) -> Ordering {
    if a.filled != b.filled {
        return b.filled.cmp(&a.filled);
    }
    if a.hoy_last == b.hoy_last {
        return Ordering::Equal;
    }
    let ahead = (a.hoy_last - b.hoy_last).rem_euclid(crate::calendar::YEAR_HOURS[1]);
    if ahead < HALF_YEAR_H {
        Ordering::Less
    } else {
        Ordering::Greater
    }
}

pub struct StateStore {
    dir: PathBuf,
    files: [PathBuf; 2],
    toggle: usize,
}

/// Час последнего шага от эпохи. В состоянии (формат v3, общий с Python) лежит только
/// час года, а он повторяется ежегодно: простой в 8994 ч неотличим от простого в 234 ч.
/// Абсолютный момент хранится отдельным файлом, чтобы не менять общий формат; его
/// отсутствие или несогласованность с состоянием не мешает работе - рантайм тогда
/// восстанавливает момент по часу года.
const META: &str = "last_hour.bin";

impl StateStore {
    pub fn new(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;
        let files = [dir.join("state_a.bin"), dir.join("state_b.bin")];
        let mut st = StateStore { dir, files, toggle: 0 };
        // следующая запись - в файл, который старше (или отсутствует)
        if let Some(newest) = st.candidates().first() {
            st.toggle = if *newest == st.files[0] { 1 } else { 0 };
        }
        Ok(st)
    }

    /// Существующие файлы состояния, новейший первым.
    pub fn candidates(&self) -> Vec<PathBuf> {
        let mut v: Vec<(PathBuf, std::time::SystemTime)> = self
            .files
            .iter()
            .filter_map(|f| fs::metadata(f).and_then(|m| m.modified()).ok().map(|t| (f.clone(), t)))
            .collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        v.into_iter().map(|(f, _)| f).collect()
    }

    /// Час последнего шага, если он записан.
    pub fn load_last_hour(&self) -> Option<i64> {
        let raw = fs::read(self.dir.join(META)).ok()?;
        Some(i64::from_le_bytes(raw.get(..8)?.try_into().ok()?))
    }

    pub fn save_last_hour(&self, hour: i64) -> Result<()> {
        let f = self.dir.join(META);
        let tmp = f.with_extension("tmp");
        fs::write(&tmp, hour.to_le_bytes())?;
        fs::rename(&tmp, &f)?;
        Ok(())
    }

    pub fn save(&mut self, bytes: &[u8]) -> Result<PathBuf> {
        let f = self.files[self.toggle % 2].clone();
        let tmp = f.with_extension("bin.tmp");
        {
            let mut fh = File::create(&tmp)?;
            fh.write_all(bytes)?;
            fh.sync_all()?;
        }
        fs::rename(&tmp, &f)?;
        if let Ok(d) = File::open(&self.dir) {
            let _ = d.sync_all();
        }
        self.toggle += 1;
        Ok(f)
    }
}
