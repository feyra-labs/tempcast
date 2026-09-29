//! Атомарная запись состояния в два чередующихся файла.
//!
//! Запись: временный файл, fsync, rename поверх файла очереди, fsync каталога. Отключение
//! питания в любой момент оставляет хотя бы один целый файл: второй файл очереди
//! содержит предыдущий час. Имена файлов те же, что у рантайма на Python, состояния
//! взаимозаменяемы.
//!
//! Чтение: порядок задаёт само содержимое, а не время изменения файла. Две записи подряд
//! могут получить одинаковую отметку времени, и сортировка по ней выбрала бы файл
//! произвольно. Свежее то состояние, у которого позже абсолютный час последнего шага;
//! состояние без шагов старше любого другого. Время изменения остаётся тай-брейком для
//! неразличимого содержимого.
use std::cmp::Ordering;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::state::Snapshot;
use crate::Result;

/// Порядок свежести двух состояний по их содержимому: более свежее идёт первым.
pub fn newer_first(a: &Snapshot, b: &Snapshot) -> Ordering {
    b.last_hour.cmp(&a.last_hour)
}

pub struct StateStore {
    dir: PathBuf,
    files: [PathBuf; 2],
    toggle: usize,
}

impl StateStore {
    pub fn new(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;
        let files = [dir.join("state_a.bin"), dir.join("state_b.bin")];
        let mut st = StateStore { dir, files, toggle: 0 };
        // следующая запись - в файл, который старше или отсутствует
        if let Some(newest) = st.candidates().first() {
            st.toggle = if *newest == st.files[0] { 1 } else { 0 };
        }
        Ok(st)
    }

    /// Существующие файлы состояния, новейший по времени изменения первым.
    pub fn candidates(&self) -> Vec<PathBuf> {
        let mut v: Vec<(PathBuf, std::time::SystemTime)> = self
            .files
            .iter()
            .filter_map(|f| fs::metadata(f).and_then(|m| m.modified()).ok().map(|t| (f.clone(), t)))
            .collect();
        v.sort_by_key(|a| std::cmp::Reverse(a.1));
        v.into_iter().map(|(f, _)| f).collect()
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
