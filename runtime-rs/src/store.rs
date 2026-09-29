//! Атомарная запись состояния в два чередующихся файла.
//!
//! Запись: временный файл, fsync, rename поверх файла очереди, fsync каталога. Отключение
//! питания в любой момент оставляет хотя бы один целый файл: второй файл очереди
//! содержит предыдущий час. Имена файлов те же, что у хоста на Python, состояния
//! взаимозаменяемы.
//!
//! Чтение: порядок задаёт само содержимое, а не время изменения файла. Свежее то
//! состояние, у которого позже абсолютный час последнего шага; состояние без шагов
//! старше любого состояния с шагами; файл, который не разбирается, идёт в конец. Время
//! изменения решает только при равенстве. Следующая запись идёт в файл, который не был
//! восстановлен, чтобы восстановленное состояние оставалось целым до конца записи.
use std::cmp::Reverse;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::Result;

pub const STATE_FILES: [&str; 2] = ["state_a.bin", "state_b.bin"];

pub struct StateStore {
    dir: PathBuf,
    files: [PathBuf; 2],
    toggle: usize,
}

impl StateStore {
    pub fn new(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;
        let files = STATE_FILES.map(|f| dir.join(f));
        Ok(StateStore { dir, files, toggle: 0 })
    }

    /// Существующие файлы состояния, новейший по времени изменения первым.
    pub fn candidates(&self) -> Vec<PathBuf> {
        let mut v: Vec<(PathBuf, std::time::SystemTime)> = self
            .files
            .iter()
            .filter_map(|f| fs::metadata(f).and_then(|m| m.modified()).ok().map(|t| (f.clone(), t)))
            .collect();
        v.sort_by_key(|a| Reverse(a.1));
        v.into_iter().map(|(f, _)| f).collect()
    }

    /// Файлы состояния с содержимым, свежий первым.
    ///
    /// `last_hour` разбирает байты состояния и возвращает абсолютный час последнего шага
    /// или None для состояния без шагов; ошибка разбора отправляет файл в конец.
    pub fn ordered<F>(&self, last_hour: F) -> Vec<(PathBuf, Vec<u8>)>
    where
        F: Fn(&[u8]) -> Result<Option<i64>>,
    {
        let mut v: Vec<((u8, Reverse<i64>), PathBuf, Vec<u8>)> = self
            .candidates()
            .into_iter()
            .filter_map(|f| fs::read(&f).ok().map(|b| (f, b)))
            .map(|(f, b)| {
                let key = match last_hour(&b) {
                    Ok(Some(h)) => (0, Reverse(h)),
                    Ok(None) => (1, Reverse(0)),
                    Err(_) => (2, Reverse(0)),
                };
                (key, f, b)
            })
            .collect();
        v.sort_by_key(|a| a.0);
        v.into_iter().map(|(_, f, b)| (f, b)).collect()
    }

    /// Следующая запись идёт в другой файл, чтобы восстановленный остался целым.
    pub fn mark_restored(&mut self, path: &Path) {
        self.toggle = if path == self.files[0] { 1 } else { 0 };
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;
    use std::time::{Duration, SystemTime};

    fn dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("mayak-store-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        d
    }

    /// Разбор для теста: первый байт 1 - час в следующих восьми, 0 - шагов не было.
    fn last_hour(b: &[u8]) -> Result<Option<i64>> {
        match b.first() {
            Some(1) => Ok(Some(i64::from_le_bytes(b[1..9].try_into().unwrap()))),
            Some(0) => Ok(None),
            _ => Err(Error::new("не состояние")),
        }
    }

    fn put(path: &Path, bytes: &[u8], mtime: u64) {
        fs::write(path, bytes).unwrap();
        let f = File::options().write(true).open(path).unwrap();
        f.set_modified(SystemTime::UNIX_EPOCH + Duration::from_secs(mtime)).unwrap();
    }

    fn state(h: i64) -> Vec<u8> {
        let mut v = vec![1u8];
        v.extend_from_slice(&h.to_le_bytes());
        v
    }

    fn names(v: &[(PathBuf, Vec<u8>)]) -> Vec<String> {
        v.iter().map(|(f, _)| f.file_name().unwrap().to_string_lossy().into_owned()).collect()
    }

    #[test]
    fn content_wins_over_mtime() {
        let d = dir("content");
        let st = StateStore::new(&d).unwrap();
        put(&d.join("state_a.bin"), &state(1000), 2_000_000_000);
        put(&d.join("state_b.bin"), &state(2000), 1_000_000_000);
        assert_eq!(names(&st.ordered(last_hour)), ["state_b.bin", "state_a.bin"]);
        put(&d.join("state_b.bin"), &[0u8], 1_000_000_000);
        assert_eq!(names(&st.ordered(last_hour)), ["state_a.bin", "state_b.bin"]);
        put(&d.join("state_a.bin"), b"garbage", 3_000_000_000);
        assert_eq!(names(&st.ordered(last_hour)), ["state_b.bin", "state_a.bin"]);
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn equal_hours_fall_back_to_mtime() {
        let d = dir("tie");
        let st = StateStore::new(&d).unwrap();
        put(&d.join("state_a.bin"), &state(7), 1_000_000_000);
        put(&d.join("state_b.bin"), &state(7), 1_000_000_100);
        assert_eq!(names(&st.ordered(last_hour)), ["state_b.bin", "state_a.bin"]);
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn next_write_goes_to_the_other_file() {
        let d = dir("toggle");
        let mut st = StateStore::new(&d).unwrap();
        st.mark_restored(&d.join("state_a.bin"));
        assert_eq!(st.save(&state(1)).unwrap(), d.join("state_b.bin"));
        assert_eq!(st.save(&state(2)).unwrap(), d.join("state_a.bin"));
        st.mark_restored(&d.join("state_b.bin"));
        assert_eq!(st.save(&state(3)).unwrap(), d.join("state_a.bin"));
        let left: Vec<_> = fs::read_dir(&d).unwrap().map(|e| e.unwrap().file_name()).collect();
        assert!(left.iter().all(|n| !n.to_string_lossy().ends_with(".tmp")));
        let _ = fs::remove_dir_all(&d);
    }
}
