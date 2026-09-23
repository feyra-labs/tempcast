//! Пиковая и текущая резидентная память процесса (Linux: /proc/self/status).

fn status_kib(key: &str) -> Option<u64> {
    let text = std::fs::read_to_string("/proc/self/status").ok()?;
    text.lines()
        .find(|l| l.starts_with(key))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse().ok())
}

/// Пиковый RSS процесса (VmHWM), байт. None вне Linux.
pub fn peak_rss_bytes() -> Option<u64> {
    status_kib("VmHWM:").map(|k| k * 1024)
}

/// Текущий RSS (VmRSS), байт.
pub fn rss_bytes() -> Option<u64> {
    status_kib("VmRSS:").map(|k| k * 1024)
}
