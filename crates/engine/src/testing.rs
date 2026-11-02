//! Helpers for tests in this and other crates. Tests never touch the real machine
//! (docs/pitfalls.md): use [`session`] instead of `Session::default()`, which points at this
//! computer's config folder and engine store.

use crate::Session;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A fresh folder under the system temp folder, unique to this test process and call.
pub fn temp_dir() -> PathBuf {
    static N: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!("craft-anything-test-{}-{}", std::process::id(), N.fetch_add(1, Ordering::SeqCst)));
    let _ = std::fs::create_dir_all(&dir);
    dir
}

/// A session whose config, plugin caches, engine store and projects live in a fresh temp folder.
/// It is in Deep mode, so tools may look inside the sample targets tests open; a test of Clean room
/// turns `deep_mode` off.
pub fn session() -> Session {
    let mut s = Session::isolated(&temp_dir());
    s.settings.deep_mode = true;
    s
}
