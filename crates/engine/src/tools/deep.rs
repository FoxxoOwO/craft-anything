//! The Deep-mode hook the native tools consult.
//!
//! When the session is **not** clean-room (Deep mode is on) and Ghidra is installed, the native
//! decompile path prefers Ghidra's result through the *same* tool id (`native_function_pseudocode`),
//! recording `environment.engine = "ghidra"` and a higher confidence. When Deep mode is off, or
//! Ghidra is absent, or anything about the bridge fails, this returns `None` and the caller falls
//! back to the built-in clean-room lifter — the clean-room path is never changed.
//!
//! The Ghidra session is launched lazily on first use and cached on the target, keyed by the
//! target's content and an analysis settings hash. A launch that fails is remembered so we don't
//! retry it on every call.

use crate::settings::Settings;
use crate::tool::{Ctx, ToolOutput};
use craft_anything_engines::ghidra::{self, DeepDecompiler, DeepFunction};
use craft_anything_evidence::{Confidence, Environment, EvidenceBuilder, Limitation, Location, ToolError, host_triple};
use std::sync::Arc;
use std::time::Duration;

/// The analysis settings hash for the default Deep profile. Part of the project cache key, so a
/// different pinned language re-analyses rather than reusing the auto-detected project.
fn settings_hash(language_id: Option<&str>) -> String {
    let seed = format!("craft-anything-ghidra-fast-profile-v2|lang={}", language_id.unwrap_or(""));
    craft_anything_evidence::sha256_hex(seed.as_bytes())
}

/// A per-target Ghidra language override set by `core_target_open` (wins over the global setting).
pub struct LanguageOverride(pub String);

/// Cached decision for one target: a live provider, or why there isn't one.
struct DeepSlot {
    provider: Result<Arc<dyn DeepDecompiler>, String>,
}

/// How long Ghidra's one-time analysis of a target may take by default: a big game can take hours,
/// and it's saved, so later sessions open it in moments.
const ANALYSIS_LIMIT_SECS: u64 = 8 * 3600;

/// Why Ghidra wasn't used, as said to the builder. These say only that; the caller adds what it did
/// instead (the built-in decompiler's answer, or that no function was found).
/// Deep mode is on but Ghidra isn't installed.
pub const NOT_INSTALLED: &str = "Deep mode is on, but Ghidra isn't installed. The person can get Ghidra from Settings → Engines.";

/// Get (or lazily create) the deep provider for the active target. `Err(Some(why))`: Deep mode is
/// on but Ghidra can't be used; `Err(None)`: Clean room (never launches). Either way the caller
/// uses the built-in engine.
fn provider_for_active(ctx: &mut Ctx<'_>) -> Result<Arc<dyn DeepDecompiler>, Option<String>> {
    if ctx.session.clean_room() {
        return Err(None);
    }
    // Already decided for this target?
    if let Ok(t) = ctx.session.active()
        && let Some(slot) = t.ext.get::<DeepSlot>()
    {
        return slot.provider.clone().map_err(Some);
    }

    // The target's bytes and identity.
    let (bytes, sha256) = match ctx.session.active() {
        Ok(t) => (t.target.bytes.clone(), t.target.sha256.clone()),
        Err(_) => return Err(None),
    };
    if bytes.is_empty() {
        return remember(ctx, Err("Ghidra needs the program's bytes, which aren't loaded, so the built-in decompiler was used.".into()));
    }

    // A per-target language override (from core_target_open) wins over the global setting.
    let language = ctx.session.active().ok().and_then(|t| t.ext.get::<LanguageOverride>().map(|o| o.0.clone()));
    // Not remembered: once the person installs Ghidra, the next call uses it.
    let Some((cfg, digest)) = config(&ctx.session.settings, language) else {
        return Err(Some(NOT_INSTALLED.into()));
    };
    let cancel = ctx.cancel.clone();
    let cancel_fn = move || cancel.is_cancelled();
    let progress = ctx.progress;
    let report = |f: f32, m: &str| (progress)(crate::tool::Progress { fraction: Some(f), message: m.to_string() });

    match ghidra::Session::start(&cfg, bytes.as_ref(), &sha256, &digest, &cancel_fn, &report) {
        Ok(session) => remember(ctx, Ok(Arc::new(session))),
        // Stopped by the caller: try again next time rather than remembering a failure.
        Err(ghidra::GhidraError::Cancelled) => Err(Some("Ghidra was stopped before it was ready.".into())),
        Err(e) => {
            log::warn!("Deep mode (Ghidra) unavailable; using the built-in engine: {e}");
            remember(ctx, Err(format!("Ghidra couldn't open this program ({e}).")))
        }
    }
}

/// Ghidra's settings for a target (and its analysis settings hash), the same for the app and the
/// builder's tool server so both find one shared analysis. `None`: Ghidra isn't installed.
fn config(settings: &Settings, language_override: Option<String>) -> Option<(ghidra::Config, String)> {
    let g = &settings.engines.ghidra;
    let language_id = language_override.or_else(|| g.language_id.clone());
    let digest = settings_hash(language_id.as_deref());
    let cfg = ghidra::Config {
        analyze_headless: settings.ghidra_launcher()?,
        java_home: settings.ghidra_java(),
        startup_timeout: Duration::from_secs(g.startup_timeout_secs.unwrap_or(ANALYSIS_LIMIT_SECS)),
        store_dir: settings.engine_store_dir(),
        language_id,
        compiler_spec: g.compiler_spec.clone(),
        memory_gb: g.memory_gb,
    };
    Some((cfg, digest))
}

/// Analyse a program with Ghidra ahead of time (Deep mode), so the builder's first decompile opens
/// a saved analysis instead of waiting hours for one. Shares the analysis with any other process
/// doing the same. `progress` gets a line now and then.
pub fn prepare(settings: &Settings, bytes: &[u8], sha256: &str, progress: &(dyn Fn(&str) + Send + Sync)) -> Result<(), String> {
    let (cfg, digest) = config(settings, None).ok_or_else(|| NOT_INSTALLED.to_string())?;
    let never = || false;
    ghidra::analyse(&cfg, bytes, sha256, &digest, &never, &|_, m: &str| progress(m)).map(|_| ()).map_err(|e| e.to_string())
}

fn remember(ctx: &mut Ctx<'_>, provider: Result<Arc<dyn DeepDecompiler>, String>) -> Result<Arc<dyn DeepDecompiler>, Option<String>> {
    if let Ok(t) = ctx.session.active_mut() {
        t.ext.insert(Arc::new(DeepSlot { provider: provider.clone() }));
    }
    provider.map_err(Some)
}

/// Decompile `entry` with the deep engine if available. `Err(why)` → the caller uses the built-in
/// lifter and passes on why Ghidra wasn't used (`None` in Clean room). `Ok(Err(..))` only for a
/// cancellation the caller should surface.
pub fn pseudo_code(ctx: &mut Ctx<'_>, entry: u64, params: serde_json::Value) -> Result<Result<ToolOutput, ToolError>, Option<String>> {
    let provider = provider_for_active(ctx)?;
    match provider.decompile(entry) {
        Ok(Some(df)) => Ok(build_evidence(ctx, provider.as_ref(), df, params)),
        Ok(None) => Err(Some("Ghidra had no answer for this function.".into())),
        Err(ghidra::GhidraError::Cancelled) => Ok(Err(ToolError::cancelled())),
        Err(e) => {
            log::warn!("Ghidra decompile failed; falling back to the built-in engine: {e}");
            // A dead bridge (Ghidra crashed or was killed) is started again on the next call; the
            // saved analysis makes that quick.
            if matches!(e, ghidra::GhidraError::Connection(_) | ghidra::GhidraError::Protocol(_))
                && let Ok(t) = ctx.session.active_mut()
            {
                t.ext.remove::<DeepSlot>();
            }
            Err(Some(match e {
                // The bridge looks up the function containing the address and, when its analysis
                // has none, tries to make one there; the message says why it couldn't.
                ghidra::GhidraError::Bridge { ref code, ref message } if code == "not_found" => {
                    format!("Ghidra's analysis has no function at this address and it couldn't make one: {message}.")
                }
                e => format!("Ghidra couldn't decompile this function ({e})."),
            }))
        }
    }
}

fn build_evidence(ctx: &Ctx<'_>, provider: &dyn DeepDecompiler, df: DeepFunction, params: serde_json::Value) -> Result<ToolOutput, ToolError> {
    let subject = ctx.session.active()?.target.subject();
    let env = Environment {
        clean_room: false,
        engine: provider.engine().to_string(),
        engine_version: provider.engine_version(),
        host: host_triple(),
        language_id: provider.language_id(),
    };
    let line_count = df.pseudocode.as_deref().map(|p| p.lines().count()).unwrap_or(0);
    let mut e = EvidenceBuilder::new("native_function_pseudocode", subject, env)
        .params(params)
        .result(serde_json::json!({
            "function": df.name,
            "entry": format!("{:#x}", df.entry),
            "signature": df.signature,
            "pseudo_code": df.pseudocode,
            "engine": provider.engine(),
        }))
        .summary(format!("Pseudo-C for {} from the Deep engine ({line_count} lines).", df.name))
        // Deep decompilation is derived by a mature external engine: stronger than the built-in
        // inferred lifter, but still a reading aid rather than source.
        .confidence(Confidence::Derived)
        .target_content(true)
        .limitation(Limitation::new("deep_engine", "the pseudo-C comes from the external Deep engine (Ghidra), not the original source"))
        .location(Location::address(df.entry));
    if df.created {
        e = e.limitation(Limitation::new(
            "made_on_demand",
            "Ghidra's analysis had no function here, so it made one starting at this address (for this session, never saved); its bounds come from following the code from there",
        ));
    }
    Ok(ToolOutput::evidence(e.build()?))
}
