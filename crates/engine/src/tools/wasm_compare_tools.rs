//! Interface diff between two WebAssembly modules: imports and exports by name, kind and exact
//! signature, reported as added, removed or changed. Built on the built-in WASM reader; nothing is
//! instantiated or run.

use super::native_common::{page, params_json, spec};
use super::static_input::{self, builder, format_err};
use crate::tool::{Ctx, Effects, Requires, ToolOutput, ToolSpec, Toolset, parse_params, schema_of};
use craft_anything_evidence::{Authority, Confidence, Limitation, Location, ToolError};
use craft_anything_formats::wasm::{self, WasmReport};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;

pub fn specs() -> Vec<ToolSpec> {
    vec![spec(
        "formats_wasm_compare",
        "Compare WebAssembly interfaces",
        "Compare the imports and exports of two WebAssembly modules (absolute paths): each import by module and name, each export by name, with its kind (func, table, memory, global, tag) and exact signature (function parameter/result types, memory and table limits, global type). Lists what was added, removed or changed, paged, with counts including unchanged entries; scope picks imports, exports or both. Static comparison only: matching signatures do not prove the modules behave the same.",
        "See how two versions of a WebAssembly module differ in what they import and export.",
        Toolset::Formats,
        Effects::EVIDENCE,
        Requires::Nothing,
        schema_of::<CompareWasmParams>,
        formats_wasm_compare,
    )]
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Scope {
    Imports,
    Exports,
    #[default]
    Both,
}

fn d_limit() -> usize {
    200
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CompareWasmParams {
    /// Absolute path of the older (left) .wasm module.
    pub left_path: String,
    /// Absolute path of the newer (right) .wasm module.
    pub right_path: String,
    /// imports, exports or both (default).
    #[serde(default)]
    pub scope: Scope,
    /// First change of the page.
    #[serde(default)]
    pub offset: usize,
    /// Changes per page (1–5000, default 200).
    #[serde(default = "d_limit")]
    pub limit: usize,
}

/// (direction, module, name) → every (kind, signature) occurrence, sorted.
type Interface = BTreeMap<(&'static str, Option<String>, String), Vec<(String, String)>>;

/// What an exported item looks like, resolved through the module's index spaces.
fn export_signature(r: &WasmReport, kind: &str, index: u32) -> String {
    let found = match kind {
        "func" => r
            .functions
            .iter()
            .find(|f| f.index == index)
            .map(|f| f.signature.clone())
            .or_else(|| r.imports.iter().filter(|i| i.kind == "func").nth(index as usize).map(|i| i.signature.clone())),
        "memory" => r
            .memories
            .iter()
            .find(|m| m.index == index)
            .map(|m| format!("{}..{}{}", m.initial, m.maximum.map(|x| x.to_string()).unwrap_or_default(), if m.shared { " shared" } else { "" })),
        "table" => r
            .tables
            .iter()
            .find(|t| t.index == index)
            .map(|t| format!("{} {}..{}", t.element_type, t.initial, t.maximum.map(|x| x.to_string()).unwrap_or_default())),
        "global" => r.globals.iter().find(|g| g.index == index).map(|g| format!("{}{}", if g.mutable { "mut " } else { "" }, g.value_type)),
        _ => None,
    };
    found.unwrap_or_else(|| format!("unresolved {kind} #{index}"))
}

fn interface(r: &WasmReport, scope: Scope) -> Interface {
    let mut m: Interface = BTreeMap::new();
    if scope != Scope::Exports {
        for i in &r.imports {
            m.entry(("import", Some(i.module.clone()), i.name.clone())).or_default().push((i.kind.to_string(), i.signature.clone()));
        }
    }
    if scope != Scope::Imports {
        for e in &r.exports {
            m.entry(("export", None, e.name.clone())).or_default().push((e.kind.to_string(), export_signature(r, e.kind, e.index)));
        }
    }
    for v in m.values_mut() {
        v.sort();
    }
    m
}

#[derive(Debug, Serialize)]
struct Change {
    direction: &'static str,
    module: Option<String>,
    name: String,
    /// added, removed or changed
    state: &'static str,
    left: Vec<Value>,
    right: Vec<Value>,
}

fn entries(v: Option<&Vec<(String, String)>>) -> Vec<Value> {
    v.map(|v| v.iter().map(|(k, s)| json!({"kind": k, "signature": s})).collect()).unwrap_or_default()
}

fn formats_wasm_compare(ctx: &mut Ctx<'_>, params: Value) -> Result<ToolOutput, ToolError> {
    let p: CompareWasmParams = parse_params(params)?;
    let left = static_input::load_path(&p.left_path, wasm::MAX_INPUT as u64)?;
    let right = static_input::load_path(&p.right_path, wasm::MAX_INPUT as u64)?;
    ctx.report(None, format!("Comparing {} with {}", left.name, right.name));
    let lr = wasm::inspect(&left.bytes).map_err(|e| format_err(e, "a WebAssembly module").with_details(json!({"side": "left"})))?;
    let rr = wasm::inspect(&right.bytes).map_err(|e| format_err(e, "a WebAssembly module").with_details(json!({"side": "right"})))?;
    ctx.cancel.check()?;
    let (li, ri) = (interface(&lr, p.scope), interface(&rr, p.scope));

    let mut changes = Vec::new();
    let mut unchanged = 0usize;
    let keys: std::collections::BTreeSet<_> = li.keys().chain(ri.keys()).collect();
    for key in keys {
        let (l, r) = (li.get(key), ri.get(key));
        let state = match (l, r) {
            (Some(a), Some(b)) if a == b => {
                unchanged += 1;
                continue;
            }
            (Some(_), Some(_)) => "changed",
            (Some(_), None) => "removed",
            _ => "added",
        };
        changes.push(Change { direction: key.0, module: key.1.clone(), name: key.2.clone(), state, left: entries(l), right: entries(r) });
    }
    let count = |s: &str| changes.iter().filter(|c| c.state == s).count();
    let (added, removed, changed) = (count("added"), count("removed"), count("changed"));
    let (rows, cov) = page(&changes, p.offset, p.limit);

    let mut lims = vec![Limitation::new(
        "static_interface_only",
        "Only declared names, kinds and types are compared; equal signatures do not prove the same behaviour, and host-side glue code is not examined.",
    )];
    for (side, r) in [("left", &lr), ("right", &rr)] {
        if !r.validation.valid {
            lims.push(Limitation::new(format!("{side}_invalid"), format!("The {side} module does not validate; its interface may be incomplete.")));
        }
        if r.truncated || r.parse_error.is_some() {
            lims.push(Limitation::new(format!("{side}_partial"), format!("The {side} module was not read completely; some entries may be missing.")));
        }
    }
    let summary = format!(
        "{} vs {}: {added} added, {removed} removed, {changed} changed, {unchanged} unchanged ({}).",
        left.name,
        right.name,
        match p.scope {
            Scope::Imports => "imports",
            Scope::Exports => "exports",
            Scope::Both => "imports and exports",
        }
    );
    let side = |i: &static_input::Input, r: &WasmReport| json!({"path": i.path, "sha256": i.subject.sha256, "bytes": i.bytes.len(), "imports": r.imports.len(), "exports": r.exports.len(), "valid": r.validation.valid});
    let b = builder(ctx, "formats_wasm_compare", right.subject.clone(), params_json(&p))
        .result(json!({
            "left": side(&left, &lr),
            "right": side(&right, &rr),
            "scope": p.scope,
            "counts": {"added": added, "removed": removed, "changed": changed, "unchanged": unchanged},
            "changes": rows,
        }))
        .summary(summary)
        .confidence(Confidence::Observed)
        .authority(Authority::OriginalFiles)
        .limitations(lims)
        .coverage(cov)
        .location(Location::new("path", left.path.display().to_string()).with_label("left"))
        .location(Location::new("path", right.path.display().to_string()).with_label("right"));
    Ok(ToolOutput::evidence(b.build()?))
}

#[cfg(test)]
mod tests {
    use super::super::native_common::test_support::{call, call_err};
    use crate::Engine;
    use craft_anything_testkit::{formats as fx, temp_file};
    use serde_json::json;

    /// A module importing env.log (i64)->() and exporting add and sub as (i32, i32)->(i64).
    fn right_module() -> Vec<u8> {
        let mut m = b"\0asm\x01\0\0\0".to_vec();
        let sec = |id: u8, body: &[u8], m: &mut Vec<u8>| {
            m.push(id);
            m.push(body.len() as u8);
            m.extend_from_slice(body);
        };
        sec(1, &[2, 0x60, 1, 0x7e, 0, 0x60, 2, 0x7f, 0x7f, 1, 0x7e], &mut m);
        sec(2, &[1, 3, b'e', b'n', b'v', 3, b'l', b'o', b'g', 0, 0], &mut m);
        sec(3, &[1, 1], &mut m);
        sec(7, &[2, 3, b'a', b'd', b'd', 0, 1, 3, b's', b'u', b'b', 0, 1], &mut m);
        sec(10, &[1, 4, 0, 0x42, 0, 0x0b], &mut m);
        m
    }

    #[test]
    fn reports_added_removed_changed() {
        let e = Engine::new(crate::testing::session());
        let l = temp_file("old.wasm", &fx::wasm_module());
        let r = temp_file("new.wasm", &right_module());
        let v = call(&e, "formats_wasm_compare", json!({"left_path": l, "right_path": r}));
        let res = &v["result"];
        assert_eq!(res["right"]["valid"], true, "{res}");
        assert_eq!(res["counts"], json!({"added": 1, "removed": 2, "changed": 2, "unchanged": 0}));
        let ch = res["changes"].as_array().unwrap();
        let find = |dir: &str, name: &str| ch.iter().find(|c| c["direction"] == dir && c["name"] == name).unwrap().clone();
        assert_eq!(find("export", "sub")["state"], "added");
        assert_eq!(find("export", "memory")["state"], "removed");
        let add = find("export", "add");
        assert_eq!(add["state"], "changed");
        assert_ne!(add["left"][0]["signature"], add["right"][0]["signature"]);
        assert_eq!(find("import", "log")["module"], "env");

        let v = call(&e, "formats_wasm_compare", json!({"left_path": l, "right_path": l, "scope": "exports"}));
        assert_eq!(v["result"]["counts"], json!({"added": 0, "removed": 0, "changed": 0, "unchanged": 3}));

        let bad = temp_file("bad.wasm", b"not wasm");
        call_err(&e, "formats_wasm_compare", json!({"left_path": l, "right_path": bad}));
    }
}
