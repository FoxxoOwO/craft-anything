//! Per-function tools: info, assembly, callers, callees, references, pseudo-C, instructions,
//! the containing function of an address, one instruction in detail, call-target resolution,
//! and best-effort data typing.

use super::native_common::{ev, hex, name_of, native, params_json, resolve_function, select_document, spec, user_names};
use crate::tool::{Ctx, Effects, Requires, ToolOutput, ToolSpec, Toolset, parse_params, schema_of};
use craft_anything_analysis::native as an;
use craft_anything_disasm::pseudo::lift;
use craft_anything_disasm::{Flow, XrefKind};
use craft_anything_evidence::{Confidence, Coverage, Limitation, Location, ToolError};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// The decompile tool's id, for code that decides on whether a function was decompiled.
pub const PSEUDOCODE_TOOL: &str = "native_function_pseudocode";

pub fn specs() -> Vec<ToolSpec> {
    use Toolset::Native;
    let e = Effects::EVIDENCE;
    let t = Requires::Target;
    vec![
        spec(
            "native_function_show",
            "Function info",
            "Entry, end, size, basic blocks, body ranges, call counts, jump tables and how the function was found.",
            "Basic facts about one function.",
            Native,
            e,
            t,
            schema_of::<ProcParams>,
            native_function_show,
        ),
        spec(
            "native_function_disassemble",
            "Function assembly",
            "The function's assembly as one text listing (address, bytes, instruction, names).",
            "See the machine instructions of a function.",
            Native,
            e,
            t,
            schema_of::<ProcParams>,
            native_function_disassemble,
        ),
        spec(
            "native_function_callees_list",
            "What it calls",
            "Direct callees of a function, with resolved names and imports.",
            "See which functions this one calls.",
            Native,
            e,
            t,
            schema_of::<ProcParams>,
            native_function_callees_list,
        ),
        spec(
            "native_function_callers_list",
            "Who calls it",
            "Call sites that call (or tail-jump to) this function.",
            "See which functions call this one.",
            Native,
            e,
            t,
            schema_of::<ProcParams>,
            native_function_callers_list,
        ),
        spec(
            "native_function_references_list",
            "Function references",
            "Raw reference edges of a function: incoming (to its entry) or outgoing (from its code).",
            "See everything that points into or out of a function.",
            Native,
            e,
            t,
            schema_of::<RefParams>,
            native_function_references_list,
        ),
        spec(
            PSEUDOCODE_TOOL,
            "Decompile",
            "Structured pseudo-C from the built-in clean-room lifter: register-level variables, if/else and loops from the control-flow graph, resolved call names, inline string literals. Honest about its limits.",
            "Read a function as C-like code.",
            Native,
            e | Effects::CARRIES_TARGET_CONTENT,
            t,
            schema_of::<ProcParams>,
            native_function_pseudocode,
        ),
        spec(
            "native_function_address_show",
            "Function address",
            "Resolve a function name to its canonical entry address.",
            "Find where a function starts.",
            Native,
            e,
            t,
            schema_of::<ProcParams>,
            native_function_address_show,
        ),
        spec(
            "native_function_instructions_list",
            "Function instructions",
            "Every instruction of a function with bytes, mnemonic, operands, flow and referenced addresses.",
            "List every instruction of a function.",
            Native,
            e,
            t,
            schema_of::<ProcParams>,
            native_function_instructions_list,
        ),
        spec(
            "native_function_find",
            "Which function is this in",
            "The function whose code contains an address.",
            "Find the function an address belongs to.",
            Native,
            e,
            t,
            schema_of::<AddrParams>,
            native_function_find,
        ),
        spec(
            "native_instruction_show",
            "Inspect instruction",
            "The instruction (or data) at an address: operands, flow, registers read and written, references, containing function.",
            "Look closely at one instruction.",
            Native,
            e,
            t,
            schema_of::<AddrParams>,
            native_instruction_show,
        ),
        spec(
            "native_call_site_resolve",
            "Resolve a call",
            "Resolve the target(s) of one call or jump site: direct target, import, jump-table candidates, or unresolved.",
            "Find out what a call actually calls.",
            Native,
            e,
            t,
            schema_of::<AddrParams>,
            native_call_site_resolve,
        ),
        spec(
            "native_data_show",
            "Inspect data",
            "Best-effort typing of data at an address: string, pointer (to a function, string or data), pointer table, integer, or a requested type (u8…u64, i32, i64, f32, f64, pointer, cstring, utf16).",
            "Work out what kind of data is stored somewhere.",
            Native,
            e,
            t,
            schema_of::<DataTypeParams>,
            native_data_show,
        ),
    ]
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ProcParams {
    /// Open target (document) to use; the active one when omitted.
    #[serde(default)]
    pub document: Option<String>,
    /// Function name or address (0x-hex; any address inside it works).
    pub function: String,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    Incoming,
    #[default]
    Outgoing,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RefParams {
    /// Open target (document) to use; the active one when omitted.
    #[serde(default)]
    pub document: Option<String>,
    /// Function name or address.
    pub function: String,
    /// incoming or outgoing (default).
    #[serde(default)]
    pub direction: Direction,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AddrParams {
    /// Open target (document) to use; the active one when omitted.
    #[serde(default)]
    pub document: Option<String>,
    /// Virtual address (0x-hex).
    pub address: String,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DataTypeParams {
    /// Open target (document) to use; the active one when omitted.
    #[serde(default)]
    pub document: Option<String>,
    /// Address of the data (0x-hex).
    #[serde(default)]
    pub address: Option<String>,
    /// Type to read it as: u8, u16, u32, u64, i32, i64, f32, f64, pointer, cstring, utf16. Inferred when omitted.
    #[serde(rename = "type", default)]
    pub ty: Option<String>,
}

macro_rules! with_proc {
    ($ctx:ident, $params:ident, $P:ty, |$p:ident, $st:ident, $t:ident, $f:ident| $body:block) => {{
        let $p: $P = parse_params($params)?;
        select_document($ctx, &$p.document)?;
        let $st = native($ctx)?;
        let $t = $ctx.session.active()?;
        let $f = resolve_function($t, &$st.prog, &$p.function)?;
        $body
    }};
}

fn native_function_show(ctx: &mut Ctx<'_>, params: Value) -> Result<ToolOutput, ToolError> {
    with_proc!(ctx, params, ProcParams, |p, st, t, f| {
        let user = user_names(t);
        let names = |a: u64| user.get(&a).cloned();
        let info = an::function_info(&st.prog, f, &names);
        let sig = lift(&st.img, &st.prog, f, &names).signature;
        let mut info = info;
        info["signature"] = json!(sig);
        let summary = format!("{} at {:#x}: {} bytes in {} blocks.", info["name"].as_str().unwrap_or(""), f.entry, f.size(), f.blocks.len());
        let e = ev(ctx, "native_function_show", params_json(&p))?
            .result(info)
            .summary(summary)
            .confidence(Confidence::Derived)
            .location(Location::address(f.entry))
            .build()?;
        Ok(ToolOutput::evidence(e))
    })
}

fn native_function_disassemble(ctx: &mut Ctx<'_>, params: Value) -> Result<ToolOutput, ToolError> {
    with_proc!(ctx, params, ProcParams, |p, st, t, f| {
        let insns = st.prog.instructions(&st.img, f, true);
        let mut text = String::new();
        let mut last_block = None;
        for i in &insns {
            let block = f.block_at(i.addr).map(|b| b.start);
            if block != last_block {
                if let Some(n) = name_of(t, &st.prog, i.addr) {
                    text.push_str(&format!("{n}:\n"));
                } else if block == Some(i.addr) && i.addr != f.entry {
                    text.push_str(&format!("loc_{:x}:\n", i.addr));
                }
                last_block = block;
            }
            let bytes = st.img.read(i.addr, usize::from(i.len)).map(hex::encode).unwrap_or_default();
            let mut line = format!("  {:#x}  {:<20} {}", i.addr, bytes, i.text());
            let note = match i.flow {
                Flow::Call { target } | Flow::Jump { target } | Flow::Cond { target } => name_of(t, &st.prog, target),
                Flow::IndirectCall { slot: Some(s) } | Flow::IndirectJump { slot: Some(s) } => name_of(t, &st.prog, s),
                _ => i.refs.iter().find_map(|r| {
                    st.prog.string_at(*r).map(|s| format!("\"{}\"", s.text.chars().take(60).collect::<String>())).or_else(|| name_of(t, &st.prog, *r))
                }),
            };
            if let Some(c) = t.annotations.inline_comments.get(&i.addr) {
                line.push_str(&format!("  ; {c}"));
            } else if let Some(n) = note {
                line.push_str(&format!("  ; {n}"));
            }
            if let Some(c) = t.annotations.comments.get(&i.addr) {
                text.push_str(&format!("  ; {c}\n"));
            }
            text.push_str(&line);
            text.push('\n');
        }
        let e = ev(ctx, "native_function_disassemble", params_json(&p))?
            .result(json!({"function": f.name, "entry": hex(f.entry), "assembly": text, "instructions": insns.len()}))
            .summary(format!("{} instructions of {}.", insns.len(), f.name))
            .coverage(Coverage::complete(insns.len() as u64))
            .location(Location::address(f.entry))
            .build()?;
        Ok(ToolOutput::evidence(e))
    })
}

fn native_function_callees_list(ctx: &mut Ctx<'_>, params: Value) -> Result<ToolOutput, ToolError> {
    with_proc!(ctx, params, ProcParams, |p, st, t, f| {
        let user = user_names(t);
        let names = |a: u64| user.get(&a).cloned();
        let list = an::callees(&st.prog, f, &names);
        let unresolved = list.iter().filter(|c| c["indirect"] == true).count();
        let mut b = ev(ctx, "native_function_callees_list", params_json(&p))?
            .result(json!({"function": f.name, "entry": hex(f.entry), "callees": list}))
            .summary(format!("{} makes {} calls ({unresolved} unresolved).", f.name, list.len()))
            .confidence(Confidence::Derived)
            .coverage(Coverage::complete(list.len() as u64))
            .location(Location::address(f.entry));
        if unresolved > 0 {
            b = b.limitation(Limitation::new("indirect_calls_unresolved", "calls through registers are listed without a target"));
        }
        Ok(ToolOutput::evidence(b.build()?))
    })
}

fn native_function_callers_list(ctx: &mut Ctx<'_>, params: Value) -> Result<ToolOutput, ToolError> {
    with_proc!(ctx, params, ProcParams, |p, st, t, f| {
        let user = user_names(t);
        let names = |a: u64| user.get(&a).cloned();
        let list = an::callers(&st.prog, f, &names);
        let pointers =
            st.prog.xrefs_to.get(&f.entry).map(|v| v.iter().filter(|x| x.kind == XrefKind::Pointer || x.kind == XrefKind::Data).count()).unwrap_or(0);
        let mut b = ev(ctx, "native_function_callers_list", params_json(&p))?
            .result(json!({"function": f.name, "entry": hex(f.entry), "callers": list, "address_taken": pointers}))
            .summary(format!(
                "{} is called from {} places{}.",
                f.name,
                list.len(),
                if pointers > 0 { format!(" and its address is taken {pointers} times") } else { String::new() }
            ))
            .confidence(Confidence::Derived)
            .coverage(Coverage::complete(list.len() as u64))
            .location(Location::address(f.entry))
            .limitation(Limitation::new("direct_calls_only", "callers that reach it through function pointers are not listed (see address_taken)"));
        if list.is_empty() {
            b = b.limitation(Limitation::new("no_static_callers", "no direct callers found; it may be an entry point, a callback or only reached indirectly"));
        }
        Ok(ToolOutput::evidence(b.build()?))
    })
}

fn native_function_references_list(ctx: &mut Ctx<'_>, params: Value) -> Result<ToolOutput, ToolError> {
    with_proc!(ctx, params, RefParams, |p, st, t, f| {
        let mut rows = Vec::new();
        match p.direction {
            Direction::Incoming => {
                for x in st.prog.xrefs_to.get(&f.entry).map(Vec::as_slice).unwrap_or(&[]) {
                    if st.prog.function_containing(x.from).is_some_and(|c| c.entry == f.entry) && x.kind == XrefKind::Jump {
                        continue;
                    }
                    rows.push(json!({"from": hex(x.from), "to": hex(x.to), "kind": x.kind, "function": st.prog.function_containing(x.from).map(|c| name_of(t, &st.prog, c.entry))}));
                }
            }
            Direction::Outgoing => {
                for (from, xs) in st.prog.xrefs_from.range(f.entry..f.end.max(f.entry + 1)) {
                    if !f.contains(*from) {
                        continue;
                    }
                    for x in xs {
                        if x.kind == XrefKind::Jump && f.contains(x.to) {
                            continue;
                        }
                        rows.push(json!({"from": hex(x.from), "to": hex(x.to), "kind": x.kind, "name": name_of(t, &st.prog, x.to)}));
                    }
                }
            }
        }
        let e = ev(ctx, "native_function_references_list", params_json(&p))?
            .result(json!({"function": f.name, "entry": hex(f.entry), "direction": p.direction, "references": rows}))
            .summary(format!("{} {:?} references for {}.", rows.len(), p.direction, f.name).to_lowercase())
            .confidence(Confidence::Derived)
            .coverage(Coverage::complete(rows.len() as u64))
            .location(Location::address(f.entry))
            .build()?;
        Ok(ToolOutput::evidence(e))
    })
}

fn native_function_pseudocode(ctx: &mut Ctx<'_>, params: Value) -> Result<ToolOutput, ToolError> {
    let p: ProcParams = parse_params(params)?;
    select_document(ctx, &p.document)?;
    // Resolve the function entry in a scoped borrow so the deep hook can take `&mut ctx`. The
    // built-in analysis stops short on a huge program (a 200 MB game has far more functions than
    // it maps), so in Deep mode an address it can't place (0x…, FUN_…, sub_…) goes to Ghidra,
    // which analysed the whole program and finds the function containing it.
    let (entry, unknown) = {
        let st = native(ctx)?;
        let t = ctx.session.active()?;
        match resolve_function(t, &st.prog, &p.function) {
            Ok(f) => (f.entry, None),
            Err(e) => match ghidra_address(&p.function).filter(|_| !ctx.session.clean_room()) {
                Some(a) => (a, Some(e)),
                None => return Err(e),
            },
        }
    };
    // Deep mode (Ghidra) is preferred through this same tool id when it is on and available;
    // otherwise, and whenever anything about it fails, we use the built-in clean-room lifter.
    let why_not_deep = match super::deep::pseudo_code(ctx, entry, params_json(&p)) {
        Ok(result) => return result,
        Err(why) => why,
    };
    if let Some(e) = unknown {
        return Err(match why_not_deep {
            Some(why) => e.with_remediation(format!(
                "{why} The built-in analysis has no function there either: it may be data, or code reached only through a table of pointers. Try native_function_find, or xrefs to the address to find the code that uses it."
            )),
            None => e,
        });
    }
    // Built-in clean-room lifter.
    let st = native(ctx)?;
    let t = ctx.session.active()?;
    let f = resolve_function(t, &st.prog, &p.function)?;
    let user = user_names(t);
    let names = |a: u64| user.get(&a).cloned();
    let ps = lift(&st.img, &st.prog, f, &names);
    let e = ev(ctx, "native_function_pseudocode", params_json(&p))?
        .result(json!({"function": name_of(t, &st.prog, f.entry), "entry": hex(f.entry), "signature": ps.signature, "pseudo_code": ps.text, "calls": ps.calls, "strings": ps.strings, "structured": ps.structured, "gotos": ps.gotos, "inline_asm": ps.asm_fallbacks, "engine": "clean-room lifter"}))
        .summary(format!("Pseudo-C for {} ({} lines{}).", f.name, ps.text.lines().count(), if ps.structured { "" } else { ", partly unstructured" }))
        .confidence(Confidence::Inferred)
        .target_content(true)
        .limitations(ps.limitations.iter().map(|(c, m)| Limitation::new(c.clone(), m.clone())))
        .limitations(why_not_deep.map(|w| Limitation::new("deep_engine_unavailable", format!("{w} So this came from the built-in decompiler, which is less exact."))))
        .location(Location::address(f.entry))
        .build()?;
    Ok(ToolOutput::evidence(e))
}

/// An address Ghidra can look up: `0x…`, a bare hex address, or a generated name like `FUN_…`.
fn ghidra_address(function: &str) -> Option<u64> {
    let s = function.trim();
    let s = ["FUN_", "SUB_", "sub_", "LAB_"].iter().find_map(|p| s.strip_prefix(p)).unwrap_or(s);
    super::parse_address(s).ok()
}

fn native_function_address_show(ctx: &mut Ctx<'_>, params: Value) -> Result<ToolOutput, ToolError> {
    with_proc!(ctx, params, ProcParams, |p, st, t, f| {
        let name = name_of(t, &st.prog, f.entry).unwrap_or_else(|| f.name.clone());
        let e = ev(ctx, "native_function_address_show", params_json(&p))?
            .result(json!({"function": name, "address": hex(f.entry)}))
            .summary(format!("{name} starts at {:#x}.", f.entry))
            .location(Location::address(f.entry))
            .build()?;
        Ok(ToolOutput::evidence(e))
    })
}

fn insn_json(st: &super::native_common::NativeState, t: &crate::session::OpenTarget, i: &craft_anything_disasm::Insn) -> Value {
    json!({
        "address": hex(i.addr),
        "bytes": st.img.read(i.addr, usize::from(i.len)).map(hex::encode).unwrap_or_default(),
        "length": i.len,
        "mnemonic": i.mnemonic,
        "operands": i.operands,
        "text": i.text(),
        "flow": i.flow,
        "condition": i.cond,
        "references": i.refs.iter().filter(|r| st.img.is_mapped(**r)).map(|r| json!({"address": hex(*r), "name": name_of(t, &st.prog, *r), "string": st.prog.string_at(*r).map(|s| s.text.clone())})).collect::<Vec<_>>(),
        "reads": i.reads,
        "writes": i.writes,
    })
}

fn native_function_instructions_list(ctx: &mut Ctx<'_>, params: Value) -> Result<ToolOutput, ToolError> {
    with_proc!(ctx, params, ProcParams, |p, st, t, f| {
        let insns = st.prog.instructions(&st.img, f, true);
        let rows: Vec<Value> = insns.iter().map(|i| insn_json(&st, t, i)).collect();
        let e = ev(ctx, "native_function_instructions_list", params_json(&p))?
            .result(json!({"function": f.name, "entry": hex(f.entry), "instructions": rows}))
            .summary(format!("{} instructions in {}.", rows.len(), f.name))
            .coverage(Coverage::complete(rows.len() as u64))
            .location(Location::address(f.entry))
            .build()?;
        Ok(ToolOutput::evidence(e))
    })
}

fn native_function_find(ctx: &mut Ctx<'_>, params: Value) -> Result<ToolOutput, ToolError> {
    let p: AddrParams = parse_params(params)?;
    select_document(ctx, &p.document)?;
    let a = super::parse_address(&p.address)?;
    let st = native(ctx)?;
    let t = ctx.session.active()?;
    let f = st.prog.function_containing(a);
    let result = match f {
        Some(f) => json!({"address": hex(a), "function": name_of(t, &st.prog, f.entry), "entry": hex(f.entry), "offset": a - f.entry}),
        None => {
            json!({"address": hex(a), "function": null, "reason": if st.img.is_exec(a) { "not inside any discovered function" } else { "not in executable code" }})
        }
    };
    let summary = match f {
        Some(f) => format!("{a:#x} is inside {} (+{:#x}).", f.name, a - f.entry),
        None => format!("{a:#x} isn't inside a known function."),
    };
    let mut b =
        ev(ctx, "native_function_find", params_json(&p))?.result(result).summary(summary).confidence(Confidence::Derived).location(Location::address(a));
    if f.is_none() {
        b = b.limitation(Limitation::new("not_found", "the address may be in code the engine didn't discover (absence is not proof)"));
    }
    Ok(ToolOutput::evidence(b.build()?))
}

fn native_instruction_show(ctx: &mut Ctx<'_>, params: Value) -> Result<ToolOutput, ToolError> {
    let p: AddrParams = parse_params(params)?;
    select_document(ctx, &p.document)?;
    let a = super::parse_address(&p.address)?;
    let st = native(ctx)?;
    let t = ctx.session.active()?;
    let f = st.prog.function_containing(a);
    let (result, summary, conf) = if st.img.is_exec(a) {
        // Decode with the AArch64 constant tracker primed from the block start.
        let insn = match f.and_then(|f| f.block_at(a)) {
            Some(b) if st.prog.mode == craft_anything_disasm::Mode::A64 => {
                let mut consts = craft_anything_disasm::ArmConsts::default();
                let mut cur = b.start;
                let mut found = None;
                while cur <= a {
                    let Some(mut i) = st.img.decode(cur, true) else { break };
                    consts.apply(&mut i);
                    cur = i.end();
                    if i.addr == a {
                        found = Some(i);
                        break;
                    }
                }
                found
            }
            _ => st.img.decode(a, true),
        };
        match insn {
            Some(i) => {
                let mut v = insn_json(&st, t, &i);
                v["kind"] = json!("instruction");
                v["function"] = json!(f.map(|f| name_of(t, &st.prog, f.entry)));
                v["xrefs_from"] = json!(st.prog.xrefs_from.get(&a));
                v["xrefs_to"] = json!(st.prog.xrefs_to.get(&a).map(Vec::len).unwrap_or(0));
                (v, format!("{a:#x}: {}", i.text()), Confidence::Observed)
            }
            None => (
                json!({"address": hex(a), "kind": "invalid", "bytes": st.img.read(a, 8).map(hex::encode)}),
                format!("{a:#x} doesn't decode as an instruction."),
                Confidence::Observed,
            ),
        }
    } else {
        let bytes = st.img.read(a, 16).map(hex::encode);
        (
            json!({"address": hex(a), "kind": "data", "bytes": bytes, "string": st.prog.string_at(a).map(|s| s.text.clone()), "name": name_of(t, &st.prog, a)}),
            format!("{a:#x} is data, not code."),
            Confidence::Observed,
        )
    };
    let e = ev(ctx, "native_instruction_show", params_json(&p))?.result(result).summary(summary).confidence(conf).location(Location::address(a)).build()?;
    Ok(ToolOutput::evidence(e))
}

fn native_call_site_resolve(ctx: &mut Ctx<'_>, params: Value) -> Result<ToolOutput, ToolError> {
    let p: AddrParams = parse_params(params)?;
    select_document(ctx, &p.document)?;
    let a = super::parse_address(&p.address)?;
    let st = native(ctx)?;
    let t = ctx.session.active()?;
    let f = st.prog.function_containing(a);
    let call = f.and_then(|f| f.calls.iter().find(|c| c.site == a));
    let jt = f.and_then(|f| f.jump_tables.iter().find(|j| j.site == a));
    let insn = st.img.decode(a, true);
    let (status, targets, mut lims): (&str, Vec<Value>, Vec<Limitation>) = match (call, jt, insn.as_ref().map(|i| i.flow)) {
        (Some(c), _, _) if c.import.is_some() => ("resolved", vec![json!({"import": c.import, "slot": c.target.map(hex)})], vec![]),
        (Some(c), _, _) if c.target.is_some() => {
            ("resolved", vec![json!({"address": c.target.map(hex), "name": c.target.and_then(|x| name_of(t, &st.prog, x))})], vec![])
        }
        (_, Some(j), _) => (
            if j.targets.len() == 1 { "resolved" } else { "ambiguous" },
            j.targets.iter().map(|x| json!({"address": hex(*x), "via": "jump table", "table": hex(j.table)})).collect(),
            vec![Limitation::new("jump_table_bounds", "jump-table size is inferred from a bounds check or from where valid entries stop")],
        ),
        (_, _, Some(Flow::Call { target } | Flow::Jump { target } | Flow::Cond { target })) => {
            ("resolved", vec![json!({"address": hex(target), "name": name_of(t, &st.prog, target)})], vec![])
        }
        (_, _, Some(Flow::IndirectCall { slot: Some(s) } | Flow::IndirectJump { slot: Some(s) })) => {
            ("resolved", vec![json!({"slot": hex(s), "name": name_of(t, &st.prog, s), "pointer_value": st.img.bin.read_ptr(&st.img.data, s).map(hex)})], vec![])
        }
        (_, _, Some(Flow::IndirectCall { slot: None } | Flow::IndirectJump { slot: None })) => (
            "unresolved",
            vec![],
            vec![Limitation::new("indirect_target_unknown", "the target is computed at run time (register or table); static analysis can't pin it")],
        ),
        (_, _, Some(_)) => ("not_a_call", vec![], vec![]),
        (_, _, None) => ("not_code", vec![], vec![]),
    };
    if f.is_none() && status != "not_code" {
        lims.push(Limitation::new("outside_function", "the site isn't inside a discovered function"));
    }
    let summary = format!("Call site {a:#x}: {status} ({} candidate targets).", targets.len());
    let e = ev(ctx, "native_call_site_resolve", params_json(&p))?
        .result(json!({"site": hex(a), "instruction": insn.map(|i| i.text()), "status": status, "targets": targets}))
        .summary(summary)
        .confidence(Confidence::Derived)
        .limitations(lims)
        .location(Location::address(a))
        .build()?;
    Ok(ToolOutput::evidence(e))
}

fn native_data_show(ctx: &mut Ctx<'_>, params: Value) -> Result<ToolOutput, ToolError> {
    let p: DataTypeParams = parse_params(params)?;
    select_document(ctx, &p.document)?;
    let a = super::native_common::address_or_cursor(ctx.session, &p.address)?;
    let st = native(ctx)?;
    let t = ctx.session.active()?;
    let img = &st.img;
    let le = img.bin.little_endian;
    let rd = |n: usize| -> Option<u64> {
        let b = img.read(a, n)?;
        if b.len() < n {
            return None;
        }
        let mut v = 0u64;
        for (i, x) in b.iter().enumerate() {
            let sh = if le { i * 8 } else { (n - 1 - i) * 8 };
            v |= u64::from(*x) << sh;
        }
        Some(v)
    };
    let ptr = img.pointer_size() as usize;
    let describe = |v: u64| -> Value {
        if let Some(f) = st.prog.functions.get(&v) {
            json!({"points_to": "function", "address": hex(v), "name": name_of(t, &st.prog, f.entry)})
        } else if let Some(s) = st.prog.string_at(v) {
            json!({"points_to": "string", "address": hex(v), "text": s.text})
        } else if img.is_mapped(v) {
            json!({"points_to": if img.is_exec(v) { "code" } else { "data" }, "address": hex(v), "name": name_of(t, &st.prog, v)})
        } else {
            Value::Null
        }
    };
    let (ty, value, conf) = match p.ty.as_deref() {
        Some(ty) => {
            let v = match ty {
                "u8" => rd(1).map(|v| json!(v)),
                "u16" => rd(2).map(|v| json!(v)),
                "u32" => rd(4).map(|v| json!(v)),
                "u64" => rd(8).map(|v| json!(v)),
                "i32" => rd(4).map(|v| json!(v as u32 as i32)),
                "i64" => rd(8).map(|v| json!(v as i64)),
                "f32" => rd(4).map(|v| json!(f32::from_bits(v as u32)).to_string().into()),
                "f64" => rd(8).map(|v| json!(f64::from_bits(v)).to_string().into()),
                "pointer" => rd(ptr).map(|v| json!({"value": hex(v), "target": describe(v)})),
                "cstring" => img.read(a, 4096).map(|b| json!(String::from_utf8_lossy(b.split(|c| *c == 0).next().unwrap_or(&[])))),
                "utf16" => img.read(a, 8192).map(|b| {
                    let units: Vec<u16> = b.as_chunks::<2>().0.iter().map(|c| u16::from_le_bytes(*c)).take_while(|u| *u != 0).collect();
                    json!(String::from_utf16_lossy(&units))
                }),
                other => return Err(ToolError::invalid(format!("unknown type {other}; use u8, u16, u32, u64, i32, i64, f32, f64, pointer, cstring or utf16"))),
            };
            (ty.to_string(), v.ok_or_else(|| ToolError::invalid(format!("{a:#x} isn't backed by enough bytes for {ty}")))?, Confidence::Observed)
        }
        None => {
            if let Some(s) = st.prog.string_at(a) {
                ("string".into(), json!({"text": s.text, "encoding": s.encoding, "starts_at": hex(s.addr)}), Confidence::Derived)
            } else if let Some(v) = rd(ptr).filter(|v| *v != 0 && img.is_mapped(*v)) {
                // A run of pointers is a table (vtable, callback list, string table).
                let mut n = 1;
                while n < 512
                    && img
                        .bin
                        .read(&img.data, a + (n * ptr) as u64, ptr)
                        .and_then(|_| img.bin.read_ptr(&img.data, a + (n * ptr) as u64))
                        .is_some_and(|v| img.is_mapped(v))
                {
                    n += 1;
                }
                let ty = if n > 1 { format!("pointer[{n}]") } else { "pointer".into() };
                (ty, json!({"value": hex(v), "target": describe(v), "table_length": n}), Confidence::Inferred)
            } else if let Some(v) = rd(4) {
                (
                    "u32".into(),
                    json!({"value": v, "hex": hex(v), "as_i32": v as u32 as i32, "as_f32": f32::from_bits(v as u32).to_string()}),
                    Confidence::Inferred,
                )
            } else {
                return Err(ToolError::invalid(format!("{a:#x} isn't backed by bytes in the file")));
            }
        }
    };
    let e = ev(ctx, "native_data_show", params_json(&p))?
        .result(json!({"address": hex(a), "type": ty, "value": value, "name": name_of(t, &st.prog, a), "section": img.bin.section_at(a).map(|s| s.name.clone()), "references_to": st.prog.xrefs_to.get(&a).map(Vec::len).unwrap_or(0)}))
        .summary(format!("{a:#x} looks like {ty}."))
        .confidence(conf)
        .limitation(Limitation::new("no_type_recovery", "the clean-room engine doesn't recover declared types; this reads the bytes and infers what they look like"))
        .location(Location::address(a))
        .build()?;
    Ok(ToolOutput::evidence(e))
}

#[cfg(test)]
mod tests {
    use super::super::native_common::test_support::*;
    use craft_anything_testkit as tk;
    use serde_json::json;

    #[test]
    fn function_tools() {
        let (elf, l) = tk::elf64(tk::ElfArch::X86_64, true);
        let e = engine_with("prog", &elf);
        let r = call(&e, "native_function_show", json!({"function": "helper"}));
        assert_eq!(r["result"]["basic_blocks"], 3);
        assert_eq!(r["confidence"], "derived");
        let r = call(&e, "native_function_disassemble", json!({"function": "main"}));
        assert!(r["result"]["assembly"].as_str().unwrap().contains("; \"Hello, Craft!\""));
        let r = call(&e, "native_function_callees_list", json!({"function": "main"}));
        assert_eq!(r["result"]["callees"].as_array().unwrap().len(), 3);
        let r = call(&e, "native_function_callers_list", json!({"function": "dispatch"}));
        assert_eq!(r["result"]["callers"][0]["name"], "main");
        let r = call(&e, "native_function_references_list", json!({"function": "main", "direction": "outgoing"}));
        assert!(r["result"]["references"].as_array().unwrap().iter().any(|x| x["kind"] == "string"));
        let r = call(&e, "native_function_references_list", json!({"function": "helper", "direction": "incoming"}));
        assert_eq!(r["result"]["references"].as_array().unwrap().len(), 1);
        let r = call(&e, "native_function_pseudocode", json!({"function": "main"}));
        assert_eq!(r["confidence"], "inferred");
        assert!(r["result"]["pseudo_code"].as_str().unwrap().contains("puts(\"Hello, Craft!\")"));
        assert!(!r["limitations"].as_array().unwrap().is_empty());
        // Deep mode without Ghidra: the built-in answer says so, instead of passing as Ghidra's.
        let store = tempfile::tempdir().unwrap();
        {
            let mut s = e.session();
            s.settings.deep_mode = true;
            s.settings.engines.store_dir = Some(store.path().to_path_buf());
        }
        let r = call(&e, "native_function_pseudocode", json!({"function": "main"}));
        assert_eq!(r["environment"]["engine"], "builtin");
        let why = r["limitations"].as_array().unwrap().iter().find(|l| l["code"] == "deep_engine_unavailable").expect("says why Ghidra wasn't used");
        assert!(why.to_string().contains("isn't installed") && why.to_string().contains("came from the built-in decompiler"), "{why}");
        // A function the built-in analysis never mapped goes to Ghidra in Deep mode; without
        // Ghidra the error says why rather than just "not found".
        let err = call_err(&e, "native_function_pseudocode", json!({"function": "FUN_7fff0000"}));
        let hint = err.remediation.unwrap_or_default();
        assert!(hint.contains("isn't installed") && hint.contains("native_function_find"), "{hint}");
        assert!(!hint.contains("came from the built-in decompiler"), "nothing was decompiled: {hint}");
        // Clean room never decompiles: the tool is refused, saying how to turn Deep mode on.
        e.session().settings.deep_mode = false;
        let err = call_err(&e, "native_function_pseudocode", json!({"function": "main"}));
        assert!(crate::clean_room::is_refusal(&err) && err.remediation.unwrap_or_default().contains("Deep mode"), "{}", err.message);
        e.session().settings.deep_mode = true;
        assert_eq!(
            (super::ghidra_address("FUN_145f7f5b0"), super::ghidra_address("0x10"), super::ghidra_address("main")),
            (Some(0x145f7f5b0), Some(0x10), None)
        );
        let r = call(&e, "native_function_address_show", json!({"function": "dispatch"}));
        assert_eq!(r["result"]["address"], format!("{:#x}", l.code_va + tk::x64::DISPATCH));
        let r = call(&e, "native_function_instructions_list", json!({"function": "helper"}));
        assert_eq!(r["result"]["instructions"][0]["text"], "push rbp");
        let inside = format!("{:#x}", l.code_va + tk::x64::HELPER + 4);
        let r = call(&e, "native_function_find", json!({"address": inside}));
        assert_eq!(r["result"]["function"], "helper");
        let r = call(&e, "native_instruction_show", json!({"address": format!("{:#x}", l.code_va + 8)}));
        assert_eq!(r["result"]["references"][0]["string"], tk::HELLO);
        let r = call(&e, "native_call_site_resolve", json!({"address": format!("{:#x}", l.code_va + 15)}));
        assert_eq!(r["result"]["targets"][0]["import"], "puts");
        let jmp = format!("{:#x}", l.code_va + tk::x64::DISPATCH + 0x15);
        let r = call(&e, "native_call_site_resolve", json!({"address": jmp}));
        assert_eq!(r["result"]["status"], "ambiguous");
        assert_eq!(r["result"]["targets"].as_array().unwrap().len(), 3);
        let r = call(&e, "native_data_show", json!({"address": format!("{:#x}", l.rodata_va)}));
        assert_eq!(r["result"]["type"], "string");
        let r = call(&e, "native_data_show", json!({"address": format!("{:#x}", l.rodata_va + tk::rodata::TABLE), "type": "i32"}));
        assert!(r["result"]["value"].is_number());
        let err = call_err(&e, "native_function_show", json!({"function": "nope"}));
        assert!(err.remediation.unwrap().contains("native_functions_search"));
        // by address works too
        let r = call(&e, "native_function_show", json!({"function": format!("{:#x}", l.code_va)}));
        assert_eq!(r["result"]["name"], "main");
    }
}
