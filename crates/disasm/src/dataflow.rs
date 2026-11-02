//! Register def-use inside one function: which instruction's result each instruction reads,
//! where the incoming parameters are used, and the function's side effects (stores, calls).
//! Reaching definitions over the CFG, register-level, bounded.

use crate::decode::{ArmConsts, Flow, Insn, Kind, Mode, Op};
use crate::image::Image;
use crate::program::{Function, Program};
use craft_anything_formats::{Format, hex_addr};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DfNode {
    pub id: usize,
    /// `instruction` or `parameter`
    pub kind: &'static str,
    #[serde(with = "hex_addr::opt")]
    pub addr: Option<u64>,
    pub text: String,
    pub defines: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DfEdge {
    pub from: usize,
    pub to: usize,
    pub register: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Effect {
    #[serde(with = "hex_addr")]
    pub addr: u64,
    /// store, call
    pub kind: &'static str,
    pub text: String,
    /// Registers whose values flow into the effect.
    pub inputs: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct DefUse {
    pub nodes: Vec<DfNode>,
    pub edges: Vec<DfEdge>,
    pub parameters: Vec<String>,
    /// (parameter, instruction address) where an incoming argument is read.
    pub parameter_uses: Vec<(String, String)>,
    pub effects: Vec<Effect>,
    pub truncated: bool,
}

fn ignored(r: &str) -> bool {
    matches!(r, "rsp" | "esp" | "sp" | "rip" | "rflags" | "eflags" | "flags" | "x30" | "x29" | "rbp")
}

/// Build the def-use graph of `f`.
pub fn def_use(img: &Image, prog: &Program, f: &Function, max_nodes: usize, max_edges: usize) -> DefUse {
    let mode = prog.mode;
    let args: &[&str] = match mode {
        Mode::X64 if img.bin.format == Format::Pe => &["rcx", "rdx", "r8", "r9"],
        Mode::X64 => &["rdi", "rsi", "rdx", "rcx", "r8", "r9"],
        Mode::A64 => &["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"],
        Mode::A32 | Mode::T32 => &["r0", "r1", "r2", "r3"],
        _ => &[],
    };
    let clobbered: &[&str] = match mode {
        Mode::X64 if img.bin.format == Format::Pe => &["rax", "rcx", "rdx", "r8", "r9", "r10", "r11"],
        Mode::X64 => &["rax", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11"],
        Mode::A64 => &["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17"],
        _ => &["rax", "rcx", "rdx", "eax", "ecx", "edx"],
    };
    let ret = if mode == Mode::A64 { "x0" } else { "rax" };
    let mut out = DefUse::default();
    // Decode.
    let mut blocks: Vec<Vec<Insn>> = Vec::new();
    for b in &f.blocks {
        let mut v = Vec::new();
        let mut consts = ArmConsts::default();
        let mut a = b.start;
        while a < b.end && v.len() < 20_000 {
            let Some(mut i) = img.decode(a, true) else { break };
            if mode == Mode::A64 {
                consts.apply(&mut i);
            }
            // Calls define the return register and clobber the scratch registers.
            if matches!(i.flow, Flow::Call { .. } | Flow::IndirectCall { .. }) {
                for c in clobbered {
                    if !i.writes.iter().any(|w| w == c) {
                        i.writes.push((*c).to_string());
                    }
                }
                for r in args {
                    if !i.reads.iter().any(|x| x == r) {
                        i.reads.push((*r).to_string());
                    }
                }
            }
            if matches!(i.flow, Flow::Return) && !i.reads.iter().any(|x| x == ret) {
                i.reads.push(ret.to_string());
            }
            a = i.end();
            v.push(i);
        }
        blocks.push(v);
    }
    let total: usize = blocks.iter().map(Vec::len).sum();
    if total > max_nodes {
        out.truncated = true;
    }
    // Nodes: parameters, then instructions (capped).
    let mut node_of: BTreeMap<u64, usize> = BTreeMap::new();
    let mut param_node: BTreeMap<String, usize> = BTreeMap::new();
    for r in args {
        let id = out.nodes.len();
        param_node.insert((*r).to_string(), id);
        out.nodes.push(DfNode { id, kind: "parameter", addr: None, text: format!("incoming {r}"), defines: vec![(*r).to_string()] });
    }
    for v in &blocks {
        for i in v {
            if out.nodes.len() >= max_nodes {
                break;
            }
            let id = out.nodes.len();
            node_of.insert(i.addr, id);
            out.nodes.push(DfNode {
                id,
                kind: "instruction",
                addr: Some(i.addr),
                text: i.text(),
                defines: i.writes.iter().filter(|w| !ignored(w)).cloned().collect(),
            });
        }
    }
    // Reaching definitions: reg → set of defining node ids.
    type Defs = BTreeMap<String, BTreeSet<usize>>;
    let index: BTreeMap<u64, usize> = f.blocks.iter().enumerate().map(|(i, b)| (b.start, i)).collect();
    let n = f.blocks.len();
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, b) in f.blocks.iter().enumerate() {
        for s in &b.succs {
            if let Some(&j) = index.get(s)
                && let Some(p) = preds.get_mut(j)
            {
                p.push(i);
            }
        }
    }
    let entry = index.get(&f.entry).copied().unwrap_or(0);
    let mut entry_defs: Defs = BTreeMap::new();
    for (r, id) in &param_node {
        entry_defs.insert(r.clone(), BTreeSet::from([*id]));
    }
    let transfer = |inp: &Defs, insns: &[Insn]| -> Defs {
        let mut cur = inp.clone();
        for i in insns {
            if let Some(&id) = node_of.get(&i.addr) {
                for w in &i.writes {
                    if !ignored(w) {
                        cur.insert(w.clone(), BTreeSet::from([id]));
                    }
                }
            }
        }
        cur
    };
    let mut outs: Vec<Defs> = vec![BTreeMap::new(); n];
    let mut ins: Vec<Defs> = vec![BTreeMap::new(); n];
    let mut changed = true;
    let mut rounds = 0;
    while changed && rounds < 64 {
        changed = false;
        rounds += 1;
        for b in 0..n {
            let mut inp: Defs = if b == entry { entry_defs.clone() } else { BTreeMap::new() };
            for &p in preds.get(b).map(Vec::as_slice).unwrap_or(&[]) {
                if let Some(po) = outs.get(p) {
                    for (r, ds) in po {
                        inp.entry(r.clone()).or_default().extend(ds.iter().copied());
                    }
                }
            }
            let o = transfer(&inp, blocks.get(b).map(Vec::as_slice).unwrap_or(&[]));
            if outs.get(b) != Some(&o) {
                if let Some(x) = outs.get_mut(b) {
                    *x = o;
                }
                changed = true;
            }
            if let Some(x) = ins.get_mut(b) {
                *x = inp;
            }
        }
    }
    if rounds >= 64 {
        out.truncated = true;
    }
    // Edges and effects.
    let mut params_used: BTreeSet<String> = BTreeSet::new();
    for (b, v) in blocks.iter().enumerate() {
        let mut cur = ins.get(b).cloned().unwrap_or_default();
        for i in v {
            let Some(&me) = node_of.get(&i.addr) else { continue };
            for r in &i.reads {
                if ignored(r) {
                    continue;
                }
                for &d in cur.get(r).map(|s| s.iter().copied().collect::<Vec<_>>()).unwrap_or_default().iter() {
                    if out.edges.len() >= max_edges {
                        out.truncated = true;
                        break;
                    }
                    out.edges.push(DfEdge { from: d, to: me, register: r.clone() });
                    if param_node.get(r) == Some(&d) {
                        params_used.insert(r.clone());
                        if out.parameter_uses.len() < 4096 {
                            out.parameter_uses.push((r.clone(), format!("{:#x}", i.addr)));
                        }
                    }
                }
            }
            let is_store = matches!(i.kind, Kind::Store | Kind::StorePair) || (i.mode_writes_memory());
            if is_store && out.effects.len() < 4096 {
                out.effects.push(Effect { addr: i.addr, kind: "store", text: i.text(), inputs: i.reads.iter().filter(|r| !ignored(r)).cloned().collect() });
            }
            if matches!(i.flow, Flow::Call { .. } | Flow::IndirectCall { .. }) && out.effects.len() < 4096 {
                let name = match i.flow {
                    Flow::Call { target } => prog.name_at(target).unwrap_or_else(|| format!("sub_{target:x}")),
                    Flow::IndirectCall { slot: Some(s) } => prog.import_slots.get(&s).map(|x| x.name.clone()).unwrap_or_else(|| i.text()),
                    _ => i.text(),
                };
                let inputs = args
                    .iter()
                    .filter(|r| cur.get(**r).is_some_and(|s| s.iter().any(|d| !param_node.values().any(|p| p == d))))
                    .map(|r| (*r).to_string())
                    .collect();
                out.effects.push(Effect { addr: i.addr, kind: "call", text: name, inputs });
            }
            for w in &i.writes {
                if !ignored(w) {
                    cur.insert(w.clone(), BTreeSet::from([me]));
                }
            }
        }
    }
    out.parameters = args.iter().filter(|a| params_used.contains(**a)).map(|a| (*a).to_string()).collect();
    out
}

impl Insn {
    /// True when an x86 instruction writes its memory operand (`mov [m], r`, `add [m], r`…).
    fn mode_writes_memory(&self) -> bool {
        matches!(self.ops.first(), Some(Op::Mem(_)))
            && matches!(self.kind, Kind::Mov | Kind::Add | Kind::Sub | Kind::And | Kind::Or | Kind::Xor | Kind::Inc | Kind::Dec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NoControl;
    use crate::program::Limits;
    use craft_anything_testkit as tk;
    use std::sync::Arc;

    #[test]
    fn helper_def_use() {
        let (elf, _) = tk::elf64(tk::ElfArch::X86_64, true);
        let img = Image::parse(Arc::from(elf)).unwrap();
        let p = Program::analyze(&img, &Limits::default(), &NoControl).unwrap();
        let h = p.function_by_name("helper").unwrap();
        let du = def_use(&img, &p, h, 5000, 10000);
        assert_eq!(du.parameters, vec!["rdi".to_string()]);
        // `add eax, edi` reads rdi from the parameter and from the loop's `sub edi, 1`.
        let add = du.nodes.iter().find(|n| n.text.starts_with("add eax")).unwrap();
        let froms: Vec<_> = du.edges.iter().filter(|e| e.to == add.id && e.register == "rdi").map(|e| e.from).collect();
        assert_eq!(froms.len(), 2, "{du:?}");
        let main = p.function_by_name("main").unwrap();
        let du = def_use(&img, &p, main, 5000, 10000);
        assert!(du.effects.iter().any(|e| e.kind == "call" && e.text == "puts"));
        assert!(du.effects.iter().any(|e| e.kind == "store"));
        let tiny = def_use(&img, &p, main, 3, 2);
        assert!(tiny.truncated);
    }
}
