//! Counts C/C++ multi-declarator declarations and the names they drop, split by whether the
//! carrying node is a kind tilth indexes at all.
//!
//! #81 reported ~97k lost names on a large tree, counting `declaration`, `field_declaration` and
//! `type_definition` together. But an ordinary `declaration` is deliberately *not* a definition
//! kind (`treesitter.rs`: "a local variable or a prototype, not a definition"), so its declarators
//! are not indexed whether there is one of them or ten — those names are not lost to the
//! multi-declarator bug. This splits the two so the fix can be sized against the part it fixes.
//!
//! Usage: `cargo run --release --example count_multi_declarators -- <dir>`

// One cast of a declarator count to `f64` for a percentage. See the note in
// `calibrate_parse_budget.rs` — same reasoning, same irrelevance.
#![allow(clippy::cast_precision_loss)]

use std::path::Path;

use tilth::__calibration::{grammar_for, language_of, parse, Lang};

#[derive(Default)]
struct Counts {
    nodes: u64,
    multi_nodes: u64,
    /// Declarators beyond the first — the names a one-name-per-node path drops.
    dropped: u64,
}

impl Counts {
    fn add(&mut self, declarators: u64) {
        self.nodes += 1;
        if declarators > 1 {
            self.multi_nodes += 1;
            self.dropped += declarators - 1;
        }
    }
}

/// Declarator children, enumerated exactly as the fix does: by the `declarator` field name.
///
/// A kind list would be a different question — it can miss a shape the grammar adds and it has to
/// remember to exclude the `type` child — and then this would size something other than what the
/// fix recovers.
fn declarator_count(node: tree_sitter::Node) -> u64 {
    let mut cursor = node.walk();
    node.children_by_field_name("declarator", &mut cursor)
        .count() as u64
}

/// Iterative, not recursive: a deeply nested C++ AST overflows the stack on a real tree, which
/// showed up as a bare exit code rather than a panic.
fn walk(root: tree_sitter::Node, field: &mut Counts, typedef: &mut Counts, decl: &mut Counts) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        match node.kind() {
            "field_declaration" => field.add(declarator_count(node)),
            "type_definition" => typedef.add(declarator_count(node)),
            "declaration" => decl.add(declarator_count(node)),
            _ => {}
        }
        let mut cursor = node.walk();
        stack.extend(node.named_children(&mut cursor));
    }
}

fn main() {
    let root = std::env::args().nth(1).expect("usage: <dir>");
    let (mut field, mut typedef, mut decl) =
        (Counts::default(), Counts::default(), Counts::default());
    let (mut files, mut bytes) = (0u64, 0u64);

    for entry in ignore::WalkBuilder::new(&root)
        .hidden(false)
        .git_ignore(false)
        .parents(false)
        .build()
        .flatten()
    {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(lang) = grammar_for(path) else {
            continue;
        };
        if !matches!(lang, Lang::C | Lang::Cpp) {
            continue;
        }
        let Ok(content) = std::fs::read_to_string(path) else {
            continue;
        };
        let Some(ts_lang) = language_of(lang) else {
            continue;
        };
        let Some(tree) = parse(&content, lang, &ts_lang) else {
            continue;
        };
        files += 1;
        bytes += content.len() as u64;
        walk(tree.root_node(), &mut field, &mut typedef, &mut decl);
        if files % 5_000 == 0 {
            eprintln!("  {files} files…");
        }
    }

    println!("root: {}", Path::new(&root).display());
    println!(
        "C/C++ files parsed: {files}  ({:.1} MB)",
        bytes as f64 / 1e6
    );
    println!();
    println!("kind                indexed?   nodes    multi-decl   names dropped");
    for (name, indexed, c) in [
        ("field_declaration", "yes", &field),
        ("type_definition", "yes", &typedef),
        ("declaration", "NO", &decl),
    ] {
        println!(
            "{name:<20}{indexed:<10}{:<9}{:<13}{}",
            c.nodes, c.multi_nodes, c.dropped
        );
    }
    println!();
    println!(
        "recoverable by fixing the indexed kinds: {} names across {} nodes",
        field.dropped + typedef.dropped,
        field.multi_nodes + typedef.multi_nodes
    );
    println!(
        "attributed to `declaration`, which is not a definition kind: {} names",
        decl.dropped
    );
}
