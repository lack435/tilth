use crate::types::Lang;

/// Extract test structure (describe/it/test) via tree-sitter queries.
/// Returns a structured test outline with suite nesting, or None if
/// no test structure was found.
pub fn outline(content: &str, lang: Lang, max_lines: usize) -> Option<String> {
    let language = crate::lang::outline::outline_language(lang)?;

    let tree = crate::lang::parse_masked(content, Some(lang), &language)?;

    let lines: Vec<&str> = content.lines().collect();
    let root = tree.root_node();
    let mut entries = Vec::new();

    extract_test_calls(root, &lines, 0, max_lines, &mut entries);

    if entries.is_empty() {
        return None;
    }

    Some(entries.join("\n"))
}

/// Recursively find describe/it/test call expressions.
fn extract_test_calls(
    node: tree_sitter::Node,
    lines: &[&str],
    depth: usize,
    max_lines: usize,
    entries: &mut Vec<String>,
) {
    if entries.len() >= max_lines {
        return;
    }

    let kind = node.kind();

    // Look for call expressions: describe(...), it(...), test(...)
    if kind == "call_expression" || kind == "expression_statement" {
        if let Some(name) = extract_test_name(node, lines) {
            let line = node.start_position().row as u32 + 1;
            let indent = "  ".repeat(depth);
            let label = if name.starts_with("describe") || name.starts_with("context") {
                "suite"
            } else {
                "test"
            };
            entries.push(format!("{indent}[{line}] {label}: {name}"));

            // Recurse into the callback body for nested describes
            if label == "suite" {
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    extract_test_calls(child, lines, depth + 1, max_lines, entries);
                }
                return;
            }
        }
    }

    // Recurse
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        extract_test_calls(child, lines, depth, max_lines, entries);
    }
}

/// Extract the function name and first string argument from a call expression.
fn extract_test_name(node: tree_sitter::Node, lines: &[&str]) -> Option<String> {
    let mut cursor = node.walk();

    // Find the function name
    let func = node.children(&mut cursor).find(|c| {
        let k = c.kind();
        k == "identifier" || k == "member_expression" || k == "call_expression"
    })?;

    let func_text = get_node_text(func, lines);
    if !matches!(
        func_text.as_str(),
        "describe" | "it" | "test" | "context" | "specify"
    ) {
        return None;
    }

    // Find the first string argument
    let mut cursor2 = node.walk();
    let args = node
        .children(&mut cursor2)
        .find(|c| c.kind() == "arguments")?;

    let mut cursor3 = args.walk();
    let first_arg = args.children(&mut cursor3).find(|c| {
        let k = c.kind();
        k == "string" || k == "template_string" || k == "string_literal"
    })?;

    let arg_text = get_node_text(first_arg, lines);
    // Strip quotes
    let cleaned = arg_text
        .trim_matches('"')
        .trim_matches('\'')
        .trim_matches('`');

    Some(format!("{func_text}(\"{cleaned}\")"))
}

fn get_node_text(node: tree_sitter::Node, lines: &[&str]) -> String {
    // The third copy of this slice. All three clamped only the end column, and a tree-sitter
    // column that falls past a CRLF-stripped row — or inside a multi-byte character — panicked
    // in a walk thread and aborted the process. See `clamp_col` for both mechanisms.
    let row = node.start_position().row;
    let Some(line) = lines.get(row) else {
        return String::new();
    };
    let col_start = crate::lang::treesitter::clamp_col(line, node.start_position().column);
    let col_end = if row == node.end_position().row {
        crate::lang::treesitter::clamp_col(line, node.end_position().column).max(col_start)
    } else {
        line.len()
    };
    line[col_start..col_end].to_string()
}
