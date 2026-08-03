use std::path::Path;

/// Depth-limited outline for JSON, YAML, TOML.
///
/// A leading BOM is dropped once, here, rather than in each arm. `serde_json` rejects one
/// outright, so a BOM'd `.json` file outlined as `[parse error: …]` instead of its
/// structure — and PowerShell's `Set-Content -Encoding utf8` writes a BOM, so this is the
/// ordinary way to produce such a file on Windows rather than an exotic one.
///
/// Stripping at this level and not inside `yaml_outline` is deliberate: that arm derives
/// its indent from `line.len() - trimmed.len()`, and a BOM removed per-line after the
/// split would leave a 3-byte skew in the indent of the first key. Removing it from the
/// content before anything splits it keeps the byte offsets honest, and incidentally stops
/// the BOM rendering as a stray character in the first YAML key and the `key_value`
/// passthrough. `toml` strips a BOM itself, so that arm neither needs nor notices this.
pub fn outline(path: &Path, content: &str, max_lines: usize) -> String {
    // Redundant since #42 moved the same strip up into `generate`, this arm's only caller.
    // Kept because the reason it is here is a property of *this* function — `yaml_outline`
    // derives indent from `line.len() - trimmed.len()` — not of who happens to call it, and
    // a direct caller added later would reintroduce the skew.
    let content = crate::lang::outline::strip_bom(content);
    match path.extension().and_then(|e| e.to_str()) {
        Some("json") => json_outline(content, max_lines),
        Some("yaml" | "yml") => yaml_outline(content, max_lines),
        Some("toml") => toml_outline(content, max_lines),
        _ => key_value_outline(content, max_lines),
    }
}

fn json_outline(content: &str, max_lines: usize) -> String {
    let value: serde_json::Value = match serde_json::from_str(content) {
        Ok(v) => v,
        Err(e) => return format!("[parse error: {e}]"),
    };
    let mut lines = Vec::new();
    walk_json(&value, "", 0, 2, max_lines, &mut lines);
    lines.join("\n")
}

fn walk_json(
    value: &serde_json::Value,
    prefix: &str,
    depth: usize,
    max_depth: usize,
    max_lines: usize,
    lines: &mut Vec<String>,
) {
    if lines.len() >= max_lines {
        return;
    }

    match value {
        serde_json::Value::Object(map) => {
            if depth >= max_depth {
                if !prefix.is_empty() {
                    lines.push(format!("{prefix}: {{{} keys}}", map.len()));
                }
                return;
            }
            for (key, val) in map {
                if lines.len() >= max_lines {
                    return;
                }
                let full_key = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                match val {
                    serde_json::Value::Object(inner) => {
                        if depth + 1 >= max_depth {
                            let keys: Vec<&String> = inner.keys().take(5).collect();
                            let key_list = keys
                                .iter()
                                .map(|k| k.as_str())
                                .collect::<Vec<_>>()
                                .join(", ");
                            let suffix = if inner.len() > 5 { ", ..." } else { "" };
                            lines.push(format!(
                                "{full_key}: {{{} keys}} [{key_list}{suffix}]",
                                inner.len()
                            ));
                        } else {
                            walk_json(val, &full_key, depth + 1, max_depth, max_lines, lines);
                        }
                    }
                    serde_json::Value::Array(arr) => {
                        let preview = if arr.is_empty() {
                            "[]".to_string()
                        } else {
                            let first = truncate_json_value(&arr[0], 40);
                            format!("[{} items] [{first}]", arr.len())
                        };
                        lines.push(format!("{full_key}: {preview}"));
                    }
                    _ => {
                        let val_str = truncate_json_value(val, 40);
                        let type_name = json_type_name(val);
                        lines.push(format!("{full_key}: {val_str} ({type_name})"));
                    }
                }
            }
        }
        serde_json::Value::Array(arr) => {
            lines.push(format!("{prefix}: [{} items]", arr.len()));
        }
        _ => {
            let val_str = truncate_json_value(value, 40);
            lines.push(format!("{prefix}: {val_str}"));
        }
    }
}

fn json_type_name(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::String(_) => "string",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::Bool(_) => "boolean",
        serde_json::Value::Null => "null",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

fn truncate_json_value(v: &serde_json::Value, max: usize) -> String {
    let s = match v {
        serde_json::Value::String(s) => format!("\"{s}\""),
        other => other.to_string(),
    };
    if s.len() > max {
        format!(
            "{}...",
            crate::types::truncate_str(&s, max.saturating_sub(3))
        )
    } else {
        s
    }
}

/// YAML outline via line scan — no parser needed.
/// Detect keys by: optional whitespace, then a word, then `: ` or `:`+EOL.
/// Indentation level = nesting depth (2-space standard).
fn yaml_outline(content: &str, max_lines: usize) -> String {
    let mut entries = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if entries.len() >= max_lines {
            break;
        }
        let trimmed = line.trim_start();
        // Skip comments, blank lines, and list items
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with('-') {
            continue;
        }
        // Look for key: value or key: (block)
        if let Some(colon) = trimmed.find(':') {
            let key = &trimmed[..colon];
            // Keys shouldn't contain spaces (that would be a value line)
            if key.contains(' ') {
                continue;
            }
            let indent = line.len() - trimmed.len();
            let depth = indent / 2;
            if depth <= 2 {
                let prefix = "  ".repeat(depth);
                let after_colon = trimmed[colon + 1..].trim();
                if after_colon.is_empty() {
                    // Block mapping — just show key
                    entries.push(format!("[{}] {prefix}{key}:", i + 1));
                } else {
                    let val = if after_colon.len() > 40 {
                        format!("{}...", crate::types::truncate_str(after_colon, 37))
                    } else {
                        after_colon.to_string()
                    };
                    entries.push(format!("[{}] {prefix}{key}: {val}", i + 1));
                }
            }
        }
    }
    entries.join("\n")
}

fn toml_outline(content: &str, max_lines: usize) -> String {
    // toml 1.1: `Value::from_str` (via `content.parse()`) parses a single value
    // and rejects further content. `toml::from_str` parses the full document.
    let value: toml::Value = match toml::from_str(content) {
        Ok(v) => v,
        Err(e) => return format!("[parse error: {e}]"),
    };
    let mut lines = Vec::new();
    walk_toml(&value, 0, 2, max_lines, &mut lines);
    lines.join("\n")
}

fn walk_toml(
    value: &toml::Value,
    depth: usize,
    max_depth: usize,
    max_lines: usize,
    lines: &mut Vec<String>,
) {
    if lines.len() >= max_lines {
        return;
    }
    let indent = "  ".repeat(depth);

    if let toml::Value::Table(table) = value {
        for (key, val) in table {
            if lines.len() >= max_lines {
                return;
            }
            match val {
                toml::Value::Table(inner) if depth < max_depth => {
                    lines.push(format!("{indent}[{key}]"));
                    walk_toml(val, depth + 1, max_depth, max_lines, lines);
                }
                toml::Value::Table(inner) => {
                    lines.push(format!("{indent}{key}: {{{} keys}}", inner.len()));
                }
                toml::Value::Array(arr) => {
                    lines.push(format!("{indent}{key}: [{} items]", arr.len()));
                }
                _ => {
                    let val_str = val.to_string();
                    let truncated = if val_str.len() > 40 {
                        format!("{}...", crate::types::truncate_str(&val_str, 37))
                    } else {
                        val_str
                    };
                    lines.push(format!("{indent}{key}: {truncated}"));
                }
            }
        }
    }
}

fn key_value_outline(content: &str, max_lines: usize) -> String {
    content
        .lines()
        .take(max_lines)
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A UTF-8 BOM. Written as bytes and read back off disk, because a fixture built from
    /// a `&str` literal is exactly what let this class of bug survive three times.
    const UTF8_BOM: &[u8] = &[0xEF, 0xBB, 0xBF];

    /// The BOM'd and unmarked forms of `body`, both round-tripped through real files in
    /// `dir` so the BOM travels the route it does in production, and both returned with the
    /// path `outline` should be called with.
    ///
    /// The caller owns `dir` and must keep it alive for the duration of the test. An
    /// earlier version created the `TempDir` here and let it drop on return, which worked
    /// only because `outline` reads nothing but the extension — any future filesystem touch
    /// would have turned these tests into confusing failures.
    fn bom_and_plain(
        dir: &Path,
        stem: &str,
        ext: &str,
        body: &str,
    ) -> (std::path::PathBuf, String, String) {
        let mut written = Vec::new();
        for (name, prefix) in [
            (format!("{stem}_plain.{ext}"), &[][..]),
            (format!("{stem}_bom.{ext}"), UTF8_BOM),
        ] {
            let path = dir.join(name);
            let mut bytes = prefix.to_vec();
            bytes.extend_from_slice(body.as_bytes());
            std::fs::write(&path, &bytes).unwrap();
            let content = std::fs::read_to_string(&path).unwrap();
            written.push((path, content));
        }
        let (path, plain) = written[0].clone();
        let bom = written[1].1.clone();
        // Asserted for every format, not just JSON: without it, a `read_to_string` that
        // silently dropped the BOM would leave each test comparing two identical unmarked
        // runs and passing for no reason. That is the exact vacuous pass these fixtures
        // exist to rule out.
        assert!(
            bom.starts_with('\u{feff}') && !plain.starts_with('\u{feff}'),
            "fixture is broken: the BOM did not survive the write/read round trip, so this \
             test proves nothing"
        );
        (path, plain, bom)
    }

    /// `serde_json` rejects a leading BOM rather than skipping it, so a BOM'd `.json` file
    /// outlined as `[parse error: expected value at line 1 column 1]` instead of its
    /// structure. PowerShell's `Set-Content -Encoding utf8` writes a BOM, so this is the
    /// ordinary way to produce such a file on Windows, not an exotic one.
    #[test]
    fn bom_json_outlines_identically_to_the_unmarked_file() {
        let body = "{\"name\":\"demo\",\"version\":\"1.0.0\",\"scripts\":{\"build\":\"tsc\"}}";

        let dir = tempfile::tempdir().unwrap();
        let (path, plain, bom) = bom_and_plain(dir.path(), "cfg", "json", body);

        let plain_out = outline(&path, &plain, 50);
        // Pinned against a literal as well as compared, so two parse errors cannot agree.
        assert!(
            plain_out.contains("name: \"demo\" (string)")
                && plain_out.contains("scripts.build: \"tsc\" (string)"),
            "fixture is broken: unmarked JSON outline changed: {plain_out}"
        );
        assert_eq!(
            outline(&path, &bom, 50),
            plain_out,
            "a BOM changed the JSON outline"
        );
    }

    /// The BOM used to render as a stray character in the first key. Not a parse failure —
    /// YAML is line-scanned here — but the same unstripped bytes, and stripping them at the
    /// `outline` entry point rather than per-line is what keeps `yaml_outline`'s
    /// `line.len() - trimmed.len()` indent arithmetic free of a 3-byte skew.
    #[test]
    fn bom_yaml_outlines_identically_to_the_unmarked_file() {
        let body = "name: demo\nversion: 1.0.0\njobs:\n  build:\n    runs-on: ubuntu\n";

        let dir = tempfile::tempdir().unwrap();
        let (path, plain, bom) = bom_and_plain(dir.path(), "cfg", "yaml", body);

        let plain_out = outline(&path, &plain, 50);
        assert!(
            plain_out.contains("name"),
            "fixture is broken: unmarked YAML outline changed: {plain_out}"
        );
        let bom_out = outline(&path, &bom, 50);
        assert!(
            !bom_out.contains('\u{feff}'),
            "the BOM leaked into the YAML outline: {bom_out:?}"
        );
        assert_eq!(bom_out, plain_out, "a BOM changed the YAML outline");
    }

    /// The `toml` crate strips a BOM itself, so this arm never had the bug. Pinned anyway:
    /// the uniform strip at the `outline` entry point must not disturb the one format that
    /// was already correct, and a future `toml` bump that stopped stripping would otherwise
    /// reintroduce the failure silently.
    #[test]
    fn bom_toml_outlines_identically_to_the_unmarked_file() {
        let body = "[package]\nname = \"demo\"\nversion = \"0.1.0\"\n";

        let dir = tempfile::tempdir().unwrap();
        let (path, plain, bom) = bom_and_plain(dir.path(), "cfg", "toml", body);

        let plain_out = outline(&path, &plain, 50);
        assert!(
            plain_out.contains("[package]"),
            "fixture is broken: unmarked TOML outline changed: {plain_out}"
        );
        assert_eq!(
            outline(&path, &bom, 50),
            plain_out,
            "a BOM changed the TOML outline"
        );
    }

    /// Pins the contract that `toml::Value` exposes `Table` and `Array` enum
    /// variants and that top-level tables render as `[section]` headers. If a
    /// future `toml` bump renames either variant, this test fails at compile
    /// time (the variant doesn't exist) or behavior time (output is empty),
    /// instead of silently producing blank outlines for every TOML file tilth
    /// reads.
    #[test]
    fn toml_outline_section_headers_and_variants() {
        let content = "\
[package]
name = \"demo\"
version = \"0.1.0\"

[dependencies]
serde = \"1\"
tokio = { version = \"1\", features = [\"full\"] }

[features]
default = [\"std\"]
";
        let out = toml_outline(content, 100);
        assert!(
            out.contains("[package]"),
            "missing [package] header:\n{out}"
        );
        assert!(
            out.contains("[dependencies]"),
            "missing [dependencies] header:\n{out}"
        );
        assert!(
            out.contains("[features]"),
            "missing [features] header:\n{out}"
        );
        assert!(
            out.contains("name: \"demo\""),
            "missing scalar key under section:\n{out}"
        );
    }

    /// Tables beyond `max_depth` collapse to a key-count summary rather than
    /// expanding inline. Guards the depth-control path that uses
    /// `toml::Value::Table(inner)` in two arms — if a future bump merges them
    /// or changes the inner shape, the collapsed-summary format breaks first.
    #[test]
    fn toml_outline_deep_tables_collapse() {
        let content = "\
[a]
[a.b]
[a.b.c]
key = \"deep\"
";
        // max_depth in toml_outline is hard-coded to 2; `[a.b.c]` is depth 3.
        let out = toml_outline(content, 100);
        // The collapsed form names the key-count of the inner table.
        assert!(
            out.contains('{') && out.contains("keys}"),
            "deep table should collapse to key-count summary:\n{out}"
        );
    }

    /// Arrays render with item count, not their contents. Pins the
    /// `Value::Array` arm.
    #[test]
    fn toml_outline_arrays_show_item_count() {
        let content = "\
[build]
features = [\"a\", \"b\", \"c\", \"d\"]
";
        let out = toml_outline(content, 100);
        assert!(
            out.contains("features: [4 items]"),
            "array should render as item count, got:\n{out}"
        );
    }

    /// Parse errors produce a structured marker rather than a panic.
    #[test]
    fn toml_outline_parse_error_is_caught() {
        let out = toml_outline("this is not = valid toml [[[ at all", 10);
        assert!(
            out.starts_with("[parse error:"),
            "malformed TOML should surface as parse error marker, got:\n{out}"
        );
    }
}
