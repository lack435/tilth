//! Re-derive `lang::parse_budget::TREE_BYTES_PER_SOURCE_BYTE` from a corpus of real source.
//!
//! ```text
//! cargo run --release --example calibrate_parse_budget -- <dir> [<dir>...]
//! ```
//!
//! Prints, for every candidate predictor, the corpus maximum — the value a conservative constant has
//! to clear — and the mean over-estimate that constant would impose, which is what it costs in
//! admitted parallelism. Then the densest files by each predictor, and a verdict on whether the
//! shipped constant still bounds the corpus (exit code 1 if not).
//!
//! ## Why this is an example and not a test
//!
//! Measuring a tree exactly means observing tree-sitter's allocations, and tree-sitter is a C
//! library that allocates through `malloc` — a Rust `#[global_allocator]` sees **none** of it, and
//! the first attempt at this measurement duly reported zero bytes for every file. The only hook is
//! `ts_set_allocator`, which is a process-global write with a hard ordering requirement: any block
//! allocated by the previous allocator must never be freed by the new one.
//!
//! A first version of this lived in `#[cfg(test)]` and tried to satisfy that by tagging its own
//! blocks with a magic header and leaking anything untagged. That is unsound, and review said so:
//! deciding whether a block is ours reads 16 bytes *below* the pointer, which for a foreign
//! allocation is out of bounds — undefined behaviour, not merely unusual. `#[ignore]` was not
//! sufficient isolation either: under `cargo test -- --include-ignored` the lib test binary's other
//! parsing tests allocated inside the measurement window, and 5 runs in 12 produced spurious
//! assertion failures whose message told the reader to raise the constant.
//!
//! An example is a fresh process with a real `main`, so the swap happens before any tree-sitter
//! allocation can exist. The foreign-block problem does not arise, no header is needed at all, and
//! nothing else in the process is parsing. `cargo build --examples` still compiles it, so it cannot
//! rot silently.

use std::alloc::{alloc, dealloc, Layout};
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

static LIVE: AtomicUsize = AtomicUsize::new(0);

/// Sizes of the blocks this allocator handed out, keyed by pointer.
///
/// A side table rather than an in-band header: with no header there is nothing to read below a
/// pointer, so an unknown pointer is simply absent from the map instead of being probed out of
/// bounds. The swap happens before `main` does anything else, so in practice every pointer is
/// present — the `None` arm exists so that being wrong about that is a visible warning rather than
/// undefined behaviour.
static BLOCKS: Mutex<Option<HashMap<usize, usize>>> = Mutex::new(None);
static FOREIGN: AtomicUsize = AtomicUsize::new(0);

fn blocks<R>(f: impl FnOnce(&mut HashMap<usize, usize>) -> R) -> R {
    let mut guard = BLOCKS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    f(guard.get_or_insert_with(HashMap::new))
}

/// 16 bytes is C's `max_align_t` on every platform tilth builds for.
const ALIGN: usize = 16;

fn layout(size: usize) -> Layout {
    Layout::from_size_align(size.max(1), ALIGN).expect("layout")
}

unsafe fn tracked_alloc(size: usize) -> *mut c_void {
    let p = alloc(layout(size));
    if p.is_null() {
        return std::ptr::null_mut();
    }
    blocks(|m| m.insert(p as usize, size));
    LIVE.fetch_add(size, Ordering::Relaxed);
    p.cast()
}

unsafe fn tracked_free(p: *mut c_void) {
    if p.is_null() {
        return;
    }
    match blocks(|m| m.remove(&(p as usize))) {
        Some(size) => {
            LIVE.fetch_sub(size, Ordering::Relaxed);
            dealloc(p.cast(), layout(size));
        }
        // Not ours: leak it rather than guess a layout. Counted and reported, because if this ever
        // fires the measurement is not trustworthy.
        None => {
            FOREIGN.fetch_add(1, Ordering::Relaxed);
        }
    }
}

unsafe extern "C" fn ts_malloc(size: usize) -> *mut c_void {
    tracked_alloc(size)
}

unsafe extern "C" fn ts_calloc(n: usize, size: usize) -> *mut c_void {
    // C's contract is to fail, not to truncate.
    let Some(total) = n.checked_mul(size) else {
        return std::ptr::null_mut();
    };
    let p = tracked_alloc(total);
    if !p.is_null() {
        std::ptr::write_bytes(p.cast::<u8>(), 0, total);
    }
    p
}

unsafe extern "C" fn ts_realloc(p: *mut c_void, size: usize) -> *mut c_void {
    if p.is_null() {
        return tracked_alloc(size);
    }
    let old = blocks(|m| m.get(&(p as usize)).copied());
    let q = tracked_alloc(size);
    if q.is_null() {
        return q;
    }
    match old {
        Some(old) => {
            std::ptr::copy_nonoverlapping(p.cast::<u8>(), q.cast::<u8>(), old.min(size));
            tracked_free(p);
        }
        None => {
            FOREIGN.fetch_add(1, Ordering::Relaxed);
        }
    }
    q
}

unsafe extern "C" fn ts_free(p: *mut c_void) {
    tracked_free(p);
}

/// The constant under test. Kept in sync by `parse_budget`'s
/// `the_worst_single_file_estimate_matches_the_ceiling_derivation`, which fails if the shipped value
/// changes without the ceiling being re-derived.
const SHIPPED_BYTES_PER_SOURCE_BYTE: f64 = 128.0;

/// Smaller files cannot move a ceiling, and their trees are dominated by fixed per-parse overhead —
/// a 500-byte file's tree is ~23 KB, which is 46 B/byte and pure noise for a constant meant to bound
/// large ones. Their total contribution is `threads x ~23 KB`, under a megabyte.
const MIN_BYTES: usize = 10_000;

struct Row {
    lang: String,
    live: usize,
    bytes: usize,
    lines: usize,
    tokens: usize,
    path: String,
}

/// Each identifier/number run is one token, each other non-whitespace byte is one.
fn count_tokens(src: &str) -> usize {
    let mut n = 0usize;
    let mut in_word = false;
    for b in src.bytes() {
        let word = b.is_ascii_alphanumeric() || b == b'_';
        if word {
            if !in_word {
                n += 1;
            }
        } else if !b.is_ascii_whitespace() {
            n += 1;
        }
        in_word = word;
    }
    n.max(1)
}

fn main() {
    // First statement of the process, before anything can have parsed. This is the whole reason
    // this is an example rather than a test — see the module doc.
    unsafe {
        tree_sitter::set_allocator(
            Some(ts_malloc),
            Some(ts_calloc),
            Some(ts_realloc),
            Some(ts_free),
        );
    }

    let roots: Vec<String> = std::env::args().skip(1).collect();
    if roots.is_empty() {
        eprintln!(
            "usage: cargo run --release --example calibrate_parse_budget -- <dir> [<dir>...]\n\
             \n\
             Measures exact tree-sitter tree bytes per file and reports which predictor bounds them.\n\
             Point it at as many and as varied a set of trees as you can: the two predictors this\n\
             replaced were each disproved by the next corpus tried."
        );
        std::process::exit(2);
    }

    let mut rows: Vec<Row> = Vec::new();
    for root in &roots {
        for entry in ignore::WalkBuilder::new(root).build().flatten() {
            if !entry.file_type().is_some_and(|t| t.is_file()) {
                continue;
            }
            let path = entry.path();
            let Some(lang) = tilth::__calibration::grammar_for(path) else {
                continue;
            };
            let Some(ts_lang) = tilth::__calibration::language_of(lang) else {
                continue;
            };
            let Ok(src) = std::fs::read_to_string(path) else {
                continue;
            };
            if src.len() < MIN_BYTES {
                continue;
            }

            let before = LIVE.load(Ordering::Relaxed);
            let Some(tree) = tilth::__calibration::parse(&src, lang, &ts_lang) else {
                continue;
            };
            let live = LIVE.load(Ordering::Relaxed).saturating_sub(before);
            drop(tree);

            rows.push(Row {
                lang: format!("{lang:?}"),
                live,
                bytes: src.len(),
                lines: src.lines().count().max(1),
                tokens: count_tokens(&src),
                path: path.display().to_string(),
            });
        }
    }

    if rows.len() < 8 {
        eprintln!(
            "only {} files >= {MIN_BYTES} B with a grammar under {}; too few to say anything",
            rows.len(),
            roots.join(", ")
        );
        std::process::exit(2);
    }

    let foreign = FOREIGN.load(Ordering::Relaxed);
    assert_eq!(
        foreign, 0,
        "{foreign} allocations were not ours, so something parsed before the allocator swap and \
         these numbers are not trustworthy"
    );

    let mut langs: Vec<&str> = rows.iter().map(|r| r.lang.as_str()).collect();
    langs.sort_unstable();
    langs.dedup();
    println!(
        "\n{} files >= {MIN_BYTES} B, grammars: {}",
        rows.len(),
        langs.join(", ")
    );

    let report = |name: &str, of: &dyn Fn(&Row) -> usize| -> f64 {
        let ratios: Vec<f64> = rows
            .iter()
            .map(|r| r.live as f64 / of(r).max(1) as f64)
            .collect();
        let max = ratios.iter().copied().fold(f64::MIN, f64::max);
        let min = ratios.iter().copied().fold(f64::MAX, f64::min);
        let mean_over = ratios.iter().map(|x| max / x).sum::<f64>() / ratios.len() as f64;
        println!(
            "  per {name:5}  max={max:12.1} B  spread={:9.1}x  mean over-estimate={mean_over:9.2}x",
            max / min
        );
        max
    };
    let per_byte = report("byte", &|r: &Row| r.bytes);
    report("line", &|r: &Row| r.lines);
    report("token", &|r: &Row| r.tokens);

    println!("\n  densest per source byte:");
    rows.sort_by(|a, b| {
        (b.live as f64 / b.bytes as f64).total_cmp(&(a.live as f64 / a.bytes as f64))
    });
    for r in rows.iter().take(5) {
        println!(
            "    {:6.1} B/byte  {:8} B  {:6} lines  {}",
            r.live as f64 / r.bytes as f64,
            r.bytes,
            r.lines,
            r.path
        );
    }

    println!();
    if per_byte > SHIPPED_BYTES_PER_SOURCE_BYTE {
        eprintln!(
            "FAIL: TREE_BYTES_PER_SOURCE_BYTE ({SHIPPED_BYTES_PER_SOURCE_BYTE}) does not bound this \
             corpus — {per_byte:.1} B/byte needed by\n      {}\n\
             The budget under-charges by that ratio, so raise the constant and re-derive the default \
             ceiling, which is a function of it.",
            rows[0].path
        );
        std::process::exit(1);
    }
    println!(
        "OK: TREE_BYTES_PER_SOURCE_BYTE ({SHIPPED_BYTES_PER_SOURCE_BYTE}) bounds this corpus \
         ({per_byte:.1} B/byte worst, {:.2}x margin).",
        SHIPPED_BYTES_PER_SOURCE_BYTE / per_byte
    );
}
