#!/usr/bin/env python3
"""Build the three Seahorse explainer pages (vedantjumle.com/seahorse/<page>/) from the result files.

    python3 tools/manuscript/explainers/build_explainers.py --results results \
        --out ../personal-website/seahorse

Writes <out>/binding/index.html, <out>/subject-position/index.html and <out>/automatic-selection/index.html.
Each data page has a template.html with JSON placeholders (__DATA__, and __STATS__ on the subject-position page)
and an extract.py that computes them from results/; the numbers in the charts, tables and tiles come from there,
never retyped. The binding page is a static explainer (page.html, copied as is). The main project page is built
separately by tools/manuscript/build.py and links to these three.

Runs used (all local, under results/):
  dream_subj_20261022_1755            the subject-position test (Qwen2.5-1.5B, teacher-forced; meta.json,
                                      rows.jsonl.gz, verdict.json)
  dream_select_20261104_0032          automatic slice selection, teacher-forced, Qwen2.5-1.5B (and the examples)
  dream_select_cluster_20261104_0129  automatic slice selection on DelftBlue: Qwen2.5-7B and Llama-3.2-3B
                                      teacher-forced (<model>/tf/), and judged generation for all three models
                                      (<model>/verdict_gen.json)
Each extract.py also runs on its own: python3 -I <page>/extract.py <results> <out.json>.
Only the standard library.
"""

import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (source folder, output folder)
STATIC = [("binding", "binding")]
TEMPLATED = [("subject_position", "subject-position"), ("automatic_selection", "automatic-selection")]


def load_extract(folder):
    spec = importlib.util.spec_from_file_location(f"explainer_{folder}", HERE / folder / "extract.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def as_script_json(obj):
    """JSON for a <script type="application/json"> block (no '</' that could close the script)."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def fill(template, values):
    for key, obj in values.items():
        n = template.count(key)
        if n != 1:
            raise SystemExit(f"placeholder {key} appears {n} times in the template (need exactly 1)")
        template = template.replace(key, as_script_json(obj))
    if "__DATA__" in template or "__STATS__" in template:
        raise SystemExit("unfilled placeholder left in the page")
    return template


def write(out, folder, text):
    path = out / folder / "index.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path} ({len(text.encode('utf-8')):,} bytes)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=HERE.parents[2] / "results")
    ap.add_argument("--out", type=Path, required=True, help="the website's seahorse/ folder")
    a = ap.parse_args()
    for src, dst in STATIC:
        write(a.out, dst, (HERE / src / "page.html").read_text(encoding="utf-8"))
    for src, dst in TEMPLATED:
        values = load_extract(src).placeholders(a.results)
        write(a.out, dst, fill((HERE / src / "template.html").read_text(encoding="utf-8"), values))


if __name__ == "__main__":
    main()
