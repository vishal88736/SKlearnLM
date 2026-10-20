#!/usr/bin/env python3
"""Validate published corpus files, split separation and optional notebook preservation."""
import argparse
import ast
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import zlib

from prepare_corpus import EOS, build_outputs, load_jsonl, sha


def normalize_code(value):
    return re.sub(r"\n{3,}", "\n\n", "\n".join(
        line.rstrip() for line in value.replace("\r\n", "\n").splitlines()
    )).strip()


def audit(data, html_root=None):
    documents = load_jsonl(data / "documents.jsonl")
    manifest = json.loads((data / "manifest.json").read_text())
    assert len(documents) == 1075, "Unexpected document count"
    assert len({row["id"] for row in documents}) == len(documents), "Duplicate document IDs"
    assert len({row["source_path"] for row in documents}) == len(documents), "Duplicate source pages"
    assert sum(row["category"] == "api_reference" for row in documents) == 584
    group_splits = defaultdict(set)
    bodies = Counter()
    paragraph_splits = defaultdict(set)
    for row in documents:
        assert row["split"] in {"train", "validation", "test"}
        assert isinstance(row["text"], str) and row["text"].startswith("# ")
        assert sha(row["text"]) == row["text_sha256"], row["source_path"]
        assert EOS not in row["text"] and "\ufffd" not in row["text"] and "\x00" not in row["text"]
        assert row["source_url"].endswith(row["source_path"])
        group_splits[row["group_id"]].add(row["split"])
        bodies[row["text"]] += 1
        for block in re.split(r"\n\s*\n", row["text"]):
            value = re.sub(r"\s+", " ", block).strip()
            if len(value) >= 160:
                paragraph_splits[value].add(row["split"])
    assert all(len(values) == 1 for values in group_splits.values()), "Topic group crosses splits"
    assert all(count == 1 for count in bodies.values()), "Duplicate whole documents"
    for name, details in manifest["files"].items():
        content = (data / name).read_bytes()
        assert len(content) == details["bytes"] and sha(content) == details["sha256"], name
    for split in ["corpus", "train", "validation", "test"]:
        content = (data / (split + ".txt")).read_text()
        rows = documents if split == "corpus" else [row for row in documents if row["split"] == split]
        assert content.count(EOS) == len(rows), f"Boundary count: {split}"
        parts = [part.strip() for part in content.split(EOS) if part.strip()]
        assert Counter(parts) == Counter(row["text"] for row in rows), f"Membership mismatch: {split}"
        assert rows, f"Empty split: {split}"
    rebuilt, _ = build_outputs(documents, manifest["source_metadata"], manifest["seed"])
    for name, content in rebuilt.items():
        assert (data / name).read_bytes() == content.encode("utf-8"), f"Rebuild differs: {name}"
    result = {
        "documents": len(documents), "api_pages": 584,
        "groups_disjoint": True, "file_hashes_valid": True, "rebuild_identical": True,
        "splits": manifest["statistics"]["splits"],
        "shared_long_blocks_across_splits": sum(len(values) > 1 for values in paragraph_splits.values()),
        "note": "Shared generic API prose can remain even when topic groups are disjoint.",
    }
    if html_root:
        lookup = {row["source_path"]: row["text"] for row in documents}
        cells = notebooks = syntax = 0
        for path in sorted((html_root / "notebooks").rglob("*.ipynb")):
            notebook = json.loads(path.read_text())
            target = path.relative_to(html_root / "notebooks").with_suffix(".html").as_posix()
            body = normalize_code(lookup[target])
            code = []
            for index, cell in enumerate(notebook.get("cells", [])):
                if cell.get("cell_type") != "code":
                    continue
                value = cell.get("source", "")
                value = "".join(value) if isinstance(value, list) else value
                if not value.strip():
                    continue
                assert normalize_code(value) in body, f"Missing code: {target}, cell {index}"
                code.append(value)
                cells += 1
            ast.parse("\n\n".join(code))
            syntax += 1
            notebooks += 1
        assert (notebooks, cells, syntax) == (281, 1601, 281)
        result["notebooks"] = notebooks
        result["code_cells_preserved"] = cells
        result["notebook_code_syntax_checks"] = syntax
        inventory = (html_root / "objects.inv").read_bytes().split(b"\n", 4)
        items = 0
        for item in zlib.decompress(inventory[4]).decode().splitlines():
            name, kind, _, target, _ = item.split(None, 4)
            if kind.startswith("py:"):
                target = target.split("#")[0]
                assert target in lookup and name.rsplit(".", 1)[-1] in lookup[target], name
                items += 1
        result["python_inventory_items_present"] = items
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--html-root", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = json.dumps(audit(args.data, args.html_root), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.report:
        args.report.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
