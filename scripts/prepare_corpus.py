#!/usr/bin/env python3
"""Build the SKlearnLM document corpus with the Python standard library."""
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path
import argparse
import hashlib
import json
import random
import re

VERSION = "1.10.dev0"
BASE_URL = "https://scikit-learn.org/dev/"
EOS = "<|endoftext|>"
BT = chr(96)
VOID = set("area base br col embed hr img input link meta param source track wbr".split())
NOISE = set((
    "headerlink viewcode-link sphx-glr-thumbcontainer sphx-glr-thumbnails "
    "sphx-glr-download-link-note sphx-glr-download-link sphx-glr-footer "
    "sphx-glr-signature sphx-glr-timing sk-top-container sk-container "
    "sk-text-repr-fallback"
).split())
CORE = {"api_reference", "user_guide", "examples"}
ALIASES = {
    "bicluster": "cluster", "biclustering": "cluster",
    "text": "feature_extraction", "neural_networks": "neural_network",
    "lda_qda": "discriminant_analysis", "sgd": "linear_model",
    "partial_dependence": "inspection", "permutation_importance": "inspection",
    "classification_threshold": "model_selection", "callbacks": "callback",
    "get_config": "configuration", "set_config": "configuration",
    "config_context": "configuration", "show_versions": "configuration",
    "clustering": "cluster", "cross_validation": "model_selection",
    "grid_search": "model_selection", "learning_curve": "model_selection",
    "model_evaluation": "metrics", "neural_networks_supervised": "neural_network",
    "neural_networks_unsupervised": "neural_network",
    "unsupervised_reduction": "decomposition", "density": "neighbors",
}
ROOT_GUIDES = set((
    "getting_started common_pitfalls model_persistence data_interoperability "
    "metadata_routing data_transforms model_selection inspection visualizations "
    "callbacks datasets supervised_learning unsupervised_learning computing machine_learning_map"
).split())


def sha(value):
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


class Node:
    def __init__(self, tag, attrs=(), parent=None):
        self.tag, self.attrs, self.parent = tag, dict(attrs), parent
        self.children = []

    @property
    def classes(self):
        return set(self.attrs.get("class", "").split())


class Tree(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node("root")
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        node = Node(tag, attrs, self.stack[-1])
        self.stack[-1].children.append(node)
        if tag not in VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                return

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def noisy(node):
    return (node.tag in {"script", "style", "nav", "aside", "footer", "button"}
            or bool(node.classes & NOISE)
            or node.attrs.get("aria-hidden") == "true" or "hidden" in node.attrs)


def walk(node):
    if isinstance(node, str) or noisy(node):
        return
    yield node
    for child in node.children:
        if isinstance(child, Node):
            yield from walk(child)


def plain(node):
    if isinstance(node, str):
        return node
    return "" if noisy(node) else "".join(plain(child) for child in node.children)


def line(value):
    return re.sub(r"\s+", " ", value).strip()


def code_span(value):
    value = line(value)
    longest = max((len(m[0]) for m in re.finditer(r"\x60+", value)), default=0)
    fence = BT * max(1, longest + 1)
    pad = " " if value.startswith(BT) or value.endswith(BT) else ""
    return fence + pad + value + pad + fence


def code_fence(value, language):
    longest = max((len(m[0]) for m in re.finditer(r"\x60+", value)), default=0)
    fence = BT * max(3, longest + 1)
    return fence + language + "\n" + value + ("" if value.endswith("\n") else "\n") + fence


def clean_layout(value):
    output, fence_size, blank = [], 0, False
    for item in value.split("\n"):
        match = re.match(r"^\s*(\x60{3,})(.*)$", item)
        if fence_size:
            output.append(item)
            if match and len(match[1]) >= fence_size and not match[2].strip():
                fence_size = 0
            continue
        if match:
            fence_size = len(match[1])
            output.append(item.rstrip())
            blank = False
        elif not item.strip():
            if not blank:
                output.append("")
            blank = True
        else:
            output.append(item.rstrip())
            blank = False
    if fence_size:
        raise ValueError("Unclosed code fence")
    return "\n".join(output).strip()


class Renderer:
    def __init__(self):
        self.code_blocks = []

    def children(self, node):
        return "".join(self.render(child) for child in node.children)

    def render(self, node):
        if isinstance(node, str):
            return re.sub(r"\s+", " ", node.replace("\xa0", " "))
        if noisy(node):
            return ""
        tag = node.tag
        if tag == "pre":
            value, language, cursor = plain(node), "text", node
            while cursor:
                found = sorted(c.removeprefix("highlight-").lower()
                               for c in cursor.classes if c.startswith("highlight-"))
                if found:
                    language = found[0]
                    break
                cursor = cursor.parent
            if value.lstrip().startswith(">>>"):
                language = "pycon"
            elif language in {"default", "none", "console"}:
                language = "text"
            self.code_blocks.append({"language": language, "text": value})
            return "\n\n" + code_fence(value, language) + "\n\n"
        if "math" in node.classes:
            value = plain(node).strip()
            return "\n\n" + value + "\n\n" if tag == "div" else value
        if tag == "table":
            rows = []
            for row in walk(node):
                if row.tag == "tr":
                    cells = [c for c in row.children if isinstance(c, Node) and c.tag in {"td", "th"}]
                    if cells:
                        rows.append([line(self.children(c)).replace("|", r"\|") for c in cells])
            if not rows:
                return ""
            width = max(map(len, rows))
            rows = [row + [""] * (width - len(row)) for row in rows]
            # A neutral header also works for dataframe outputs whose first row is data.
            rows.insert(0, [""] * width)
            output = ["| " + " | ".join(row) + " |" for row in rows]
            output.insert(1, "| " + " | ".join(["---"] * width) + " |")
            return "\n\n" + "\n".join(output) + "\n\n"
        if tag == "code" or (tag == "dt" and "sig" in node.classes):
            value = code_span(plain(node))
            return "\n\n" + value + "\n\n" if tag == "dt" else value
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            return "\n\n" + "#" * int(tag[1]) + " " + self.children(node).strip() + "\n\n"
        if tag in {"ul", "ol"}:
            items = [c for c in node.children if isinstance(c, Node) and c.tag == "li"]
            output = []
            for index, item in enumerate(items, 1):
                marker = f"{index}. " if tag == "ol" else "- "
                body = self.children(item).strip().splitlines()
                if body:
                    output.append(marker + body[0] + "".join(
                        "\n" + " " * len(marker) + text if text else "\n" for text in body[1:]
                    ))
            return "\n\n" + "\n".join(output) + "\n\n"
        if tag in {"strong", "b", "em", "i"}:
            value = self.children(node).strip()
            marker = "**" if tag in {"strong", "b"} else "*"
            return marker + value + marker if value else ""
        if tag == "span" and "classifier" in node.classes:
            return " : " + self.children(node)
        if tag == "img":
            alt = line(node.attrs.get("alt", ""))
            return "\n\n[Figure: " + alt + "]\n\n" if alt else ""
        if tag == "br":
            return "\n"
        if tag == "hr":
            return "\n\n---\n\n"
        body = self.children(node)
        if tag in {"p", "div", "section", "dt", "dd", "dl", "figure",
                   "figcaption", "blockquote", "details", "summary"}:
            return "\n\n" + body.strip() + "\n\n" if body.strip() else ""
        return body


def category_for(path):
    if path.startswith("modules/generated/"):
        return "api_reference"
    if path.startswith("auto_examples/"):
        return "examples"
    if path.startswith("whats_new/") or path == "whats_new.html":
        return "release_notes"
    if path.startswith("developers/"):
        return "developer_documentation"
    if path.startswith("api/"):
        return "api_index"
    if path.startswith(("modules/", "datasets/", "computing/")) or Path(path).stem in ROOT_GUIDES:
        return "user_guide"
    return "project"


def topic_for(path, category):
    if category in {"api_reference", "api_index"}:
        pieces = Path(path).stem.split(".")
        topic = pieces[1] if len(pieces) > 1 else "configuration"
    elif category == "examples":
        topic = Path(path).parts[1]
        if topic == "index.html":
            topic = "examples_overview"
    elif category == "user_guide":
        parts = Path(path).parts
        topic = parts[0] if parts[0] in {"datasets", "computing"} else Path(path).stem
    else:
        topic = category
    return ALIASES.get(topic, topic)


def extract_page(raw, path):
    from html import unescape
    match = re.search(
        r'<article\b[^>]*\bclass="[^"]*\bbd-article\b[^"]*"[^>]*>(.*?)</article>', raw, re.S
    )
    if not match and path == "index.html":
        match = re.search(r"<main\b[^>]*>(.*?)</main>", raw, re.S)
    if not match:
        return None
    tree = Tree()
    tree.feed(match[1])
    renderer = Renderer()
    text = clean_layout(renderer.render(tree.root))
    if not text:
        return None
    headers = [n for n in walk(tree.root) if n.tag == "h1"]
    if headers:
        title = line(plain(headers[0]))
    else:
        title_match = re.search(r"<title>(.*?)</title>", raw, re.S)
        title = re.sub(r"\s*[—–-]\s*scikit-learn .*? documentation\s*$", "",
                       unescape(title_match[1])).strip() if title_match else path
        text = "# " + title + "\n\n" + text
    if EOS in text or "\ufffd" in text or "\x00" in text:
        raise ValueError(f"Reserved boundary token or invalid character in {path}")
    category = category_for(path)
    topic = topic_for(path, category)
    return {
        "id": sha(path)[:20], "title": title, "source_path": path,
        "source_url": BASE_URL + path, "version": VERSION, "category": category,
        "topic": topic,
        "group_id": ("topic/" if category in CORE | {"api_index"} else "project/") + topic,
        "split": None, "text": text, "text_sha256": sha(text),
        "code_block_count": len(renderer.code_blocks), "license": "BSD-3-Clause",
    }


def extract_documents(html_root):
    root = Path(html_root)
    documents, skipped = [], []
    for path in sorted(root.rglob("*.html")):
        relative = path.relative_to(root).as_posix()
        raw = path.read_text(encoding="utf-8")
        document = extract_page(raw, relative)
        if document:
            documents.append(document)
        else:
            skipped.append({
                "source_path": relative,
                "reason": "redirect" if re.search(r'http-equiv=["\']refresh', raw, re.I)
                else "no_document_text",
            })
    raw_index = (root / "searchindex.js").read_bytes()
    index = raw_index.decode("utf-8")
    index = json.loads(index[index.index("(") + 1:index.rfind(")")])
    expected = {name + ".html" for name in index["docnames"]}
    exported = {row["source_path"] for row in documents}
    missing = expected - exported
    if missing != {"whats_new/_contributors.html"} or exported - expected:
        raise ValueError(f"Unexpected index mismatch: missing={sorted(missing)}, extra={sorted(exported - expected)}")
    return documents, {
        "indexed_pages": len(expected), "exported_pages": len(exported),
        "internal_definition_pages_omitted": sorted(missing),
        "source_searchindex_sha256": sha(raw_index), "skipped_html_files": skipped,
    }


def assign_splits(documents, seed=42):
    """Keep whole topic families together; approximately 90/5/5 by characters."""
    groups = defaultdict(list)
    for row in documents:
        groups[row["group_id"]].append(row)
    eligible = sorted(group for group, rows in groups.items()
                      if any(row["category"] in CORE for row in rows))
    if len(eligible) < 8:
        raise ValueError("At least eight core topic groups are needed for grouped holdouts")
    random.Random(seed).shuffle(eligible)
    priority = {group: index for index, group in enumerate(eligible)}
    weights = {group: sum(len(row["text"]) for row in rows) for group, rows in groups.items()}
    target = sum(weights.values()) * 0.05
    assignments = {group: "train" for group in groups}
    available = set(eligible)
    for split in ("validation", "test"):
        current, selected = 0, []
        while available:
            candidates = list(available)
            if len(selected) < 2:
                smaller = [group for group in candidates if weights[group] <= target * 0.45]
                if smaller:
                    candidates = smaller
            best = min(candidates, key=lambda group: (
                abs(target - current - weights[group]), priority[group], group
            ))
            if len(selected) >= 3 and abs(target - current - weights[best]) >= abs(target - current):
                break
            selected.append(best)
            available.remove(best)
            current += weights[best]
            assignments[best] = split
    for row in documents:
        row["split"] = assignments[row["group_id"]]


def load_jsonl(path):
    return [json.loads(item) for item in Path(path).read_text(encoding="utf-8").splitlines()
            if item.strip()]


def encode_documents(documents):
    return "\n\n".join(row["text"] + "\n" + EOS for row in documents) + "\n"


def build_outputs(documents, source_metadata, seed=42):
    documents = sorted(documents, key=lambda row: row["source_path"])
    assign_splits(documents, seed)
    outputs = {
        "documents.jsonl": "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                                   for row in documents),
        "corpus.txt": encode_documents(documents),
    }
    statistics = {
        "documents": len(documents),
        "categories": dict(sorted(Counter(row["category"] for row in documents).items())),
        "code_blocks": sum(row["code_block_count"] for row in documents),
        "splits": {},
    }
    for split in ("train", "validation", "test"):
        rows = [row for row in documents if row["split"] == split]
        random.Random(f"{seed}:{split}").shuffle(rows)
        outputs[split + ".txt"] = encode_documents(rows)
        statistics["splits"][split] = {
            "documents": len(rows), "groups": sorted({row["group_id"] for row in rows}),
            "body_characters": sum(len(row["text"]) for row in rows),
            "bytes": len(outputs[split + ".txt"].encode("utf-8")),
        }
    manifest = {
        "schema_version": 1, "documentation_version": VERSION,
        "source_base_url": BASE_URL, "source_revision_linked_by_api_pages": "4e624cfb1",
        "license": "BSD-3-Clause",
        "text_format": "Markdown with fenced code, tables and original LaTeX equations",
        "special_tokens_to_register": [EOS], "document_boundary_token": EOS, "seed": seed,
        "split_strategy": "whole topic families; approximately 90/5/5 by characters",
        "split_targets": {"train": 0.9, "validation": 0.05, "test": 0.05},
        "split_limitations": (
            "Topic grouping uses filename heuristics. Shared API boilerplate and "
            "cross-topic examples may still overlap between splits."
        ),
        "source_metadata": source_metadata, "statistics": statistics,
        "files": {name: {"bytes": len(value.encode("utf-8")), "sha256": sha(value)}
                  for name, value in sorted(outputs.items())},
    }
    outputs["manifest.json"] = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    return outputs, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--html-root", type=Path)
    source.add_argument("--from-jsonl", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.html_root:
        documents, metadata = extract_documents(args.html_root)
    else:
        documents = load_jsonl(args.from_jsonl)
        manifest_path = args.from_jsonl.parent / "manifest.json"
        metadata = json.loads(manifest_path.read_text())["source_metadata"]
    outputs, manifest = build_outputs(documents, metadata, args.seed)
    if not args.dry_run:
        args.output.mkdir(parents=True, exist_ok=True)
        for name, value in outputs.items():
            target = args.output / name
            temporary = target.with_name(target.name + ".tmp")
            temporary.write_text(value, encoding="utf-8", newline="\n")
            temporary.replace(target)
    print(json.dumps({"dry_run": args.dry_run, "output": str(args.output),
                      "statistics": manifest["statistics"], "files": manifest["files"]}, indent=2))


if __name__ == "__main__":
    main()
