"""Build versioned graphs from Chromium's live accessibility tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
from bs4 import BeautifulSoup
from torch_geometric.data import Data

from .collection_selection import select_multilabel_stratified_sites, select_split_sites
from .data import split_hash


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "learning_v1" / "src"))

from html_graph_builder import DOMNode, EDGE_PARENT_CHILD, EDGE_SIBLING, get_tag_index  # noqa: E402
from wcag_rules import NUM_RULES, RULE_INDEX, rule_mask_for_graph_source  # noqa: E402


LIVE_AX_FEATURE_VERSION = 1
EDGE_AX_RELATION = 3
MARKER_ATTRIBUTE = "data-dissertation-ax-index"
STATE_NAMES = ("focusable", "focused", "disabled", "expanded", "required", "selected", "checked", "pressed", "modal", "invalid")
RELATION_NAMES = ("labelledby", "describedby", "controls", "owns", "flowto")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _value(item: dict[str, Any] | None, default: Any = "") -> Any:
    return (item or {}).get("value", default)


def _walk_dom(node: dict[str, Any], backend_to_dom: dict[int, int]) -> None:
    attributes = node.get("attributes", [])
    attrs = {str(attributes[index]): str(attributes[index + 1]) for index in range(0, len(attributes) - 1, 2)}
    if MARKER_ATTRIBUTE in attrs and node.get("backendNodeId") is not None:
        backend_to_dom[int(node["backendNodeId"])] = int(attrs[MARKER_ATTRIBUTE])
    for key in ("children", "shadowRoots", "pseudoElements"):
        for child in node.get(key, []) or []:
            _walk_dom(child, backend_to_dom)
    if node.get("contentDocument"):
        _walk_dom(node["contentDocument"], backend_to_dom)


def _node_features(ax: dict[str, Any], dom: dict[str, Any] | None) -> tuple[torch.Tensor, int, str]:
    role = str(_value(ax.get("role"), "")); name = str(_value(ax.get("name"), ""))
    soup = BeautifulSoup("<span></span>", "lxml")
    tag_name = str((dom or {}).get("tag", "span")) or "span"
    tag = soup.new_tag(tag_name)
    for key, value in (dom or {}).get("attrs", {}).items():
        if key != MARKER_ATTRIBUTE:
            tag.attrs[key] = value
    if role:
        tag.attrs["role"] = role
    tag.string = str((dom or {}).get("text", name))[:2000]
    node = DOMNode(tag, 0)
    if dom:
        node.bbox = {key: float(dom.get(key, -1)) for key in ("x", "y", "width", "height")}
        node.is_visible = bool(dom.get("visible", False))
    base = node.get_attribute_features(False)
    properties = {str(item.get("name", "")): _value(item.get("value")) for item in ax.get("properties", [])}
    live = [min(len(name) / 300.0, 1.0), float(bool(name))]
    live.extend(float(bool(properties.get(state, False))) for state in STATE_NAMES)
    return torch.cat([base, torch.tensor(live, dtype=torch.float32)]), get_tag_index(tag_name), name


def capture_site(site_dir: Path, output: Path, text_model) -> dict[str, Any]:
    from playwright.sync_api import sync_playwright

    html_path = site_dir / "0.html"; axe_path = site_dir / "page-0_home.json"
    report = json.loads(axe_path.read_text(encoding="utf-8"))
    selectors = sorted({
        str(target[0]) for violation in report.get("violations", []) for node in violation.get("nodes", [])
        for target in [node.get("target", [])] if target and isinstance(target[0], str)
    })
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 720}, reduced_motion="reduce")
        page.route("**/*", lambda route: route.abort() if route.request.url.startswith(("http://", "https://")) else route.continue_())
        page.goto(html_path.resolve().as_uri(), wait_until="domcontentloaded")
        dom_records = page.evaluate(f"""() => {{
          const elements=[...document.querySelectorAll('*')]; const index=new Map(elements.map((el,i)=>[el,i]));
          return elements.map((el,i)=>{{ el.setAttribute('{MARKER_ATTRIBUTE}', String(i)); const r=el.getBoundingClientRect(); const s=getComputedStyle(el);
            return {{index:i,parent_index:index.has(el.parentElement)?index.get(el.parentElement):null,tag:el.tagName.toLowerCase(),
              attrs:Object.fromEntries([...el.attributes].map(a=>[a.name,a.value])),text:(el.innerText||el.textContent||'').trim().slice(0,2000),
              x:r.x,y:r.y,width:r.width,height:r.height,visible:s.display!=='none'&&s.visibility!=='hidden'&&Number(s.opacity)>0&&r.width>0&&r.height>0}};
          }}); }}""")
        session = page.context.new_cdp_session(page)
        document = session.send("DOM.getDocument", {"depth": -1, "pierce": True})["root"]
        backend_to_dom: dict[int, int] = {}; _walk_dom(document, backend_to_dom)
        ax_raw = session.send("Accessibility.getFullAXTree").get("nodes", [])
        selector_to_dom = page.evaluate("""selectors => Object.fromEntries(selectors.map(selector => { try {
          const el=document.querySelector(selector); return [selector, el ? Number(el.getAttribute('data-dissertation-ax-index')) : null];
        } catch (_) { return [selector, null]; }}))""", selectors)
        page.evaluate(f"() => document.querySelectorAll('[{MARKER_ATTRIBUTE}]').forEach(el => el.removeAttribute('{MARKER_ATTRIBUTE}'))")
        browser.close()
    dom_by_index = {int(item["index"]): item for item in dom_records}
    included = [item for item in ax_raw if not item.get("ignored")]
    ax_index = {str(item["nodeId"]): index for index, item in enumerate(included)}
    backend_to_ax = {int(item["backendDOMNodeId"]): index for index, item in enumerate(included) if item.get("backendDOMNodeId") is not None}
    dom_to_ax = {backend_to_dom[backend]: index for backend, index in backend_to_ax.items() if backend in backend_to_dom}
    base_rows = []; tag_indices = []; backend_ids = []; dom_indices = []; names = []
    for item in included:
        backend = int(item["backendDOMNodeId"]) if item.get("backendDOMNodeId") is not None else -1
        dom_index = backend_to_dom.get(backend, -1)
        features, tag_index, name = _node_features(item, dom_by_index.get(dom_index))
        base_rows.append(features); tag_indices.append(tag_index); backend_ids.append(backend); dom_indices.append(dom_index); names.append(name)
    embeddings = text_model.encode(names, batch_size=64, convert_to_tensor=True, show_progress_bar=False).detach().cpu().float()
    feature_rows = torch.cat([torch.stack(base_rows), embeddings], dim=1)
    edge_src = []; edge_dst = []; edge_types = []
    for parent_index, item in enumerate(included):
        children = [ax_index[child] for child in item.get("childIds", []) if child in ax_index]
        for child in children:
            edge_src.append(parent_index); edge_dst.append(child); edge_types.append(EDGE_PARENT_CHILD)
        for left, right in zip(children, children[1:]):
            edge_src.extend([left, right]); edge_dst.extend([right, left]); edge_types.extend([EDGE_SIBLING, EDGE_SIBLING])
        for prop in item.get("properties", []):
            if prop.get("name") not in RELATION_NAMES:
                continue
            for related in (prop.get("value") or {}).get("relatedNodes", []) or []:
                backend = related.get("backendDOMNodeId")
                if backend in backend_to_ax:
                    edge_src.append(parent_index); edge_dst.append(backend_to_ax[backend]); edge_types.append(EDGE_AX_RELATION)
    labels = torch.zeros((len(included), NUM_RULES), dtype=torch.float32)
    mapping_loss = []
    for violation in report.get("violations", []):
        rule_id = str(violation.get("id", ""))
        if rule_id not in RULE_INDEX:
            continue
        for violation_node in violation.get("nodes", []):
            target = violation_node.get("target", [])
            selector = str(target[0]) if target and isinstance(target[0], str) else ""
            dom_index = selector_to_dom.get(selector)
            current = int(dom_index) if dom_index is not None else -1
            while current >= 0 and current not in dom_to_ax:
                parent = dom_by_index.get(current, {}).get("parent_index")
                current = int(parent) if parent is not None else -1
            if current in dom_to_ax:
                labels[dom_to_ax[current], RULE_INDEX[rule_id]] = 1.0
            else:
                mapping_loss.append({"rule_id": rule_id, "selector": selector})
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long) if edge_src else torch.empty((2, 0), dtype=torch.long)
    data = Data(
        x=feature_rows, edge_index=edge_index, edge_type=torch.tensor(edge_types, dtype=torch.long),
        tag_indices=torch.tensor(tag_indices, dtype=torch.long), node_y_multi=labels,
        node_y=labels.bool().any(dim=1).long(), y=torch.tensor([int(labels.any())]), num_nodes=len(included),
    )
    data.graph_source = "a11y-tree"; data.live_accessibility_tree = True; data.live_ax_feature_version = LIVE_AX_FEATURE_VERSION
    data.backend_dom_node_ids = torch.tensor(backend_ids, dtype=torch.long); data.dom_indices = torch.tensor(dom_indices, dtype=torch.long)
    data.available_rule_mask = rule_mask_for_graph_source("a11y-tree").unsqueeze(0)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"data": data, "html_path": str(html_path), "num_nodes": len(included), "graph_source": "a11y-tree", "collector": "chromium-cdp-live-ax-v1"}, output)
    return {
        "site_id": site_dir.name, "status": "captured", "path": str(output), "nodes": len(included),
        "edges": len(edge_types), "feature_dim": int(data.x.shape[1]), "feature_version": LIVE_AX_FEATURE_VERSION,
        "dom_mapped_nodes": sum(index >= 0 for index in dom_indices), "positive_labels": int(labels.sum()),
        "mapping_loss_count": len(mapping_loss), "mapping_loss_examples": mapping_loss[:20],
        "source_html_sha256": _sha256(html_path), "source_axe_sha256": _sha256(axe_path),
    }


def run(args: argparse.Namespace) -> dict:
    from sentence_transformers import SentenceTransformer

    split = json.loads(args.split.read_text(encoding="utf-8"))
    selection_rules = set(args.selection_rules or [])
    positive_sites_by_rule = {rule: set() for rule in selection_rules}
    if selection_rules:
        for partition in ("train", "val", "test"):
            for site in split[partition]:
                report_path = args.corpus_dir / site / "page-0_home.json"
                if report_path.is_file():
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    found = {str(item.get("id")) for item in report.get("violations", [])}
                    for rule in selection_rules & found:
                        positive_sites_by_rule[rule].add(site)
    ordered = select_multilabel_stratified_sites(
        split, positive_sites_by_rule, args.max_sites, args.positive_fraction, args.minimum_positive_sites_per_rule,
    ) if selection_rules else select_split_sites(split, args.max_sites)
    selected_split = {
        "seed": split.get("seed", 42),
        **{partition: [site for name, site in ordered if name == partition] for partition in ("train", "val", "test")},
    }
    selected_split["split_hash"] = split_hash(selected_split)
    text_model = SentenceTransformer("all-MiniLM-L6-v2", device=args.device)
    old_manifest_path = args.output_dir / "live_ax_cache_manifest.json"
    old_records = {}
    if old_manifest_path.is_file():
        old_records = {item["site_id"]: item for item in json.loads(old_manifest_path.read_text(encoding="utf-8")).get("records", [])}
    records = []
    for index, (partition, site_id) in enumerate(ordered, 1):
        output = args.output_dir / site_id / "a11y-tree.pt"
        print(f"[{index}/{len(ordered)}] {partition} {site_id}", flush=True)
        if args.resume and output.is_file():
            try:
                raw = torch.load(output, map_location="cpu", weights_only=False)["data"]
                if bool(getattr(raw, "live_accessibility_tree", False)) and int(getattr(raw, "live_ax_feature_version", 0)) == LIVE_AX_FEATURE_VERSION:
                    record = dict(old_records.get(site_id, {}))
                    record.update({"site_id": site_id, "partition": partition, "status": "reused", "path": str(output), "nodes": int(raw.num_nodes), "edges": int(raw.edge_index.shape[1]), "feature_dim": int(raw.x.shape[1])})
                    records.append(record)
                    continue
            except Exception:
                pass
        site_dir = args.corpus_dir / site_id
        if not (site_dir / "0.html").is_file() or not (site_dir / "page-0_home.json").is_file():
            records.append({"site_id": site_id, "partition": partition, "status": "missing_source"}); continue
        try:
            record = capture_site(site_dir, output, text_model); record["partition"] = partition; records.append(record)
        except Exception as exc:
            records.append({"site_id": site_id, "partition": partition, "status": "collection_failed", "error": f"{type(exc).__name__}: {exc}"[:1000]})
    counts = {status: sum(item["status"] == status for item in records) for status in sorted({item["status"] for item in records})}
    manifest = {
        "schema_version": 1, "collector": "chromium-cdp-live-ax-v1", "live_ax_feature_version": LIVE_AX_FEATURE_VERSION,
        "split": str(args.split.resolve()), "split_sha256": _sha256(args.split), "split_hash": split.get("split_hash"),
        "corpus_dir": str(args.corpus_dir.resolve()), "requested_sites": len(ordered), "outcome_counts": counts, "records": records,
        "selection": {
            "rules": sorted(selection_rules), "positive_fraction": args.positive_fraction,
            "minimum_positive_sites_per_rule": args.minimum_positive_sites_per_rule,
            "available_site_support": {rule: len(sites) for rule, sites in sorted(positive_sites_by_rule.items())},
            "selected_site_support": {rule: sum(site in sites for _, site in ordered) for rule, sites in sorted(positive_sites_by_rule.items())},
            "selected_partition_support": {
                partition: {
                    rule: sum(name == partition and site in sites for name, site in ordered)
                    for rule, sites in sorted(positive_sites_by_rule.items())
                }
                for partition in ("train", "val", "test")
            },
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "collection_split.json").write_text(json.dumps(selected_split, indent=2), encoding="utf-8")
    (args.output_dir / "live_ax_cache_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True); parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-sites", type=int); parser.add_argument("--resume", action="store_true")
    parser.add_argument("--selection-rules", nargs="+", help="Predeclared axe rules used only to ensure positive/negative pilot support")
    parser.add_argument("--positive-fraction", type=float, default=0.6)
    parser.add_argument("--minimum-positive-sites-per-rule", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    report = run(parse_args()); print(json.dumps({"requested_sites": report["requested_sites"], "outcomes": report["outcome_counts"]}, indent=2))


if __name__ == "__main__":
    main()
