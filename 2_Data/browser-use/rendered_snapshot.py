"""Capture aligned HTML, visual, accessibility-tree, and screenshot evidence."""

import json
import base64
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SNAPSHOT_VERSION = 1


CAPTURE_SCRIPT = r"""() => {
    const ignoredTags = new Set(['script', 'style', 'noscript']);
    const maxColorDelta = Math.sqrt(3 * 255 * 255);

    function parseColor(value) {
        const match = String(value || '').match(/rgba?\(([^)]+)\)/);
        if (!match) return [0, 0, 0, 0];
        const parts = match[1].split(',').map((part) => part.trim());
        const r = Number.parseFloat(parts[0]) || 0;
        const g = Number.parseFloat(parts[1]) || 0;
        const b = Number.parseFloat(parts[2]) || 0;
        const a = parts.length >= 4 ? Number.parseFloat(parts[3]) : 1;
        return [r, g, b, Number.isFinite(a) ? a : 1];
    }

    function composite(foreground, background) {
        const alpha = foreground[3] + background[3] * (1 - foreground[3]);
        if (alpha <= 0) return [0, 0, 0, 0];
        return [
            (foreground[0] * foreground[3] + background[0] * background[3] * (1 - foreground[3])) / alpha,
            (foreground[1] * foreground[3] + background[1] * background[3] * (1 - foreground[3])) / alpha,
            (foreground[2] * foreground[3] + background[2] * background[3] * (1 - foreground[3])) / alpha,
            alpha,
        ];
    }

    function resolvedBackground(element) {
        const chain = [];
        let current = element;
        while (current && current.nodeType === Node.ELEMENT_NODE) {
            chain.push(parseColor(getComputedStyle(current).backgroundColor));
            current = current.parentElement;
        }
        let color = [255, 255, 255, 1];
        for (let index = chain.length - 1; index >= 0; index -= 1) {
            color = composite(chain[index], color);
        }
        return color;
    }

    function luminance(rgb) {
        const values = rgb.slice(0, 3).map((channel) => {
            const value = channel / 255;
            return value <= 0.04045
                ? value / 12.92
                : Math.pow((value + 0.055) / 1.055, 2.4);
        });
        return 0.2126 * values[0] + 0.7152 * values[1] + 0.0722 * values[2];
    }

    function contrastRatio(foreground, background) {
        const first = luminance(foreground);
        const second = luminance(background);
        return (Math.max(first, second) + 0.05) / (Math.min(first, second) + 0.05);
    }

    function colorDistance(first, second) {
        return Math.sqrt(
            Math.pow(first[0] - second[0], 2) +
            Math.pow(first[1] - second[1], 2) +
            Math.pow(first[2] - second[2], 2)
        );
    }

    function isFocusable(node, style) {
        if (node.matches('a[href], button, input, select, textarea, summary, iframe, object, embed')) {
            return true;
        }
        const tabindex = node.getAttribute('tabindex');
        return tabindex !== null && Number.parseInt(tabindex, 10) >= 0 && style.visibility !== 'hidden';
    }

    function isScrollable(node, style) {
        const overflow = `${style.overflow} ${style.overflowX} ${style.overflowY}`;
        return /(auto|scroll)/.test(overflow) && (
            node.scrollHeight > node.clientHeight || node.scrollWidth > node.clientWidth
        );
    }

    const elements = Array.from(document.querySelectorAll('*')).filter(
        (node) => !ignoredTags.has(node.tagName.toLowerCase())
    );
    const oldMarkers = elements.map((node) => ({
        node,
        hadMarker: node.hasAttribute('data-gnn-node-id'),
        marker: node.getAttribute('data-gnn-node-id'),
    }));
    elements.forEach((node, index) => node.setAttribute('data-gnn-node-id', String(index)));

    const nodes = elements.map((node, index) => {
        const rect = node.getBoundingClientRect();
        const style = window.getComputedStyle(node);
        const background = resolvedBackground(node);
        const foregroundRaw = parseColor(style.color);
        const foreground = foregroundRaw[3] < 1 ? composite(foregroundRaw, background) : foregroundRaw;
        const parentStyle = node.parentElement ? window.getComputedStyle(node.parentElement) : style;
        const parentForegroundRaw = parseColor(parentStyle.color);
        const parentForeground = parentForegroundRaw[3] < 1
            ? composite(parentForegroundRaw, background)
            : parentForegroundRaw;
        const opacity = Number.parseFloat(style.opacity);
        const fontSize = Number.parseFloat(style.fontSize);
        const fontWeight = Number.parseFloat(style.fontWeight);
        const isLargeText = fontSize >= 24 || (fontSize >= (14 * 96 / 72) && fontWeight >= 700);
        const requiredContrastRatio = isLargeText ? 3 : 4.5;
        const measuredContrastRatio = contrastRatio(foreground, background);
        const contrastDeficit = Math.max(requiredContrastRatio - measuredContrastRatio, 0) / requiredContrastRatio;
        const hasDirectText = Array.from(node.childNodes).some(
            (child) => child.nodeType === Node.TEXT_NODE && child.textContent.trim().length > 0
        );
        const hasBox = rect.width > 0 && rect.height > 0;
        const inViewport = rect.bottom > 0 && rect.right > 0 && rect.left < innerWidth && rect.top < innerHeight;
        const visible = hasBox && style.display !== 'none' && style.visibility !== 'hidden' && opacity > 0;
        const isLink = node.matches('a, [role="link"]');
        const linkColorDelta = isLink ? colorDistance(foreground, parentForeground) : -1;

        return {
            snapshot_node_id: String(index),
            x: rect.x,
            y: rect.y,
            width: rect.width,
            height: rect.height,
            visible,
            foreground_rgb: foreground.slice(0, 3).map((value) => Math.round(value)),
            background_rgb: background.slice(0, 3).map((value) => Math.round(value)),
            contrast_ratio: measuredContrastRatio,
            font_size: Number.isFinite(fontSize) ? fontSize : -1,
            font_weight: Number.isFinite(fontWeight) ? fontWeight : -1,
            required_contrast_ratio: requiredContrastRatio,
            contrast_deficit: contrastDeficit,
            has_direct_text: hasDirectText,
            opacity: Number.isFinite(opacity) ? opacity : -1,
            text_decoration_underline: style.textDecorationLine.includes('underline'),
            link_color_delta: Number.isFinite(linkColorDelta) ? linkColorDelta : -1,
            link_color_delta_normalized: Number.isFinite(linkColorDelta) && linkColorDelta >= 0
                ? linkColorDelta / maxColorDelta
                : -1,
            scrollable: isScrollable(node, style),
            focusable: isFocusable(node, style),
            clipped: rect.left < 0 || rect.top < 0 || rect.right > innerWidth || rect.bottom > innerHeight,
            in_viewport: inViewport,
        };
    });

    const doctype = document.doctype
        ? `<!DOCTYPE ${document.doctype.name}${document.doctype.publicId ? ` PUBLIC "${document.doctype.publicId}"` : ''}${document.doctype.systemId ? ` "${document.doctype.systemId}"` : ''}>\n`
        : '';
    const html = doctype + document.documentElement.outerHTML;

    // Keep the markers attached until Python has captured the DOM and AX tree.
    // Store node references in the page so cleanup can restore pre-existing values.
    globalThis.__gnnSnapshotOldMarkers = oldMarkers;

    return {
        html,
        visual: {
            version: 1,
            captured_url: location.href,
            viewport: { width: innerWidth, height: innerHeight },
            nodes,
        },
    };
}"""


CLEANUP_SCRIPT = r"""() => {
    const oldMarkers = globalThis.__gnnSnapshotOldMarkers || [];
    for (const oldMarker of oldMarkers) {
        if (oldMarker.hadMarker) {
            oldMarker.node.setAttribute('data-gnn-node-id', oldMarker.marker);
        } else {
            oldMarker.node.removeAttribute('data-gnn-node-id');
        }
    }
    delete globalThis.__gnnSnapshotOldMarkers;
}"""


def _attribute_map(node: dict[str, Any]) -> dict[str, str]:
    attributes = node.get("attributes") or []
    return {
        str(attributes[index]): str(attributes[index + 1])
        for index in range(0, len(attributes) - 1, 2)
    }


def build_backend_marker_map(root: dict[str, Any]) -> dict[str, str]:
    """Map CDP backend DOM node ids to the temporary snapshot marker ids."""
    mapping: dict[str, str] = {}

    def visit(node: dict[str, Any]) -> None:
        backend_id = node.get("backendNodeId")
        marker = _attribute_map(node).get("data-gnn-node-id")
        if backend_id is not None and marker is not None:
            mapping[str(backend_id)] = marker
        for key in ("children", "shadowRoots", "pseudoElements"):
            for child in node.get(key) or []:
                visit(child)
        content_document = node.get("contentDocument")
        if isinstance(content_document, dict):
            visit(content_document)

    visit(root)
    return mapping


async def _session_id(page):
    if hasattr(page, "_ensure_session"):
        return await page._ensure_session()
    value = page.session_id
    return await value if hasattr(value, "__await__") else value


async def capture_page_screenshot(page, session_id: str, path: Path) -> None:
    """Capture the audited page target, not BrowserSession's focused tab."""
    result = await page._client.send.Page.captureScreenshot(
        {"format": "png", "captureBeyondViewport": True},
        session_id=session_id,
    )
    encoded = result.get("data") if isinstance(result, dict) else None
    if not encoded:
        raise RuntimeError("page-targeted CDP screenshot returned no data")
    path.write_bytes(base64.b64decode(encoded, validate=True))


async def capture_rendered_snapshot(
    page,
    html_path: Path,
    *,
    ax_path: Path | None = None,
    screenshot_path: Path | None = None,
) -> dict[str, Path]:
    """Atomically write a same-session aligned evidence bundle.

    The DOM markers remain live through the CDP DOM/AX captures and screenshot.
    A failure raises and leaves no newly staged final bundle.
    """
    visual_path = html_path.with_suffix(".visual.json")
    ax_path = ax_path or html_path.with_suffix(".ax.json")
    screenshot_path = screenshot_path or html_path.with_suffix(".png")
    paths = {
        "html": html_path,
        "visual": visual_path,
        "ax": ax_path,
        "screenshot": screenshot_path,
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    staged = {
        "html": html_path.with_name(f".{html_path.name}.tmp"),
        "visual": visual_path.with_name(f".{visual_path.name}.tmp"),
        "ax": ax_path.with_name(f".{ax_path.name}.tmp"),
        "screenshot": screenshot_path.with_name(f".{screenshot_path.stem}.tmp{screenshot_path.suffix}"),
    }

    try:
        snapshot = await page.evaluate(CAPTURE_SCRIPT)
        if isinstance(snapshot, str):
            snapshot = json.loads(snapshot)
        session_id = await _session_id(page)
        document = await page._client.send.DOM.getDocument(
            {"depth": -1, "pierce": True}, session_id=session_id
        )
        ax_result = await page._client.send.Accessibility.getFullAXTree(
            {}, session_id=session_id
        )
        backend_to_marker = build_backend_marker_map(document["root"])

        await capture_page_screenshot(page, session_id, staged["screenshot"])

        ax_nodes = ax_result.get("nodes") or []
        mapped_ax_nodes = sum(
            1
            for node in ax_nodes
            if str(node.get("backendDOMNodeId")) in backend_to_marker
        )
        ax_payload = {
            "version": SNAPSHOT_VERSION,
            "captured_at": datetime.now(UTC).isoformat(),
            "captured_url": snapshot["visual"].get("captured_url"),
            "viewport": snapshot["visual"].get("viewport"),
            "nodes": ax_nodes,
            "backend_dom_to_snapshot_node": backend_to_marker,
            "mapping_stats": {
                "dom_nodes_with_marker": len(backend_to_marker),
                "ax_nodes": len(ax_nodes),
                "ax_nodes_mapped_to_snapshot": mapped_ax_nodes,
            },
        }
        staged["html"].write_text(snapshot["html"], encoding="utf-8")
        staged["visual"].write_text(
            json.dumps(snapshot["visual"], indent=2, ensure_ascii=False), encoding="utf-8"
        )
        staged["ax"].write_text(
            json.dumps(ax_payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        for key, path in paths.items():
            temporary = staged[key]
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError(f"required snapshot artifact was not written: {path}")
        for key, path in paths.items():
            staged[key].replace(path)
        return paths
    finally:
        try:
            await page.evaluate(CLEANUP_SCRIPT)
        finally:
            for temporary in staged.values():
                temporary.unlink(missing_ok=True)
