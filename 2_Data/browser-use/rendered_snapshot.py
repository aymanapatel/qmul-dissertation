"""Capture an HTML snapshot together with live computed visual features."""

import json
from pathlib import Path


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

    for (const oldMarker of oldMarkers) {
        if (oldMarker.hadMarker) {
            oldMarker.node.setAttribute('data-gnn-node-id', oldMarker.marker);
        } else {
            oldMarker.node.removeAttribute('data-gnn-node-id');
        }
    }

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


async def capture_rendered_snapshot(page, html_path: Path) -> Path:
    """Write marked-up HTML and an adjacent ``*.visual.json`` sidecar."""
    snapshot = await page.evaluate(CAPTURE_SCRIPT)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(snapshot["html"], encoding="utf-8")
    visual_path = html_path.with_suffix(".visual.json")
    visual_path.write_text(
        json.dumps(snapshot["visual"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return visual_path
