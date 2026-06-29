"""
wcag_rules.py

Constants for WCAG axe-core rules used in multi-label classification.
"""

# Mapping from axe rule ID to index
RULE_INDEX = {
    "area-alt": 0,
    "aria-allowed-attr": 1,
    "aria-command-name": 2,
    "aria-hidden-focus": 3,
    "aria-input-field-name": 4,
    "aria-meter-name": 5,
    "aria-progressbar-name": 6,
    "aria-prohibited-attr": 7,
    "aria-required-attr": 8,
    "aria-required-children": 9,
    "aria-required-parent": 10,
    "aria-roles": 11,
    "aria-toggle-field-name": 12,
    "aria-tooltip-name": 13,
    "aria-valid-attr": 14,
    "aria-valid-attr-value": 15,
    "autocomplete-valid": 16,
    "avoid-inline-spacing": 17,
    "blink": 18,
    "button-name": 19,
    "color-contrast": 20,
    "definition-list": 21,
    "dlitem": 22,
    "document-title": 23,
    "frame-title": 24,
    "html-has-lang": 25,
    "html-lang-valid": 26,
    "image-alt": 27,
    "input-button-name": 28,
    "input-image-alt": 29,
    "label": 30,
    "link-in-text-block": 31,
    "link-name": 32,
    "list": 33,
    "listitem": 34,
    "marquee": 35,
    "meta-refresh": 36,
    "meta-viewport": 37,
    "nested-interactive": 38,
    "object-alt": 39,
    "role-img-alt": 40,
    "scrollable-region-focusable": 41,
    "select-name": 42,
    "summary-name": 43,
    "svg-img-alt": 44,
    "valid-lang": 45,
}

# Reverse mapping
INDEX_TO_RULE = {v: k for k, v in RULE_INDEX.items()}

NUM_RULES = 46

# Rule categories for reporting
RULE_CATEGORIES = {
    "image-alt": "Images",
    "color-contrast": "Visual",
    "link-name": "Navigation",
    "button-name": "Forms",
    "label": "Forms",
    "html-has-lang": "Language",
    "frame-title": "Frames",
    "meta-viewport": "Mobile",
    "list": "Structure",
    "listitem": "Structure",
    "heading-order": "Structure",
    "aria-required-children": "ARIA",
    "aria-required-parent": "ARIA",
    "aria-allowed-attr": "ARIA",
    "nested-interactive": "Interaction",
    "scrollable-region-focusable": "Interaction",
    "link-in-text-block": "Links",
}
