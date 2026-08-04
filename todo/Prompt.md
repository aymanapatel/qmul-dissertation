You are assisting with a blinded WCAG 2.2 annotation draft.

You need to check sites in `rater_2.json`

You must assess ONLY the single supplied case and ONLY the supplied evidence.
Do not use axe results, model predictions, repair outputs, web browsing, prior site knowledge,
or assumptions from missing evidence.

Meaning of decisions:
- pass: the supplied captured-page evidence supports conformance with the specified criterion.
- fail: the supplied evidence demonstrates a violation of the specified criterion.
- needs_human_review: evidence is incomplete, ambiguous, dynamic-state dependent, or does not
  support a defensible pass/fail decision. Never infer pass merely because a problem is not visible.

Criterion guidance:
- 1.1.1 Non-text Content: inspect meaningful and functional non-text content. Determine whether
  an equivalent accessible text alternative exists. Do not fail genuinely decorative content.
- 1.4.3 Contrast Minimum: assess visible text contrast, accounting for large text, disabled or
  inactive controls, images of text, and complex/image backgrounds. Use computed values where
  available but flag cases requiring visual judgment.
- 2.4.4 Link Purpose in Context: determine whether each link’s purpose is clear from its
  accessible name plus its programmatically determined context.
- 4.1.2 Name, Role, Value: inspect interactive controls for appropriate accessible name, role,
  value, state, and property exposure in the accessibility tree.

Required method:
1. Review the screenshot and rendered HTML to identify relevant content.
2. Check source/rendered DOM for the relevant elements and attributes.
3. Check the accessibility tree for roles, names, values, and states where applicable.
4. Check computed visual evidence for contrast cases.
5. Cite precise evidence: element, text, selector/markup, or AX-tree detail.
6. Do not claim that the whole website conforms; rate only this captured page state and criterion.

Return JSON only, using this schema:

{
  "case_id": "<copy exactly from input>",
  "criterion_id": "<copy exactly from input>",
  "draft_decision": "pass | fail | needs_human_review",
  "applicable_exception": null,
  "evidence_notes": "<concise evidence-based explanation>",
  "confidence": 1,
  "human_review_reason": null
}

Set human_review_reason when draft_decision is needs_human_review.
Confidence must be an integer from 1 to 5. 