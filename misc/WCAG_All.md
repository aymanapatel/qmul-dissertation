Below is an **exhaustive WCAG 2.2 issue taxonomy** you can use as labels for your accessibility/GNN/RAG project. WCAG 2.2 is organised around four principles: **Perceivable, Operable, Understandable, Robust**. W3C describes WCAG success criteria as testable statements, while techniques and common failures provide implementation guidance. 

For your dissertation, these can become **violation labels**, **graph node labels**, or **retrieval categories** for your AccessibilityGraph-RAG system, which aims to reason over accessibility criteria, ARIA semantics, and structural repair patterns  [oai_citation:0‡Proposal_Diana BenavidesPrado_GNN_RAG 1.pdf](sediment://file_000000009a847246b28384498e1ceac6).

---

# Exhaustive WCAG issue list

| WCAG area | Success criterion | Common accessibility issues / dataset labels |
|---|---|---|
| **1.1 Text Alternatives** | **1.1.1 Non-text Content** | Missing `alt` text; decorative image not hidden; image button without accessible name; CAPTCHA without alternative; icon-only control without label; chart/image has insufficient text alternative. |
| **1.2 Time-based Media** | **1.2.1 Audio-only and Video-only** | Audio has no transcript; video-only content has no text/audio alternative. |
|  | **1.2.2 Captions Prerecorded** | Prerecorded video has no captions; captions miss important speech or sounds. |
|  | **1.2.3 Audio Description or Media Alternative** | Video has important visual content not described; no audio description or full media transcript. |
|  | **1.2.4 Captions Live** | Live video/audio stream has no live captions. |
|  | **1.2.5 Audio Description Prerecorded** | Prerecorded video has no audio description where visuals are needed to understand content. |
|  | **1.2.6 Sign Language** | No sign-language interpretation for prerecorded audio. |
|  | **1.2.7 Extended Audio Description** | Audio description cannot fit naturally and no extended audio description is provided. |
|  | **1.2.8 Media Alternative** | No full text/media alternative for prerecorded media. |
|  | **1.2.9 Audio-only Live** | Live audio-only content has no equivalent text alternative. |
| **1.3 Adaptable** | **1.3.1 Info and Relationships** | Headings not marked as headings; visual lists not coded as lists; table headers missing; form labels not programmatically associated; field groups missing `fieldset`/`legend`; ARIA relationships missing; layout conveys meaning but DOM does not. |
|  | **1.3.2 Meaningful Sequence** | DOM reading order differs from visual/logical order; CSS positioning creates confusing screen-reader order; multi-column content read incorrectly. |
|  | **1.3.3 Sensory Characteristics** | Instructions rely only on shape, size, colour, location, or sound, such as “click the green button” or “use the menu on the right”. |
|  | **1.3.4 Orientation** | Content locked to portrait/landscape without essential reason. |
|  | **1.3.5 Identify Input Purpose** | Form fields for name, email, address, phone, etc. lack correct `autocomplete` attributes. |
|  | **1.3.6 Identify Purpose** | UI components, icons, regions, or controls do not expose machine-readable purpose where required. |
| **1.4 Distinguishable** | **1.4.1 Use of Color** | Colour alone communicates required fields, errors, selected state, chart meaning, or status. |
|  | **1.4.2 Audio Control** | Audio auto-plays for more than 3 seconds without pause/stop/volume control. |
|  | **1.4.3 Contrast Minimum** | Text contrast below 4.5:1; large text below 3:1; disabled-looking active controls; placeholder text with poor contrast. |
|  | **1.4.4 Resize Text** | Text cannot resize to 200%; layout breaks; text overlaps/clips when zoomed. |
|  | **1.4.5 Images of Text** | Text is embedded inside images instead of real text, except allowed cases. |
|  | **1.4.6 Contrast Enhanced** | AAA-level text contrast failure: normal text below 7:1 or large text below 4.5:1. |
|  | **1.4.7 Low or No Background Audio** | Background audio makes speech hard to hear; no way to reduce background audio. |
|  | **1.4.8 Visual Presentation** | Poor paragraph spacing, line length, line spacing, foreground/background selection, or full justification issues at AAA level. |
|  | **1.4.9 Images of Text No Exception** | Images of text used even where no essential exception applies. |
|  | **1.4.10 Reflow** | Horizontal scrolling required at 320 CSS px width; responsive layout breaks; fixed-width containers overflow. |
|  | **1.4.11 Non-text Contrast** | Buttons, inputs, focus indicators, icons, charts, boundaries, or graphical controls have contrast below 3:1. |
|  | **1.4.12 Text Spacing** | Content breaks when users increase line height, paragraph spacing, letter spacing, or word spacing. |
|  | **1.4.13 Content on Hover or Focus** | Tooltip/popover cannot be dismissed, hovered, or kept visible; content disappears too quickly. |
| **2.1 Keyboard Accessible** | **2.1.1 Keyboard** | Functionality not usable by keyboard; click-only controls; custom widgets missing keyboard handlers. |
|  | **2.1.2 No Keyboard Trap** | Focus gets trapped inside widget/modal/iframe with no keyboard escape. |
|  | **2.1.3 Keyboard No Exception** | AAA: all functionality must be keyboard accessible with no exception. |
|  | **2.1.4 Character Key Shortcuts** | Single-letter shortcuts cannot be turned off, remapped, or limited to focus state. |
| **2.2 Enough Time** | **2.2.1 Timing Adjustable** | Session timeout cannot be extended; timed quiz/form has no adjustment; warning missing. |
|  | **2.2.2 Pause, Stop, Hide** | Moving carousel, animation, ticker, auto-updating content cannot be paused/stopped/hidden. |
|  | **2.2.3 No Timing** | AAA: timing is unnecessarily required for task completion. |
|  | **2.2.4 Interruptions** | Interruptions cannot be postponed or suppressed. |
|  | **2.2.5 Re-authenticating** | User loses data after session expiry/re-authentication. |
|  | **2.2.6 Timeouts** | Users are not warned about inactivity timeout duration or data loss. |
| **2.3 Seizures and Physical Reactions** | **2.3.1 Three Flashes or Below Threshold** | Content flashes more than three times per second or exceeds flash thresholds. |
|  | **2.3.2 Three Flashes** | AAA: any flashing above threshold. |
|  | **2.3.3 Animation from Interactions** | Motion animation triggered by interaction cannot be disabled. |
| **2.4 Navigable** | **2.4.1 Bypass Blocks** | No skip link; no landmark navigation; repeated nav/header cannot be bypassed. |
|  | **2.4.2 Page Titled** | Missing, empty, duplicate, or vague page title. |
|  | **2.4.3 Focus Order** | Keyboard focus order is illogical; focus jumps unexpectedly; modal focus order broken. |
|  | **2.4.4 Link Purpose In Context** | Link text like “click here”, “read more”, “learn more” lacks context. |
|  | **2.4.5 Multiple Ways** | No search, sitemap, navigation, or alternative way to find pages. |
|  | **2.4.6 Headings and Labels** | Headings/labels are vague, misleading, missing, or not descriptive. |
|  | **2.4.7 Focus Visible** | Keyboard focus indicator missing or hidden. |
|  | **2.4.8 Location** | AAA: user cannot determine current location within a site/app. |
|  | **2.4.9 Link Purpose Link Only** | AAA: link purpose not clear from link text alone. |
|  | **2.4.10 Section Headings** | AAA: content lacks section headings where needed. |
|  | **2.4.11 Focus Not Obscured Minimum** | Focused item is partly hidden by sticky header, cookie banner, modal, or overlay. |
|  | **2.4.12 Focus Not Obscured Enhanced** | AAA: focused item is fully visible and not obscured at all. |
|  | **2.4.13 Focus Appearance** | Focus indicator is too small, too low contrast, or visually unclear. |
| **2.5 Input Modalities** | **2.5.1 Pointer Gestures** | Complex gestures required without single-pointer alternative; pinch/swipe-only interaction. |
|  | **2.5.2 Pointer Cancellation** | Action fires on pointer down with no cancellation/undo; accidental activation risk. |
|  | **2.5.3 Label in Name** | Visible label does not match accessible name; speech users cannot activate by visible text. |
|  | **2.5.4 Motion Actuation** | Device motion required without alternative; no way to disable motion activation. |
|  | **2.5.5 Target Size Enhanced** | AAA: clickable/touch target too small. |
|  | **2.5.6 Concurrent Input Mechanisms** | Site unnecessarily blocks mouse, keyboard, touch, stylus, or assistive input combinations. |
|  | **2.5.7 Dragging Movements** | Drag-and-drop required without simple alternative. |
|  | **2.5.8 Target Size Minimum** | Click/touch target too small or too close to neighbouring targets. |
| **3.1 Readable** | **3.1.1 Language of Page** | Missing or incorrect `lang` attribute on page. |
|  | **3.1.2 Language of Parts** | Foreign-language phrases/sections not marked with correct `lang`. |
|  | **3.1.3 Unusual Words** | AAA: jargon, idioms, or technical terms not explained. |
|  | **3.1.4 Abbreviations** | AAA: abbreviations not expanded or explained. |
|  | **3.1.5 Reading Level** | AAA: text too complex without simpler alternative. |
|  | **3.1.6 Pronunciation** | AAA: pronunciation needed for understanding but not provided. |
| **3.2 Predictable** | **3.2.1 On Focus** | Focusing an element unexpectedly changes page, submits form, opens modal, or moves user. |
|  | **3.2.2 On Input** | Changing input unexpectedly triggers navigation, submission, or context change. |
|  | **3.2.3 Consistent Navigation** | Navigation order/location changes across pages without reason. |
|  | **3.2.4 Consistent Identification** | Same function has inconsistent labels/icons across pages. |
|  | **3.2.5 Change on Request** | AAA: context changes happen without explicit user request. |
|  | **3.2.6 Consistent Help** | Help/contact mechanisms appear inconsistently across pages. |
| **3.3 Input Assistance** | **3.3.1 Error Identification** | Form errors not identified; invalid field not announced; error only shown by colour. |
|  | **3.3.2 Labels or Instructions** | Inputs lack labels/instructions; required format not explained. |
|  | **3.3.3 Error Suggestion** | No suggestion for fixing input error when suggestion is possible. |
|  | **3.3.4 Error Prevention Legal, Financial, Data** | No review/confirm/reverse mechanism for important submissions. |
|  | **3.3.5 Help** | AAA: contextual help missing for complex forms/tasks. |
|  | **3.3.6 Error Prevention All** | AAA: no error prevention for all user submissions. |
|  | **3.3.7 Redundant Entry** | Users must re-enter information already provided in same process. |
|  | **3.3.8 Accessible Authentication Minimum** | Login requires cognitive test, memory puzzle, transcription, or object recognition without accessible alternative. |
|  | **3.3.9 Accessible Authentication Enhanced** | AAA: stronger version; authentication depends on cognitive function without sufficient accessible mechanism. |
| **4.1 Compatible** | **4.1.1 Parsing** | Obsolete/removed in WCAG 2.2, but older WCAG 2.0/2.1 audits may still check duplicate IDs, malformed markup, broken nesting. W3C notes 4.1.1 was removed in WCAG 2.2.  |
|  | **4.1.2 Name, Role, Value** | Custom controls lack accessible name, role, state, or value; ARIA incorrectly used; toggle state not announced; modal lacks role/name; combobox/tree/menu not exposed correctly. |
|  | **4.1.3 Status Messages** | Toasts, validation messages, loading states, cart updates, and search-result counts are not announced to assistive technology. |

---

# Better dataset label groups for your project

For your GNN/RAG dissertation, I would not train 80+ tiny labels first. Start with **high-value issue families**:

| Label family | WCAG criteria covered | Why good for graph modelling |
|---|---|---|
| **Missing accessible name** | 1.1.1, 2.4.6, 3.3.2, 4.1.2 | Needs node + label + ARIA relationship reasoning. |
| **Broken form labelling** | 1.3.1, 3.3.1, 3.3.2, 3.3.3 | Excellent for DOM/accessibility-tree graphs. |
| **Low contrast** | 1.4.3, 1.4.11 | Needs visual node + CSS/background relationship. |
| **Keyboard/focus issue** | 2.1.1, 2.1.2, 2.4.3, 2.4.7, 2.4.11, 2.4.13 | Needs sequential/focus-order edges. |
| **Poor semantic structure** | 1.3.1, 2.4.1, 2.4.2, 2.4.6 | Needs heading, landmark, list, table, and parent-child relationships. |
| **Unclear link/button purpose** | 2.4.4, 2.4.9, 2.5.3, 4.1.2 | Needs text, accessible name, visual label, and surrounding context. |
| **Dynamic content not announced** | 4.1.3, 3.2.1, 3.2.2 | Needs event/state-change graph and ARIA live-region reasoning. |
| **Touch/pointer target issue** | 2.5.5, 2.5.8, 2.5.1, 2.5.7 | Needs bounding boxes, spatial adjacency, and interaction modelling. |
| **Media alternative issue** | 1.2.x | Mostly content/media metadata; less graph-heavy unless media is embedded in complex UI. |
| **Authentication/error-prevention issue** | 3.3.4, 3.3.7, 3.3.8, 3.3.9 | Good for process-level graph across multiple pages/states. |

The most dissertation-friendly subset is:

> **Accessible name + form labels + contrast + keyboard/focus + semantic structure + status messages**

These are common, measurable, graph-relevant, and easier to evaluate with axe-core, Playwright, accessibility-tree snapshots, and manual validation.
