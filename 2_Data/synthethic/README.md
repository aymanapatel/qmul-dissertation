# Synthetic accessibility UI corpus

Six deliberately inaccessible, fully interactive UI applications derived from
`misc/WCAG_All_ExceptTimebased.csv`. Each application concentrates on one defect
family so scanners, accessibility-tree extractors, and repair systems can be
tested against realistic page structure rather than isolated snippets.

## Run

From this directory, serve the files with any static server, for example:

```sh
python3 -m http.server 8080
```

Then open `http://localhost:8080`.

## Applications

| Directory | Product | Deliberate primary defect |
| --- | --- | --- |
| `01-missing-name/problematic_site` | Framebox media library | Icon controls without accessible names |
| `02-broken-labels/problematic_site` | Wellnest appointment booking | Visible form labels are not programmatically associated |
| `03-low-contrast/problematic_site` | Northstar finance dashboard | Low text, control, and focus contrast |
| `04-keyboard-focus/problematic_site` | Sprintly project board | Mouse-only controls, illogical focus, suppressed focus styles |
| `05-poor-semantics/problematic_site` | The Current news reader | Visual structure implemented with generic elements |
| `06-unclear-purpose/problematic_site` | Atlas knowledge hub | Repeated vague links and buttons |

Each issue directory is laid out as `<issue>/problematic_site` (the deliberately
defective application) and `<issue>/solution` (the repaired version).

`manifest.json` is the machine-readable ground truth. Defects are intentional;
do not use these applications as production accessibility examples.

