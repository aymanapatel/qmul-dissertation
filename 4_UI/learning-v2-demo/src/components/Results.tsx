import { Check, ExternalLink, Globe2, Network } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { API_BASE, api } from "../lib/api";
import { architectureLabel, viewLabel } from "../lib/format";
import type { Job, SuggestionResult } from "../types";
import { ModelMatrix } from "./ModelMatrix";
import { PipelineLive } from "./Pipeline";
import { SuggestionDetail, SuggestionList } from "./Suggestions";
import { TraceInspector } from "./TraceInspector";

function PageEvidence({ result }: { result: SuggestionResult }) {
  const screenshot = result.screenshot_url ? `${API_BASE}${result.screenshot_url}` : null;
  const [visualEvidence, setVisualEvidence] = useState(result.visual_evidence);
  useEffect(() => {
    if (visualEvidence || !result.run_id) return;
    api<SuggestionResult>(`/v1/jobs/${result.run_id}/result`)
      .then((updated) => setVisualEvidence(updated.visual_evidence))
      .catch(() => undefined);
  }, [result.run_id, visualEvidence]);
  const contrastFailures = visualEvidence?.contrast_failures || [];
  const canvasWidth = visualEvidence?.canvas.width || 1;
  const canvasHeight = visualEvidence?.canvas.height || 1;

  return (
    <section className="page-evidence">
      <div className="panel-heading">
        <div>
          <span className="section-label">INPUT PAGE</span>
          <h2>Captured webpage</h2>
        </div>
        <a href={result.final_url} target="_blank" rel="noreferrer" aria-label="Open audited page">
          <ExternalLink size={17} />
        </a>
      </div>
      <div className="url-chip">
        <Globe2 size={14} />
        <span>{result.final_url}</span>
      </div>
      <div className="screenshot-frame">
        {screenshot ? (
          <div className="screenshot-stage">
            <img src={screenshot} alt={`Captured rendering of ${result.final_url}`} />
            {contrastFailures.map((element) => (
              <span
                className="contrast-highlight"
                key={`${element.snapshot_node_id}-${element.selector}`}
                style={{
                  left: `${(element.bounds.x / canvasWidth) * 100}%`,
                  top: `${(element.bounds.y / canvasHeight) * 100}%`,
                  width: `${(element.bounds.width / canvasWidth) * 100}%`,
                  height: `${(element.bounds.height / canvasHeight) * 100}%`,
                }}
                title={`${element.selector}: ${element.visual.contrast_ratio.toFixed(2)}:1; required ${element.visual.required_contrast_ratio}:1`}
              />
            ))}
          </div>
        ) : (
          <div>No screenshot was captured.</div>
        )}
      </div>
      {contrastFailures.length > 0 && (
        <div className="contrast-summary" role="status">
          <strong>{contrastFailures.length} computed contrast failures highlighted</strong>
          {contrastFailures.map((element) => (
            <span key={`summary-${element.snapshot_node_id}-${element.selector}`}>
              <code>{element.selector}</code>
              {element.visual.contrast_ratio.toFixed(2)}:1 / required{" "}
              {element.visual.required_contrast_ratio}:1
            </span>
          ))}
        </div>
      )}
      <div className="scan-summary">
        <div>
          <strong>{result.specialist.finding_count}</strong>
          <span>routed model findings</span>
        </div>
        <div>
          <strong>{result.specialist.model_runs.length}</strong>
          <span>specialist runs</span>
        </div>
        <div>
          <strong>{result.violation_count}</strong>
          <span>axe rules observed</span>
        </div>
        <div>
          <strong>{result.affected_node_count}</strong>
          <span>axe nodes observed</span>
        </div>
      </div>
    </section>
  );
}

export function Results({ result, job }: { result: SuggestionResult; job: Job }) {
  const [selectedId, setSelectedId] = useState(result.suggestions[0]?.finding_id || "");
  const selected = useMemo(
    () =>
      result.suggestions.find((item) => item.finding_id === selectedId) || result.suggestions[0],
    [result, selectedId],
  );

  return (
    <div className="result-stack">
      <PipelineLive job={job} />
      <ModelMatrix runs={result.specialist.model_runs} />
      <div className="results-grid">
        <PageEvidence result={result} />
        <section className="suggestions-panel">
          <div className="panel-heading">
            <div>
              <span className="section-label">ARCHITECTURE-SPECIFIC REMEDIATION</span>
              <h2>{result.suggestion_count} reviewable suggestions</h2>
            </div>
            <span className="live-model">
              <span />
              {result.suggestion_count ? "Live LLM output" : "No LLM call required"}
            </span>
          </div>
          {result.suggestions.length ? (
            <>
              <SuggestionList
                suggestions={result.suggestions}
                selected={selectedId}
                onSelect={setSelectedId}
              />
              {selected && <SuggestionDetail suggestion={selected} />}
            </>
          ) : (
            <div className="no-findings">
              <Check size={25} />
              <h3>No frozen-threshold failures</h3>
              <p>
                All six trained specialists completed, but none produced a routed failure. The
                system did not manufacture a suggestion or make an unnecessary LLM call.
              </p>
              <div className="completed-runs">
                {result.specialist.model_runs.map((run) => (
                  <span key={`${run.architecture}-${run.view}`}>
                    <Network size={14} />
                    <strong>{architectureLabel(run.architecture)}</strong>
                    {viewLabel(run.view)} · {run.node_count} nodes
                  </span>
                ))}
              </div>
            </div>
          )}
        </section>
      </div>
      <TraceInspector result={result} job={job} suggestion={selected} />
    </div>
  );
}
