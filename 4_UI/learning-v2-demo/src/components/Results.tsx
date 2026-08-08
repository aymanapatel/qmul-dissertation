import { Check, ExternalLink, Globe2, Network } from "lucide-react";
import { useMemo, useState } from "react";
import { API_BASE } from "../lib/api";
import { architectureLabel, viewLabel } from "../lib/format";
import type { Job, SuggestionResult } from "../types";
import { ModelMatrix } from "./ModelMatrix";
import { PipelineLive } from "./Pipeline";
import { SuggestionDetail, SuggestionList } from "./Suggestions";
import { TraceInspector } from "./TraceInspector";

function PageEvidence({ result }: { result: SuggestionResult }) {
  const screenshot = result.screenshot_url ? `${API_BASE}${result.screenshot_url}` : null;

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
          <img src={screenshot} alt={`Captured rendering of ${result.final_url}`} />
        ) : (
          <div>No screenshot was captured.</div>
        )}
      </div>
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
