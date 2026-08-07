import React, { FormEvent, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  AlertTriangle,
  ArrowRight,
  Bot,
  Check,
  ChevronRight,
  Code2,
  ExternalLink,
  Eye,
  FileSearch,
  Globe2,
  LoaderCircle,
  Network,
  ScanSearch,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import type { Job, RepairOperation, Suggestion, SuggestionResult } from "./types";
import "./styles.css";

const API_BASE = (import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000").replace(/\/$/, "");

const sleep = (milliseconds: number) => new Promise((resolve) => window.setTimeout(resolve, milliseconds));

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail || body);
    throw new Error(detail || `Request failed with status ${response.status}`);
  }
  return body as T;
}

function decisionLabel(suggestion: Suggestion): string {
  if (suggestion.generation_status === "failed") return "Generation failed";
  if (suggestion.decision === "requires_human_review") return "Human review";
  if (suggestion.decision === "leave_unchanged") return "Leave unchanged";
  return "Suggested change";
}

function operationText(operation: RepairOperation): string {
  if (operation.operation === "set_attribute") {
    return `Set ${operation.attribute_name}="${operation.new_value}"`;
  }
  if (operation.operation === "remove_attribute") return `Remove ${operation.attribute_name}`;
  if (operation.operation === "set_style_property") return `Set ${operation.css_property}: ${operation.new_value}`;
  if (operation.operation === "insert_label_before") return `Insert label “${operation.new_value}”`;
  if (operation.operation === "replace_text") return `Replace text with “${operation.new_value}”`;
  return operation.operation.replaceAll("_", " ");
}

function Workflow({ active }: { active: number }) {
  const steps = [
    { label: "Capture page", icon: Globe2 },
    { label: "Build graph views", icon: Network },
    { label: "Run GraphSAGE", icon: ScanSearch },
    { label: "Route findings", icon: ShieldCheck },
    { label: "Call LLM", icon: Bot },
  ];
  return <div className="workflow" aria-label="Audit progress">
    {steps.map((step, index) => {
      const Icon = step.icon;
      return <React.Fragment key={step.label}>
        <div className={`workflow-step ${active >= index ? "active" : ""}`}>
          <span>{active > index ? <Check size={16} /> : <Icon size={17} />}</span>
          <small>{step.label}</small>
        </div>
        {index < steps.length - 1 && <ChevronRight size={16} className={active > index ? "active" : ""} />}
      </React.Fragment>;
    })}
  </div>;
}

function EmptyState() {
  return <div className="empty-state">
    <div className="empty-graphic">
      <span><Globe2 size={26} /></span><ArrowRight size={22} /><span><ScanSearch size={26} /></span><ArrowRight size={22} /><span><Sparkles size={26} /></span>
    </div>
    <h2>One page in. Reviewable suggestions out.</h2>
    <p>Enter a public webpage above. The server captures that exact page, runs the frozen accessibility-tree and rendered-visual specialists, and asks the configured LLM for bounded remediation suggestions.</p>
    <div className="boundaries"><span><Check size={15} />Trained GNN inference</span><span><Check size={15} />Actual LLM call</span><span><ShieldCheck size={15} />No automatic edits</span></div>
  </div>;
}

function PageEvidence({ result }: { result: SuggestionResult }) {
  const screenshot = result.screenshot_url ? `${API_BASE}${result.screenshot_url}` : null;
  return <section className="page-evidence">
    <div className="panel-heading">
      <div><span className="section-label">INPUT PAGE</span><h2>Captured webpage</h2></div>
      <a href={result.final_url} target="_blank" rel="noreferrer" aria-label="Open audited page in a new tab"><ExternalLink size={17} /></a>
    </div>
    <div className="url-chip"><Globe2 size={14} /><span>{result.final_url}</span></div>
    <div className="screenshot-frame">
      {screenshot ? <img src={screenshot} alt={`Captured rendering of ${result.final_url}`} /> : <div>No screenshot was captured.</div>}
    </div>
    <div className="scan-summary">
      <div><strong>{result.specialist.finding_count}</strong><span>routed GNN findings</span></div>
      <div><strong>{result.violation_count}</strong><span>axe rules observed</span></div>
      {Object.entries(result.violations_by_impact).slice(0, 2).map(([impact, count]) => <div key={impact}><strong>{count}</strong><span>{impact}</span></div>)}
    </div>
  </section>;
}

function SuggestionList({ suggestions, selected, onSelect }: { suggestions: Suggestion[]; selected: string; onSelect: (id: string) => void }) {
  return <div className="suggestion-list" aria-label="Generated remediation suggestions">
    {suggestions.map((suggestion, index) => <button key={suggestion.finding_id} onClick={() => onSelect(suggestion.finding_id)} className={selected === suggestion.finding_id ? "selected" : ""}>
      <span className="suggestion-number">{String(index + 1).padStart(2, "0")}</span>
      <span className="suggestion-title"><small>{suggestion.rule_id}</small><strong>{suggestion.help || suggestion.rule_id.replaceAll("-", " ")}</strong><code>{suggestion.target.join(", ") || "page"}</code></span>
      <span className={`decision decision-${suggestion.decision || "failed"}`}>{decisionLabel(suggestion)}</span>
    </button>)}
  </div>;
}

function SuggestionDetail({ suggestion }: { suggestion: Suggestion }) {
  if (suggestion.generation_status === "failed") {
    return <div className="suggestion-detail error-detail"><AlertTriangle /><h3>The model did not return a suggestion</h3><p>{suggestion.error}</p></div>;
  }
  return <article className="suggestion-detail">
    <div className="detail-heading">
      <div><span className="section-label">LLM SUGGESTION</span><h2>{suggestion.help || suggestion.rule_id}</h2></div>
      <div className="confidence"><small>Confidence</small><strong>{Math.round((suggestion.confidence || 0) * 100)}%</strong></div>
    </div>
    <p className="rationale">{suggestion.rationale}</p>
    {suggestion.model_evidence && <div className="gnn-evidence">
      <div><Network size={18} /><span><small>TRAINED SPECIALIST</small><strong>{suggestion.model_evidence.graph_view} · {suggestion.model_evidence.architecture}</strong></span></div>
      <div className="probability"><span style={{ width: `${Math.max(2, suggestion.model_evidence.probability * 100)}%` }} /><i style={{ left: `${Math.min(99, suggestion.model_evidence.threshold * 100)}%` }} /></div>
      <div className="probability-labels"><span>Probability {(suggestion.model_evidence.probability * 100).toFixed(1)}%</span><span>Frozen threshold {(suggestion.model_evidence.threshold * 100).toFixed(1)}%</span><span>Routed: {suggestion.model_evidence.routing_status.replaceAll("_", " ")}</span></div>
    </div>}
    {(suggestion.operations || []).length > 0 && <div className="operations">
      <h3><Code2 size={17} />Proposed operations</h3>
      {suggestion.operations!.map((operation, index) => <div className="operation" key={`${operation.operation}-${index}`}>
        <span>{index + 1}</span><div><strong>{operationText(operation)}</strong><code>{operation.selector}</code></div>
      </div>)}
    </div>}
    {suggestion.requires_human_review && <div className="review-box"><Eye size={18} /><div><strong>Human judgement required</strong>{suggestion.human_review_reasons?.map((reason) => <p key={reason}>{reason}</p>)}</div></div>}
    <div className="expected"><strong>Expected resolution</strong><p>{suggestion.expected_resolution}</p></div>
    <details><summary>Validation checklist</summary><ol>{suggestion.validation_steps?.map((step) => <li key={step}>{step}</li>)}</ol></details>
    <div className="model-proof"><Bot size={16} /><span>Generated by <strong>{suggestion.model}</strong></span>{suggestion.response_id && <code>{suggestion.response_id}</code>}</div>
  </article>;
}

function Results({ result }: { result: SuggestionResult }) {
  const [selectedId, setSelectedId] = useState(result.suggestions[0]?.finding_id || "");
  const selected = useMemo(() => result.suggestions.find((item) => item.finding_id === selectedId) || result.suggestions[0], [result, selectedId]);
  return <div className="results-grid">
    <PageEvidence result={result} />
    <section className="suggestions-panel">
      <div className="panel-heading"><div><span className="section-label">SUGGESTED REMEDIATION</span><h2>{result.suggestion_count} reviewable suggestions</h2></div><span className="live-model"><span />{result.suggestion_count ? "Live LLM output" : "No LLM call required"}</span></div>
      {result.suggestions.length ? <><SuggestionList suggestions={result.suggestions} selected={selectedId} onSelect={setSelectedId} />{selected && <SuggestionDetail suggestion={selected} />}</> : <div className="no-findings"><Check size={25} /><h3>No frozen-threshold failures</h3><p>Both trained graph specialists completed, but neither produced a routed failure for this capture. The system did not manufacture a suggestion or make an unnecessary LLM call.</p><div className="completed-runs">{result.specialist.model_runs.map((run) => <span key={run.view}><Network size={14} /><strong>{run.view}</strong>{run.node_count} nodes · axe-free</span>)}</div></div>}
    </section>
  </div>;
}

function App() {
  const [url, setUrl] = useState("https://www.w3.org/");
  const [maximum, setMaximum] = useState(5);
  const [status, setStatus] = useState<"idle" | "submitting" | "scanning" | "generating">("idle");
  const [result, setResult] = useState<SuggestionResult | null>(null);
  const [error, setError] = useState("");

  const activeStage = status === "idle" ? (result ? 4 : -1) : status === "submitting" ? 0 : status === "scanning" ? 1 : 4;

  async function submit(event: FormEvent) {
    event.preventDefault(); setError(""); setResult(null); setStatus("submitting");
    try {
      const job = await api<Job>("/v1/suggestion-audits", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ url, max_suggestions: maximum, timeout_seconds: 45 }) });
      setStatus("scanning");
      let current = job;
      for (let attempt = 0; attempt < 180; attempt += 1) {
        await sleep(1000);
        current = await api<Job>(`/v1/jobs/${job.job_id}`);
        if (attempt >= 8) setStatus("generating");
        if (current.status === "failed") throw new Error(current.error || "Audit job failed");
        if (current.status === "completed") break;
      }
      if (current.status !== "completed") throw new Error("The audit did not finish within three minutes.");
      setResult(await api<SuggestionResult>(`/v1/jobs/${job.job_id}/result`));
      setStatus("idle");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The audit failed"); setStatus("idle");
    }
  }

  const busy = status !== "idle";
  return <div className="app-shell">
    <header><a href="#top" className="brand"><span><Network size={19} /></span><div><strong>Accessible</strong><small>LLM remediation demo</small></div></a><div className="api-state"><span />FastAPI · {API_BASE}</div></header>
    <main id="top">
      <section className="hero">
        <div className="hero-copy"><span className="eyebrow">TRAINED ACCESSIBILITY SUGGESTION ENGINE</span><h1>Audit one webpage.<br /><em>Suggest what to fix.</em></h1><p>The backend builds live accessibility-tree and rendered-visual graphs, runs the frozen GraphSAGE specialists and routing policy, then makes structured LLM calls for those model findings. Suggestions are displayed for review and never applied automatically.</p></div>
        <form className="url-form" onSubmit={submit}>
          <label htmlFor="page-url">Public webpage URL</label>
          <div className="url-input"><Globe2 size={20} /><input id="page-url" type="url" required value={url} onChange={(event) => setUrl(event.target.value)} placeholder="https://example.com" /><select aria-label="Maximum suggestions" value={maximum} onChange={(event) => setMaximum(Number(event.target.value))}><option value={3}>3 suggestions</option><option value={5}>5 suggestions</option><option value={8}>8 suggestions</option></select><button type="submit" disabled={busy}>{busy ? <LoaderCircle className="spin" size={18} /> : <Sparkles size={18} />}{busy ? "Working…" : "Generate suggestions"}</button></div>
          <small>Public HTTP(S) pages only. Private and local network targets are blocked.</small>
        </form>
      </section>
      <Workflow active={activeStage} />
      {error && <div className="error-banner" role="alert"><AlertTriangle size={19} /><div><strong>Audit failed</strong><p>{error}</p></div></div>}
      <section className="workspace">{result ? <Results key={result.run_id} result={result} /> : <EmptyState />}</section>
    </main>
    <footer><span>Suggestion-only dissertation prototype</span><span><ShieldCheck size={14} />Source pages are never modified</span></footer>
  </div>;
}

createRoot(document.getElementById("root")!).render(<React.StrictMode><App /></React.StrictMode>);
