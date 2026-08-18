import React from "react";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  Bot,
  Check,
  ChevronRight,
  Clock3,
  Globe2,
  Layers3,
  LoaderCircle,
  Network,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import type { Job, ProgressEvent } from "../types";

type StageState = "waiting" | "running" | "completed" | "failed";

function stageState(
  events: ProgressEvent[],
  predicate: (event: ProgressEvent) => boolean,
  expected = 1,
): StageState {
  const matches = events.filter(predicate);
  if (matches.some((event) => event.status === "failed")) return "failed";
  if (matches.some((event) => event.status === "running")) return "running";
  if (matches.filter((event) => event.status === "completed").length >= expected) {
    return "completed";
  }
  return "waiting";
}

export function Workflow({ job }: { job: Job | null }) {
  const events = job?.progress?.events || [];
  const stages = [
    {
      label: "Capture page",
      icon: Globe2,
      state: stageState(events, (event) => event.event_id === "capture_page"),
    },
    {
      label: "Build graph views",
      icon: Network,
      state: stageState(events, (event) => event.event_id.startsWith("build_"), 2),
    },
    {
      label: "Run 6 specialists",
      icon: Layers3,
      state: stageState(
        events,
        (event) =>
          event.event_id.startsWith("run_mlp") ||
          event.event_id.startsWith("run_graphsage") ||
          event.event_id.startsWith("run_gat"),
        6,
      ),
    },
    {
      label: "Route findings",
      icon: ShieldCheck,
      state: stageState(events, (event) => event.event_id === "route_findings"),
    },
    {
      label: "Call LLM",
      icon: Bot,
      state: stageState(events, (event) => event.event_id.startsWith("call_llm")),
    },
  ];

  if (job?.status === "completed" && stages[4].state === "waiting") {
    stages[4].state = "completed";
  }

  return (
    <div className="workflow" aria-label="Audit progress">
      {stages.map((step, index) => {
        const Icon = step.icon;
        return (
          <React.Fragment key={step.label}>
            <div className={`workflow-step state-${step.state}`}>
              <span>
                {step.state === "completed" ? (
                  <Check size={16} />
                ) : step.state === "running" ? (
                  <LoaderCircle className="spin" size={17} />
                ) : step.state === "failed" ? (
                  <AlertTriangle size={16} />
                ) : (
                  <Icon size={17} />
                )}
              </span>
              <small>{step.label}</small>
            </div>
            {index < stages.length - 1 && (
              <ChevronRight size={16} className={step.state === "completed" ? "active" : ""} />
            )}
          </React.Fragment>
        );
      })}
    </div>
  );
}

export function EmptyState() {
  return (
    <div className="empty-state">
      <div className="empty-graphic">
        <span>
          <Globe2 size={26} />
        </span>
        <ArrowRight size={22} />
        <span>
          <Network size={26} />
        </span>
        <ArrowRight size={22} />
        <span>
          <Sparkles size={26} />
        </span>
      </div>
      <h2>One page. Two graph views. Six model runs.</h2>
      <p>
        Enter a public webpage. The server captures the exact page, builds accessibility-tree and
        rendered-visual graphs, compares MLP, GraphSAGE and GAT, then asks the configured LLM for
        bounded suggestions.
      </p>
      <div className="boundaries">
        <span>
          <Check size={15} />
          Three trained architectures
        </span>
        <span>
          <Check size={15} />
          Inspectable prompts
        </span>
        <span>
          <ShieldCheck size={15} />
          No automatic edits
        </span>
      </div>
    </div>
  );
}

export function PipelineLive({ job }: { job: Job }) {
  const events = job.progress?.events || [];

  return (
    <div className="pipeline-live">
      <div className="pipeline-live-heading">
        <div>
          <span className="section-label">LIVE EXECUTION</span>
          <h2>{job.progress?.label || "Preparing audit"}</h2>
          <code className="job-link">GET {job.links?.self || `/v1/jobs/${job.job_id}`}</code>
        </div>
        <span className={`job-state job-${job.status}`}>
          <Activity size={14} />
          {job.status}
        </span>
      </div>
      <div className="event-list">
        {events.length === 0 && (
          <div className="event-row state-running">
            <LoaderCircle className="spin" size={18} />
            <div>
              <strong>Waiting for a worker</strong>
              <small>The accepted job is queued.</small>
            </div>
          </div>
        )}
        {events.map((event) => (
          <div className={`event-row state-${event.status}`} key={event.event_id}>
            {event.status === "completed" ? (
              <Check size={18} />
            ) : event.status === "running" ? (
              <LoaderCircle className="spin" size={18} />
            ) : (
              <AlertTriangle size={18} />
            )}
            <div>
              <strong>{event.label}</strong>
              <small>
                {Object.entries(event.details || {})
                  .filter(([, value]) => value !== null && value !== undefined)
                  .slice(0, 4)
                  .map(([key, value]) => `${key.replaceAll("_", " ")}: ${String(value)}`)
                  .join(" · ") || "In progress"}
              </small>
            </div>
            {event.duration_ms !== undefined && (
              <span>
                <Clock3 size={13} />
                {event.duration_ms < 1000
                  ? `${event.duration_ms} ms`
                  : `${(event.duration_ms / 1000).toFixed(1)} s`}
              </span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

export function SubmittingPipeline() {
  return (
    <div className="pipeline-live">
      <div className="pipeline-live-heading">
        <div>
          <span className="section-label">LIVE EXECUTION</span>
          <h2>Submitting audit request</h2>
        </div>
        <span className="job-state">
          <LoaderCircle className="spin" size={14} />
          submitting
        </span>
      </div>
      <div className="event-list">
        <div className="event-row state-running">
          <LoaderCircle className="spin" size={18} />
          <div>
            <strong>Create background job</strong>
            <small>POST /v1/suggestion-audits</small>
          </div>
        </div>
      </div>
    </div>
  );
}
