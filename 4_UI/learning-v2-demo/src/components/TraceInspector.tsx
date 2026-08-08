import { Bot, Braces, Copy, ShieldCheck, Sparkles, Terminal } from "lucide-react";
import { useState } from "react";
import type { Job, Suggestion, SuggestionResult } from "../types";

function CodePanel({ value, language = "json" }: { value: unknown; language?: "json" | "text" }) {
  const rendered = language === "text" ? String(value || "") : JSON.stringify(value, null, 2);
  const [copied, setCopied] = useState(false);

  async function copy() {
    await navigator.clipboard.writeText(rendered);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1200);
  }

  return (
    <div className="code-panel">
      <button onClick={copy} type="button">
        <Copy size={13} />
        {copied ? "Copied" : "Copy"}
      </button>
      <pre>
        <code>{rendered}</code>
      </pre>
    </div>
  );
}

export function TraceInspector({
  result,
  job,
  suggestion,
}: {
  result: SuggestionResult;
  job: Job;
  suggestion?: Suggestion;
}) {
  const [tab, setTab] = useState<"fastapi" | "system" | "user" | "response">("fastapi");
  const trace = suggestion?.api_trace;

  return (
    <section className="trace-section">
      <div className="panel-heading">
        <div>
          <span className="section-label">TRANSPARENT API LAYER</span>
          <h2>Requests, prompts and structured output</h2>
        </div>
        <span className="safe-trace">
          <ShieldCheck size={14} />
          Secrets omitted
        </span>
      </div>
      <div className="trace-tabs" role="tablist">
        <button className={tab === "fastapi" ? "active" : ""} onClick={() => setTab("fastapi")}>
          <Terminal size={15} />
          FastAPI calls
        </button>
        <button
          disabled={!trace}
          className={tab === "system" ? "active" : ""}
          onClick={() => setTab("system")}
        >
          <Bot size={15} />
          System prompt
        </button>
        <button
          disabled={!trace}
          className={tab === "user" ? "active" : ""}
          onClick={() => setTab("user")}
        >
          <Braces size={15} />
          User JSON
        </button>
        <button
          disabled={!trace}
          className={tab === "response" ? "active" : ""}
          onClick={() => setTab("response")}
        >
          <Sparkles size={15} />
          LLM response
        </button>
      </div>
      {tab === "fastapi" && (
        <CodePanel value={{ calls: result.application_api, latest_job: job }} />
      )}
      {tab === "system" && <CodePanel value={trace?.request.system_prompt} language="text" />}
      {tab === "user" && (
        <CodePanel
          value={{
            request: {
              method: trace?.request.method,
              endpoint: trace?.request.endpoint,
              api_mode: trace?.request.api_mode,
              model: trace?.request.model,
              response_format: trace?.request.response_format,
            },
            messages: [
              { role: "system", content: "Shown in System prompt tab" },
              { role: "user", content: trace?.request.user_prompt },
            ],
          }}
        />
      )}
      {tab === "response" && <CodePanel value={trace?.response || { status: "No LLM response" }} />}
    </section>
  );
}
