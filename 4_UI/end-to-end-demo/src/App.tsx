import { AlertTriangle, Globe2, LoaderCircle, ShieldCheck, Sparkles } from "lucide-react";
import { type FormEvent, useRef, useState } from "react";
import { api, sleep } from "./lib/api";
import { EmptyState, PipelineLive, SubmittingPipeline, Workflow } from "./components/Pipeline";
import { Results } from "./components/Results";
import type { Job, SuggestionResult } from "./types";

export function App() {
  const [url, setUrl] = useState("https://www.w3.org/");
  const maximum = 5;
  const [job, setJob] = useState<Job | null>(null);
  const [result, setResult] = useState<SuggestionResult | null>(null);
  const [error, setError] = useState("");
  const [auditLoading, setAuditLoading] = useState(false);
  const submittingRef = useRef(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (submittingRef.current) return;

    submittingRef.current = true;
    setError("");
    setResult(null);
    setJob(null);
    setAuditLoading(true);

    try {
      const accepted = await api<Job>("/v1/suggestion-audits", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url, max_suggestions: maximum, timeout_seconds: 45 }),
      });
      setJob(accepted);

      let current = accepted;
      for (let attempt = 0; attempt < 1200; attempt += 1) {
        await sleep(750);
        current = await api<Job>(`/v1/jobs/${accepted.job_id}`);
        setJob(current);
        if (current.status === "failed") throw new Error(current.error || "Audit job failed");
        if (current.status === "completed") break;
      }

      if (current.status !== "completed") {
        throw new Error(
          "The audit did not finish within 15 minutes. It may still be running; check the displayed job URL.",
        );
      }

      const completedResult = await api<SuggestionResult>(`/v1/jobs/${accepted.job_id}/result`);
      setResult(completedResult);
      setJob(await api<Job>(`/v1/jobs/${accepted.job_id}`));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The audit failed");
    } finally {
      submittingRef.current = false;
      setAuditLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <main id="top">
        <section className="hero">
          <div className="hero-copy" />
          <form className="url-form" onSubmit={submit}>
            <label htmlFor="page-url">Public webpage URL</label>
            <div className="url-input">
              <Globe2 size={20} />
              <input
                id="page-url"
                type="url"
                required
                value={url}
                onChange={(event) => setUrl(event.target.value)}
                placeholder="https://example.com"
              />
              <button type="submit" disabled={auditLoading}>
                {auditLoading ? (
                  <LoaderCircle className="spin" size={18} />
                ) : (
                  <Sparkles size={18} />
                )}
                {auditLoading ? "Running pipeline…" : "Generate suggestions"}
              </button>
            </div>
          </form>
        </section>
        <Workflow job={job} />
        {error && (
          <div className="error-banner" role="alert">
            <AlertTriangle size={19} />
            <div>
              <strong>Audit failed</strong>
              <p>{error}</p>
            </div>
          </div>
        )}
        <section className="workspace">
          {result && job ? (
            <Results key={result.run_id} result={result} job={job} />
          ) : auditLoading ? (
            job ? (
              <PipelineLive job={job} />
            ) : (
              <SubmittingPipeline />
            )
          ) : (
            <EmptyState />
          )}
        </section>
      </main>
      <footer>
        <span>
          made by Ayman Patel
        </span>
      </footer>
    </div>
  );
}
