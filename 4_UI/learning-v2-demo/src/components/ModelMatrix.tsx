import type { ModelRun } from "../types";
import { architectureLabel, viewLabel } from "../lib/format";

function ModelCard({ run }: { run: ModelRun }) {
  const failures = run.rules.filter((rule) => rule.predicted_fail);

  return (
    <article className="model-card">
      <div className="model-card-heading">
        <div>
          <span className={`architecture architecture-${run.architecture}`}>
            {architectureLabel(run.architecture)}
          </span>
          <h3>{viewLabel(run.view)}</h3>
        </div>
        <span className="finding-badge">{failures.length} over threshold</span>
      </div>
      <div className="graph-stats">
        <span>
          <strong>{run.node_count.toLocaleString()}</strong> nodes
        </span>
        <span>
          <strong>{run.edge_count.toLocaleString()}</strong> edges
        </span>
        <span>
          <strong>{run.findings.length}</strong> findings
        </span>
      </div>
      <details>
        <summary>Rule probabilities and thresholds</summary>
        <div className="rule-table">
          {run.rules.map((rule) => (
            <div key={rule.rule_id} className={rule.predicted_fail ? "predicted-fail" : ""}>
              <code>{rule.rule_id}</code>
              <span>{(rule.probability * 100).toFixed(1)}%</span>
              <span>threshold {(rule.threshold * 100).toFixed(1)}%</span>
              <strong>{rule.predicted_fail ? "FAIL" : "pass"}</strong>
            </div>
          ))}
        </div>
      </details>
      <small className="checkpoint">
        checkpoint {run.checkpoint_sha256.slice(0, 12)}… · axe-free inference
      </small>
    </article>
  );
}

export function ModelMatrix({ runs }: { runs: ModelRun[] }) {
  return (
    <section className="model-section">
      <div className="panel-heading">
        <div>
          <span className="section-label">TRAINED MODEL LAYER</span>
          <h2>Three architectures × two graph views</h2>
        </div>
        <span className="matrix-count">{runs.length} frozen checkpoint runs</span>
      </div>
      <div className="model-matrix">
        {runs.map((run) => (
          <ModelCard run={run} key={`${run.architecture}-${run.view}`} />
        ))}
      </div>
    </section>
  );
}
