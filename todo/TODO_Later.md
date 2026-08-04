Regenerate the frozen dataset using version-2 visual and Chromium live-AX collectors.
Independently annotate every final site–criterion pair with two annotators and adjudication.
Run multi-seed visual and GNN experiments.
Repeat the matched LLM study and add blinded human repair ratings.
Run the final held-out study once using the frozen configuration.


You now need to complete the external evaluation work. Do these steps in order.

1. Complete the detection annotations

Ask two people to independently complete:

- [Rater 1 detection sheet](/Users/aymanpatel/Desktop/Uni/Dissertation/3_Learning/learning_v2/artifacts_3107_0015/detection_annotation_packet/rater_packets/rater_1.json)
- [Rater 2 detection sheet](/Users/aymanpatel/Desktop/Uni/Dissertation/3_Learning/learning_v2/artifacts_3107_0015/detection_annotation_packet/rater_packets/rater_2.json)

They should follow [ANNOTATION_INSTRUCTIONS.md](/Users/aymanpatel/Desktop/Uni/Dissertation/3_Learning/learning_v2/artifacts_3107_0015/detection_annotation_packet/ANNOTATION_INSTRUCTIONS.md), work independently, and not inspect axe/model outputs.

Then run:

```bash
python -u -m learning_v2.annotation_finalize \
  --packet-dir learning_v2/artifacts_3107_0015/detection_annotation_packet \
  --output learning_v2/artifacts_3107_0015/detection_annotation_packet/final_independent_detection_truth.json
```

If it reports disagreements, complete the requested adjudication entries and rerun it.

2. Run LLM replicates `r2` and `r3`

Run the commands under “Complete the predeclared stochastic repair replicates” in [CORRECTED_PILOT_COMMANDS.md](/Users/aymanpatel/Desktop/Uni/Dissertation/3_Learning/learning_v2/CORRECTED_PILOT_COMMANDS.md).

These make six additional paid OpenRouter runs:

- Three RAG conditions for `r2`
- Three RAG conditions for `r3`

You can explicitly authorize me to run these if you want me to execute the paid API requests.

3. Complete blinded repair ratings

Ask two people to independently complete:

- [Rater 1 repair sheet](/Users/aymanpatel/Desktop/Uni/Dissertation/3_Learning/learning_v2/artifacts_3107_0015/repair_benchmark_v3/human_rating_packet/rater_packets/rater_1.json)
- [Rater 2 repair sheet](/Users/aymanpatel/Desktop/Uni/Dissertation/3_Learning/learning_v2/artifacts_3107_0015/repair_benchmark_v3/human_rating_packet/rater_packets/rater_2.json)

They must follow [RATING_INSTRUCTIONS.md](/Users/aymanpatel/Desktop/Uni/Dissertation/3_Learning/learning_v2/artifacts_3107_0015/repair_benchmark_v3/human_rating_packet/RATING_INSTRUCTIONS.md) and must not inspect the `coordinator` directory.

Then run:

```bash
python -u -m accessibility_system.evaluation.repair_rating_packet finalize \
  --packet-dir learning_v2/artifacts_3107_0015/repair_benchmark_v3/human_rating_packet \
  --output learning_v2/artifacts_3107_0015/repair_benchmark_v3/human_ratings.json
```

4. Generate the final results

Run the commands under “Run the frozen final studies after human labels exist” in [CORRECTED_PILOT_COMMANDS.md](/Users/aymanpatel/Desktop/Uni/Dissertation/3_Learning/learning_v2/CORRECTED_PILOT_COMMANDS.md).

Finally check readiness:

```bash
python -u -m learning_v2.readiness_audit
```

The target output is:

```json
{
  "dissertation_ready": true
}
```