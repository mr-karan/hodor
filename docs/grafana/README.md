# Grafana Dashboards

This directory contains generic Grafana dashboard JSON for Hodor metrics.

## Hodor Review Metrics

Import `hodor-review-metrics.json` into Grafana to visualize metrics emitted by Hodor's `--prometheus-push` option.

### Expected metrics

The dashboard expects these Prometheus/VictoriaMetrics series:

- `hodor_review_input_tokens_total`
- `hodor_review_output_tokens_total`
- `hodor_review_cache_read_tokens_total`
- `hodor_review_cache_write_tokens_total`
- `hodor_review_cache_hit_ratio`
- `hodor_review_cost_dollars`
- `hodor_review_turns_total`
- `hodor_review_tool_calls_total`
- `hodor_review_duration_seconds`
- `hodor_review_findings_total{priority="P0"..."P3"}`

Common labels used by the dashboard:

- `platform`
- `project`
- `model`
- `verdict`
- `priority`

### Import

1. In Grafana, go to **Dashboards → New → Import**.
2. Upload `docs/grafana/hodor-review-metrics.json`.
3. Pick your Prometheus-compatible datasource when prompted.
4. Save the dashboard in your preferred folder.

The dashboard is datasource-agnostic via a `$datasource` variable and does not include any environment-specific URLs, folder UIDs, cookies, or service account tokens.

### Pushgateway vs direct import

Some panels use `count_over_time()` and `sum_over_time()` to treat Hodor runs as events. These are most accurate when metrics are sent directly to a Prometheus text import endpoint such as VictoriaMetrics `/api/v1/import/prometheus`, where each Hodor run writes one sample.

If you use a Prometheus Pushgateway that is scraped periodically, event-count panels may over-count repeated scrapes of the same pushed sample. Token/cost/duration "latest value" panels still remain useful, but aggregate run counts may need recording rules or a shorter Pushgateway retention strategy.
