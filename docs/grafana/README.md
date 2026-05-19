# Grafana dashboard

`hodor-review-metrics.json` is a generic dashboard for metrics emitted by `--prometheus-push`.

## Import

1. Grafana: Dashboards > New > Import.
2. Upload `docs/grafana/hodor-review-metrics.json`.
3. Select your Prometheus-compatible datasource.

The dashboard uses a `$datasource` variable and does not include environment-specific URLs, folder UIDs, cookies, or service account tokens.

## Metrics

Expected series:

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

Common labels:

- `platform`
- `project`
- `model`
- `verdict`
- `priority`

## Pushgateway note

Some panels treat Hodor runs as events with `count_over_time()` or `sum_over_time()`.

Those panels are most accurate when metrics are sent directly to a text import endpoint such as VictoriaMetrics `/api/v1/import/prometheus`, where each run writes one sample.

With Prometheus Pushgateway, repeated scrapes of the same pushed sample can over-count run events. Latest-value panels for cost, tokens, duration, and findings remain useful.
