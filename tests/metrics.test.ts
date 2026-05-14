import { describe, expect, it, vi, afterEach } from "vitest";
import { pushMetrics } from "../src/metrics.js";
import type { ReviewMetrics } from "../src/types.js";

const baseMetrics: ReviewMetrics = {
  inputTokens: 300,
  outputTokens: 50,
  cacheReadTokens: 700,
  cacheWriteTokens: 125,
  totalTokens: 1175,
  cost: 0.0123,
  turns: 4,
  toolCalls: 9,
  durationSeconds: 42,
};

describe("pushMetrics", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("pushes cache write tokens, cache hit ratio, findings by priority, and labels", async () => {
    const mockFetch = vi.fn(async () => new Response("", { status: 202 }));
    vi.stubGlobal("fetch", mockFetch);

    await pushMetrics({
      pushgatewayUrl: "https://pushgateway.example.com/",
      metrics: baseMetrics,
      findings: [{ priority: 1 }, { priority: 2 }, { priority: 2 }],
      labels: {
        platform: "gitlab",
        model: "anthropic/claude",
        verdict: "incorrect",
        project: "acme/web-app",
        mr_iid: "732",
      },
    });

    expect(mockFetch).toHaveBeenCalledWith(
      "https://pushgateway.example.com/metrics/job/hodor",
      expect.objectContaining({ method: "POST" }),
    );
    const body = String(mockFetch.mock.calls[0]?.[1]?.body);
    expect(body).toContain("hodor_review_cache_write_tokens_total");
    expect(body).toContain("hodor_review_cache_write_tokens_total{platform=\"gitlab\",model=\"anthropic/claude\",verdict=\"incorrect\",project=\"acme/web-app\",mr_iid=\"732\"} 125");
    expect(body).toContain("hodor_review_cache_hit_ratio{platform=\"gitlab\",model=\"anthropic/claude\",verdict=\"incorrect\",project=\"acme/web-app\",mr_iid=\"732\"} 0.7");
    expect(body).toContain("hodor_review_findings_total{platform=\"gitlab\",model=\"anthropic/claude\",verdict=\"incorrect\",project=\"acme/web-app\",mr_iid=\"732\",priority=\"P0\"} 0");
    expect(body).toContain("hodor_review_findings_total{platform=\"gitlab\",model=\"anthropic/claude\",verdict=\"incorrect\",project=\"acme/web-app\",mr_iid=\"732\",priority=\"P1\"} 1");
    expect(body).toContain("hodor_review_findings_total{platform=\"gitlab\",model=\"anthropic/claude\",verdict=\"incorrect\",project=\"acme/web-app\",mr_iid=\"732\",priority=\"P2\"} 2");
    expect(body).toContain("hodor_review_findings_total{platform=\"gitlab\",model=\"anthropic/claude\",verdict=\"incorrect\",project=\"acme/web-app\",mr_iid=\"732\",priority=\"P3\"} 0");
  });

  it("emits zero cache hit ratio when there are no input tokens", async () => {
    const mockFetch = vi.fn(async () => new Response("", { status: 202 }));
    vi.stubGlobal("fetch", mockFetch);

    await pushMetrics({
      pushgatewayUrl: "https://pushgateway.example.com",
      metrics: { ...baseMetrics, inputTokens: 0, cacheReadTokens: 0 },
    });

    const body = String(mockFetch.mock.calls[0]?.[1]?.body);
    expect(body).toContain("hodor_review_cache_hit_ratio 0");
  });

  it("posts directly to VictoriaMetrics prometheus import endpoints", async () => {
    const mockFetch = vi.fn(async () => new Response("", { status: 202 }));
    vi.stubGlobal("fetch", mockFetch);

    await pushMetrics({
      pushgatewayUrl: "https://metrics.example.com/api/v1/import/prometheus",
      metrics: baseMetrics,
    });

    expect(mockFetch).toHaveBeenCalledWith(
      "https://metrics.example.com/api/v1/import/prometheus",
      expect.objectContaining({ method: "POST" }),
    );
  });
});
