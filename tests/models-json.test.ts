import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  assertPublicOpenRouterFallbackAllowed,
  createModelRuntime,
  loadModelsJsonConfig,
} from "../src/models-json.js";

const VALID_MODELS = {
  providers: {
    MyCorp: {
      baseUrl: "https://llm.example.com/v1",
      api: "openai-completions",
      models: [
        {
          id: "my-model",
          reasoning: true,
          contextWindow: 131072,
          maxTokens: 32768,
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        },
      ],
    },
  },
};

let dir: string;

function writeModels(name: string, body: string): string {
  const path = join(dir, name);
  writeFileSync(path, body);
  return path;
}

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "hodor-models-json-"));
});

afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
});

describe("loadModelsJsonConfig", () => {
  it("returns null when the variable is unset or blank", () => {
    expect(loadModelsJsonConfig(undefined)).toBeNull();
    expect(loadModelsJsonConfig("  ")).toBeNull();
  });

  it("fails when the file does not exist", () => {
    expect(() => loadModelsJsonConfig(join(dir, "missing.json"))).toThrow(/readable models file.*ENOENT/);
  });

  it("fails when the path is a directory", () => {
    const sub = join(dir, "models.json");
    mkdirSync(sub);
    expect(() => loadModelsJsonConfig(sub)).toThrow(/readable models file/);
  });

  it("fails on invalid JSON without echoing the file contents", () => {
    const path = writeModels("bad.json", '{"providers": {"x": "sk-secret-value"');
    let message = "";
    try {
      loadModelsJsonConfig(path);
    } catch (err) {
      message = err instanceof Error ? err.message : String(err);
    }
    expect(message).toMatch(/not valid JSON/);
    expect(message).not.toContain("sk-secret-value");
  });

  it("fails when there is no providers object", () => {
    const path = writeModels("empty.json", '{"models": []}');
    expect(() => loadModelsJsonConfig(path)).toThrow(/"providers" object/);
  });

  it("returns the provider keys with their exact case", () => {
    const path = writeModels("models.json", JSON.stringify(VALID_MODELS));
    const config = loadModelsJsonConfig(path);
    expect(config?.path).toBe(path);
    expect([...(config?.providers ?? [])]).toEqual(["MyCorp"]);
  });
});

describe("createModelRuntime", () => {
  it("resolves a mixed-case custom provider through Pi", async () => {
    const path = writeModels("models.json", JSON.stringify(VALID_MODELS));
    const runtime = await createModelRuntime(loadModelsJsonConfig(path));
    const model = runtime.getModel("MyCorp", "my-model");
    expect(model?.baseUrl).toBe("https://llm.example.com/v1");
  });

  it("rejects a models file that Pi's schema refuses", async () => {
    const path = writeModels(
      "invalid.json",
      JSON.stringify({ providers: { MyCorp: { baseUrl: "https://llm.example.com/v1", models: "not-a-list" } } }),
    );
    await expect(createModelRuntime(loadModelsJsonConfig(path))).rejects.toThrow(/HODOR_MODELS_JSON .* was rejected/);
  });

  it("creates the stock runtime when no models file is configured", async () => {
    const runtime = await createModelRuntime(null);
    expect(runtime.getError()).toBeUndefined();
  });
});

describe("assertPublicOpenRouterFallbackAllowed", () => {
  it("allows the public fallback without a models file", () => {
    expect(() => assertPublicOpenRouterFallbackAllowed(null, "vendor/new-model")).not.toThrow();
  });

  it("allows the public fallback when the file does not configure openrouter", () => {
    const config = { path: "/models.json", providers: new Set(["MyCorp"]) };
    expect(() => assertPublicOpenRouterFallbackAllowed(config, "vendor/new-model")).not.toThrow();
  });

  it("refuses the public fallback when the file configures openrouter", () => {
    const config = { path: "/models.json", providers: new Set(["openrouter"]) };
    expect(() => assertPublicOpenRouterFallbackAllowed(config, "vendor/new-model")).toThrow(
      /will not fall back to the public OpenRouter endpoint/,
    );
  });
});
