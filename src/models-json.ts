import { readFileSync, statSync } from "node:fs";
import { resolve } from "node:path";
import { ModelRuntime } from "@earendil-works/pi-coding-agent";
import { InMemoryCredentialStore } from "@earendil-works/pi-ai";

export const MODELS_JSON_ENV = "HODOR_MODELS_JSON";

/** A trusted, operator-supplied pi-format models.json. */
export interface ModelsJsonConfig {
  path: string;
  /** Provider keys defined in the file, with their exact case. */
  providers: ReadonlySet<string>;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Load the models file named by HODOR_MODELS_JSON, or null when it is unset.
 *
 * Pi treats a missing file as "no custom models" and keeps its built-in
 * endpoints. A typo would then send a review meant for a private gateway, and
 * its API key, to a public provider. Fail instead. Errors name the path but
 * never echo file contents.
 */
export function loadModelsJsonConfig(
  value: string | undefined = process.env[MODELS_JSON_ENV],
): ModelsJsonConfig | null {
  if (value === undefined || value.trim() === "") return null;
  const path = resolve(value.trim());

  let raw: string;
  try {
    if (!statSync(path).isFile()) throw new Error("not a regular file");
    raw = readFileSync(path, "utf-8");
  } catch (err) {
    const reason = err instanceof Error && "code" in err ? String(err.code) : "unreadable";
    throw new Error(`${MODELS_JSON_ENV} must point at a readable models file: ${path} (${reason})`);
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    throw new Error(`${MODELS_JSON_ENV} file is not valid JSON: ${path}`);
  }
  if (!isRecord(parsed) || !isRecord(parsed.providers)) {
    throw new Error(`${MODELS_JSON_ENV} file must contain a "providers" object: ${path}`);
  }
  return { path, providers: new Set(Object.keys(parsed.providers)) };
}

/**
 * Create the Pi model runtime. With a models file, reject any schema or
 * provider error Pi reports, because Pi would otherwise keep serving the
 * built-in definitions for the affected providers.
 */
export async function createModelRuntime(modelsJson: ModelsJsonConfig | null): Promise<ModelRuntime> {
  const runtime = await ModelRuntime.create({
    credentials: new InMemoryCredentialStore(),
    modelsPath: modelsJson?.path ?? null,
  });
  if (modelsJson) {
    const error = runtime.getError();
    if (error) {
      throw new Error(`${MODELS_JSON_ENV} (${modelsJson.path}) was rejected:\n${error}`);
    }
  }
  return runtime;
}

/**
 * Hodor builds a best-effort descriptor for OpenRouter slugs missing from the
 * registry, and it targets public OpenRouter. Refuse that when the models file
 * configures the openrouter provider, or the fallback would bypass the file's
 * gateway and send the review and API key to the public endpoint.
 */
export function assertPublicOpenRouterFallbackAllowed(
  modelsJson: ModelsJsonConfig | null,
  modelId: string,
): void {
  if (!modelsJson?.providers.has("openrouter")) return;
  throw new Error(
    `OpenRouter model "${modelId}" is not defined in ${MODELS_JSON_ENV}. The file configures the openrouter provider, so Hodor will not fall back to the public OpenRouter endpoint. Add the model to the file.`,
  );
}
