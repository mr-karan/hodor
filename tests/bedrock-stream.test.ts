import { describe, it, expect } from "vitest";
import { Agent } from "@earendil-works/pi-agent-core";
import type { StreamFn } from "@earendil-works/pi-agent-core";
import { wrapBedrockStream } from "../src/agent.js";

type StreamOptions = NonNullable<Parameters<StreamFn>[2]>;
type PayloadHook = NonNullable<StreamOptions["onPayload"]>;

class Captured extends Error {
  constructor(readonly options: StreamOptions | undefined) {
    super("captured");
  }
}

// Inner stream function that reports the options it receives, then stops the call.
const capturingStream: StreamFn = async (_model, _context, options) => {
  throw new Captured(options);
};

async function captureOptions(
  stream: StreamFn,
  model: Parameters<StreamFn>[0],
  options?: StreamOptions,
): Promise<StreamOptions | undefined> {
  try {
    await stream(model, { messages: [] }, options);
  } catch (err) {
    if (err instanceof Captured) return err.options;
    throw err;
  }
  throw new Error("inner stream function was not called");
}

const model: Parameters<StreamFn>[0] = {
  id: "openai.gpt-oss-120b-1:0",
  name: "gpt-oss-120b",
  api: "bedrock-converse-stream",
  provider: "amazon-bedrock",
  baseUrl: "",
  reasoning: true,
  input: ["text"],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  contextWindow: 1000,
  maxTokens: 1000,
};

describe("wrapBedrockStream", () => {
  it("adds reasoning fields and requestMetadata", async () => {
    const wrapped = wrapBedrockStream(capturingStream, {
      bedrockTags: { team: "review" },
      openAiReasoning: "high",
    });
    const options = await captureOptions(wrapped, model);
    expect(options).toHaveProperty("requestMetadata", { team: "review" });
    const result = await options?.onPayload?.({ modelId: "x" }, model);
    expect(result).toEqual({
      modelId: "x",
      additionalModelRequestFields: { reasoning: { effort: "high" } },
    });
  });

  it("chains an existing onPayload before adding reasoning", async () => {
    const wrapped = wrapBedrockStream(capturingStream, { openAiReasoning: "low" });
    const chained: PayloadHook = (payload) => Object.assign({}, payload, { chained: true });
    const options = await captureOptions(wrapped, model, { onPayload: chained });
    const result = await options?.onPayload?.({ a: 1 }, model);
    expect(result).toEqual({
      a: 1,
      chained: true,
      additionalModelRequestFields: { reasoning: { effort: "low" } },
    });
  });

  it("omits requestMetadata when no tags are set", async () => {
    const wrapped = wrapBedrockStream(capturingStream, { openAiReasoning: "low" });
    const options = await captureOptions(wrapped, model);
    expect(options).not.toHaveProperty("requestMetadata");
  });

  it("takes effect on the property the SDK agent loop reads", async () => {
    let received: StreamOptions | undefined;
    const inner: StreamFn = async (_model, _context, options) => {
      received = options;
      throw new Error("stop");
    };
    const agent = new Agent({ streamFn: inner, initialState: { model } });
    agent.streamFunction = wrapBedrockStream(agent.streamFunction, {
      bedrockTags: { team: "review" },
    });
    await agent.prompt("hi").catch(() => undefined);
    expect(received).toHaveProperty("requestMetadata", { team: "review" });
  });
});
