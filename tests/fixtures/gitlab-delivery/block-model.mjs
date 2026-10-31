import { appendFileSync } from 'node:fs';
import { ModelRuntime } from '@earendil-works/pi-coding-agent';

// Keep the real runtime, session, cache lookup, CLI, and publisher. Block only the LLM boundary.
ModelRuntime.prototype.streamSimple = function () {
  appendFileSync(process.env.HODOR_TEST_MODEL_CALLS, 'blocked\n');
  throw new Error('Synthetic LLM stream blocked: fresh review computation required');
};
