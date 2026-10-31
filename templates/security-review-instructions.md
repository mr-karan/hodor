# Security Review

Report only security vulnerabilities introduced or made reachable by the proposed change. Report concrete, actionable issues, not generic hardening advice. Prefer no finding over an unsupported finding.

## Evidence standard
For each finding, establish:
1. Entry point: where an attacker can supply data or perform an action.
2. Control: what the attacker controls and what access they require.
3. Path: how that input or action reaches the affected operation.
4. Boundary: which authentication, authorization, isolation, integrity, or confidentiality guarantee fails.
5. Impact: what the attacker can actually read, change, execute, or disrupt.

Use code you inspected and documented platform or library behavior. Do not invent deployment conditions, external service behavior, or attacker capabilities. Do not assume that a control you have not seen exists, or that it is absent. Evidence from code reading is sufficient; do not run an exploit.
A configuration change or exposed credential may show a boundary failure directly; it need not resemble a request-to-sink code path.

## Investigation
- Inspect removed controls as carefully as added code.
- Trace changed entry points toward sensitive operations, and changed sensitive operations back toward their callers.
- Read the relevant route registration, middleware, permission checks, schemas, and directly affected callers.
- Follow stored attacker-controlled data into later processing.
- For a logged, serialized, or returned object, read its definition to see which fields leave the process.
- Check failure paths, alternate branches, concurrency, and partial writes.
- When the decisive control could live outside the repository, such as in a gateway or proxy, report its absence only when code here shows the control belongs locally: sibling handlers enforce it, the change removes it, or the change registers the operation outside the protected group.
- Keep investigation targeted. Stop tracing when the path reaches an entry point, a value the attacker cannot influence, or a control that meets the tests below.

A control counts only when it:
- Runs on the affected path before the sensitive operation.
- Covers the relevant branches and terminates on rejection.
- Fails closed when its secret, configuration, or dependency is missing, empty, or errors.
- Validates the same identity, object, destination, or value used afterward, against a trusted source such as the verified session, not other request data.
- Actually prevents the traced attack.

A control's name, a nearby comment, a pull request description, or a suppression marker is not sufficient evidence. Absence from one function is not proof of absence from the full path.

## Relevant vulnerability classes
Apply only the classes relevant to the change:
- Authentication bypass, session flaws, account takeover, and removed or weakened limits on guessing passwords, OTPs, or tokens.
- Missing authorization, access to another user's or tenant's objects (IDOR), privilege escalation, and client-supplied ownership or role fields.
- SQL, command, template, and other interpreter injection.
- Path traversal, unsafe file operations, and unsafe deserialization.
- SSRF and attacker-controlled outbound destinations.
- XSS, CSRF, unsafe redirects, and browser trust-boundary failures.
- Signature-verification, cryptographic, and replay-protection failures.
- Secret exposure and sensitive-data leakage through responses, logs, artifacts, images, or client bundles.
- Races and business-logic flaws that bypass security controls.
- CI, infrastructure, and workload-isolation changes that grant unauthorized access. In CI configuration, anyone who can push a branch or open a pull request is a possible attacker against jobs that hold credentials.
- Resource exhaustion only when one or a few attacker requests cause disproportionate work, unbounded memory or storage, fan-out, or paid side effects. Request volume alone is not a finding.

This list guides investigation; it is not a checklist of findings to emit.

## Avoid unsupported findings
Do not report:
- Pre-existing issues unaffected by the change.
- A dependency CVE solely from its version number, without evidence that the affected behavior is reachable and applicable.
- Missing hardening without a concrete security consequence.
- Attacks that need access equal to or greater than what they would gain, including intended admin behavior.
- A weak hash used only as a non-security checksum or cache key.
- A dangerous API name without attacker influence and a relevant path.
- Self-XSS, log injection without a downstream interpreter, and CSRF against requests that browsers cannot authenticate automatically.
- Hypothetical chains requiring multiple unproven assumptions.

Do not categorically ignore tests, examples, or development configuration. Report them only when the change creates a real exposure, such as a credential that is not an obvious synthetic placeholder or public test key, a published artifact, or code that runs in a deployed environment.
Do not assume constants, configuration, environment variables, or stored values are trusted or untrusted. Treat a value as attacker-influenced only when you can identify who can write it and that writer has less privilege than the code that uses it.

## Severity
Use Hodor's priority definitions without a separate security priority mapping.
Base severity on traced impact, attacker prerequisites, reachability, and scope. State material conditions explicitly.
Do not assign P0 merely because the issue belongs to a severe vulnerability class. Do not downgrade a proven issue merely because the fix is small.

## Finding quality
Write one finding per distinct failed control or root cause.
In one concise paragraph, identify the attacker's input or action and required access, explain the reachable path and failed boundary, state the concrete impact and material conditions, and give the minimal corrective direction when established.

Anchor the finding at the smallest relevant changed-code location. For a removed control, anchor on the diff lines next to the removal. Use a recognizable vulnerability class when helpful.
Never reproduce a secret value in a finding, code excerpt, or suggestion. Describe its type and location without disclosing it.
