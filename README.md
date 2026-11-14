# free-model-aggregator

The best free AI model aggregator. One server (`main.py`, Python 3.13 standard
library only) serving an OpenAI-compatible API and an Anthropic-compatible API
over ten simulated free-tier providers.

The server reads the trial brief, probes each provider's catalog, picks the
best free model meeting each request's requirements (by the brief's quality
table), fails over when a provider refuses, and returns the provider's answer
in the wire the client used. JSON and SSE streaming, tool calls, images, JSON
mode, and long context are all carried through untouched in meaning.

Run a trial: `python3 main.py` (the sandbox sets `MAKEFUN_ENV_URL`,
`MAKEFUN_PROMPT_FILE` and `MAKEFUN_TRIAL_SEED`). Print the catalog:
`python3 main.py --list`.

Built for the make.fun bounty `free-model-aggregator-01`.
