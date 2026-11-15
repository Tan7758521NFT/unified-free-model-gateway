# Unified Free-Model Gateway

One server, two APIs: OpenAI Chat Completions and Anthropic Messages, served over a
pool of free-tier providers (OpenAI, Anthropic and Gemini wires).

For each request the gateway reads every provider's catalog, picks the best free
model that satisfies the request (tools, images, JSON mode, long context), and
fails over when a provider's trial budget runs out (402 on one provider, 429 on
the others, honoring `retry-after`). It never picks a paid model, and every
replay is deterministic.

## Run

Requires Node 22+ (standard library only, nothing to install):

```sh
node main.ts            # serves the gateway
node main.ts --list     # print the registered catalog
```

## How it works

- **Plan first:** an up-front 20-request plan from the sealed quality table, with
  per-provider planned-call caps under the published per-trial budget minimums.
- **Route:** score free models by quality, capability match and remaining budget.
- **Fail over:** per-provider chains with `Retry-After`; one call at a time per
  provider.
- **Translate faithfully:** all six wire paths (OpenAI / Anthropic / Gemini ×
  JSON / SSE), including Anthropic tool calls, images and stop reasons.

## License

MIT
