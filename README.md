# no-unconsumed-fetch

An ESLint rule that flags `fetch()` calls whose response body is never consumed or cancelled.

## Why?

In the Fetch API, if you call `fetch()` and don't consume or cancel the body, the underlying connection can stay open longer than necessary (especially in Node.js or when using HTTP/1.1 keep-alive). That can lead to:

- Connection pool exhaustion
- Stalled requests
- Subtle memory/resource leaks

See [sanity-io/sanity#12681](https://github.com/sanity-io/sanity/pull/12681) for a real-world example where an unconsumed 41-byte response caused a **156-second** stall.

## Installation

```bash
npm install --save-dev no-unconsumed-fetch
```

## Usage (flat config — `eslint.config.js`)

```js
import noUnconsumedFetch from "no-unconsumed-fetch";

export default [
  noUnconsumedFetch.configs.recommended,
  // ...your other configs
];
```

Or manually:

```js
import noUnconsumedFetch from "no-unconsumed-fetch";

export default [
  {
    plugins: {
      "no-unconsumed-fetch": noUnconsumedFetch,
    },
    rules: {
      "no-unconsumed-fetch/no-unconsumed-fetch": "error",
    },
  },
];
```

## Rule: `no-unconsumed-fetch/no-unconsumed-fetch`

### ❌ Invalid (triggers warning)

```js
// Response completely ignored
await fetch(url);

// Only reads status/headers, body left open
const res = await fetch(url);
console.log(res.status);

// Fire-and-forget
fetch(url);
```

### ✅ Valid (body consumed or response handed off)

```js
// Body consumed via .json()
const res = await fetch(url);
const data = await res.json();

// Body consumed via .text()
const res = await fetch(url);
await res.text();

// Body explicitly cancelled
const res = await fetch(url);
await res.body.cancel();

// Response returned (caller is responsible)
async function doFetch() {
  return await fetch(url);
}

// Response passed to another function
const res = await fetch(url);
processResponse(res);

// Chained directly
fetch(url).then((r) => r.json());
```

### Recognized consumption methods

| Method                   | Description                 |
| ------------------------ | --------------------------- |
| `res.json()`             | Parse body as JSON          |
| `res.text()`             | Read body as text           |
| `res.arrayBuffer()`      | Read body as ArrayBuffer    |
| `res.blob()`             | Read body as Blob           |
| `res.formData()`         | Read body as FormData       |
| `res.body.cancel()`      | Cancel the body stream      |
| `res.body.getReader()`   | Get a ReadableStream reader |
| `res.body.pipeTo()`      | Pipe the stream             |
| `res.body.pipeThrough()` | Pipe through a transform    |

## Development

```bash
pnpm install
pnpm run build   # Compile TypeScript → dist/
pnpm test        # Run rule tests
pnpm run lint    # Lint with oxlint
pnpm run format  # Format with oxfmt
```

## Releasing

Versioning and publishing are managed by [changesets](https://github.com/changesets/changesets).

1. After a user-facing change, add a changeset:

   ```bash
   pnpm changeset
   ```

   Select `patch`, `minor`, or `major` and describe the change. The generated `.changeset/*.md` file is committed alongside the change.

2. When changesets are merged to `main`, the `Release` workflow opens (or updates) a **"chore: release"** PR that bumps the version and regenerates `CHANGELOG.md`.

3. Merging that PR triggers the same workflow again, which publishes to npm via OIDC trusted publishing (no token required) and pushes a git tag.

## License

ISC
