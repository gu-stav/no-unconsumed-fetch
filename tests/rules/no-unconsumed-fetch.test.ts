import { RuleTester } from "eslint";
import * as tsParser from "@typescript-eslint/parser";
import rule = require("../../src/rules/no-unconsumed-fetch");

const ruleTester = new RuleTester({
  languageOptions: {
    ecmaVersion: 2022,
    sourceType: "module",
    globals: {
      fetch: "readonly",
      globalThis: "readonly",
      window: "readonly",
      self: "readonly",
      console: "readonly",
    },
  },
});

const tsLanguageOptions = {
  parser: tsParser,
  ecmaVersion: 2022 as const,
  sourceType: "module" as const,
  globals: {
    fetch: "readonly" as const,
    console: "readonly" as const,
  },
};

const unconsumed = [{ messageId: "unconsumed" as const }];

ruleTester.run("no-unconsumed-fetch", rule, {
  valid: [
    // -----------------------------------------------------------------
    // Body read methods
    // -----------------------------------------------------------------
    {
      name: "body read via .json()",
      code: `
        const res = await fetch(url);
        const data = await res.json();
      `,
    },
    {
      name: "body read via .text()",
      code: `
        const res = await fetch(url);
        await res.text();
      `,
    },
    {
      name: "body read via .arrayBuffer()",
      code: `
        const res = await fetch(url);
        await res.arrayBuffer();
      `,
    },
    {
      name: "body read via .blob()",
      code: `
        const res = await fetch(url);
        await res.blob();
      `,
    },
    {
      name: "body read via .formData()",
      code: `
        const res = await fetch(url);
        await res.formData();
      `,
    },

    // -----------------------------------------------------------------
    // Stream-consuming methods
    // -----------------------------------------------------------------
    {
      name: "stream cancelled via res.body.cancel()",
      code: `
        const res = await fetch(url);
        await res.body.cancel();
      `,
    },
    {
      name: "stream read via res.body.getReader()",
      code: `
        const res = await fetch(url);
        const reader = res.body.getReader();
      `,
    },
    {
      name: "stream piped via res.body.pipeTo()",
      code: `
        const res = await fetch(url);
        await res.body.pipeTo(writable);
      `,
    },
    {
      name: "stream piped via res.body.pipeThrough()",
      code: `
        const res = await fetch(url);
        res.body.pipeThrough(transform);
      `,
    },
    {
      name: "async iteration over res.body",
      code: `
        const res = await fetch(url);
        for await (const chunk of res.body) {
          console.log(chunk);
        }
      `,
    },

    // -----------------------------------------------------------------
    // Optional chaining
    // -----------------------------------------------------------------
    {
      name: "optional chaining on body read — res?.json()",
      code: `
        const res = await fetch(url);
        await res?.json();
      `,
    },
    {
      name: "optional chaining on stream method — res.body?.cancel()",
      code: `
        const res = await fetch(url);
        await res.body?.cancel();
      `,
    },

    // -----------------------------------------------------------------
    // Handoff patterns (caller/callee takes responsibility)
    // -----------------------------------------------------------------
    {
      name: "response returned from async function",
      code: `
        async function doFetch() {
          return await fetch(url);
        }
      `,
    },
    {
      name: "response returned implicitly from arrow",
      code: `
        const doFetch = () => fetch(url);
      `,
    },
    {
      name: "response passed to another function",
      code: `
        const res = await fetch(url);
        processResponse(res);
      `,
    },
    {
      name: "response handed to Promise.all as element",
      code: `
        const results = await Promise.all([fetch(a), fetch(b)]);
      `,
    },
    {
      name: "response stored in an array literal",
      code: `
        const responses = [await fetch(url1), await fetch(url2)];
      `,
    },
    {
      name: "response spread into an array",
      code: `
        const all = [...prev, await fetch(url)];
      `,
    },
    {
      name: "response stored in an object literal",
      code: `
        const wrapped = { response: await fetch(url) };
      `,
    },
    {
      name: "response thrown",
      code: `
        throw await fetch(url);
      `,
    },
    {
      name: "response yielded from generator",
      code: `
        async function* stream() {
          yield await fetch(url);
        }
      `,
    },
    {
      name: "response selected via ternary and returned",
      code: `
        async function f() {
          return condition ? await fetch(url1) : await fetch(url2);
        }
      `,
    },
    {
      name: "response selected via logical && and returned",
      code: `
        async function f() {
          return maybe && await fetch(url);
        }
      `,
    },

    // -----------------------------------------------------------------
    // Direct chaining
    // -----------------------------------------------------------------
    {
      name: "chained: fetch().then(...)",
      code: `
        fetch(url).then(res => res.json());
      `,
    },
    {
      name: "chained on awaited: (await fetch()).json()",
      code: `
        const data = (await fetch(url)).json();
      `,
    },
    {
      name: "chained: .then cancels body explicitly",
      code: `
        fetch(url).then(res => {
          void res.body?.cancel();
        });
      `,
    },
    {
      name: "chained: .then returns response (handed off downstream)",
      code: `
        fetch(url).then(res => res).catch(() => undefined);
      `,
    },
    {
      name: "chained: .then callback is a named function (not analysed)",
      code: `
        fetch(url).then(processResponse).catch(() => undefined);
      `,
    },
    {
      name: "chained: .then callback consumes via block body",
      code: `
        fetch(url)
          .then(res => {
            const shard = res.headers.get("X-Shard");
            void res.body?.cancel();
            return shard;
          })
          .catch(() => undefined);
      `,
    },
    {
      name: "arrow-wrapped fetch returned from a function",
      code: `
        function makeFetcher() {
          return () => fetch(url);
        }
      `,
    },

    // -----------------------------------------------------------------
    // Member forms of global fetch
    // -----------------------------------------------------------------
    {
      name: "globalThis.fetch consumed",
      code: `
        const res = await globalThis.fetch(url);
        await res.json();
      `,
    },
    {
      name: "window.fetch consumed",
      code: `
        const res = await window.fetch(url);
        await res.text();
      `,
    },

    // -----------------------------------------------------------------
    // Control-flow consumption
    // -----------------------------------------------------------------
    {
      name: "body consumed in both branches of an if",
      code: `
        const res = await fetch(url);
        if (res.ok) {
          await res.json();
        } else {
          await res.text();
        }
      `,
    },

    // -----------------------------------------------------------------
    // Shadowed / imported fetch
    // -----------------------------------------------------------------
    {
      name: "locally-declared fetch function is not flagged",
      code: `
        function fetch(x) { return x; }
        fetch(url);
      `,
    },
    {
      name: "imported fetch is not flagged",
      code: `
        import fetch from "node-fetch-wrapper";
        fetch(url);
      `,
    },

    // -----------------------------------------------------------------
    // Destructuring that captures the body
    // -----------------------------------------------------------------
    {
      name: "destructuring { body } is allowed",
      code: `
        const { body } = await fetch(url);
        body.cancel();
      `,
    },
    {
      name: "destructuring with a rest element captures the body",
      code: `
        const { status, ...rest } = await fetch(url);
      `,
    },

    // -----------------------------------------------------------------
    // TypeScript type expressions (transparent wrappers around fetch)
    // -----------------------------------------------------------------
    {
      name: "TS `as` cast on awaited fetch, then consumed",
      code: `
        const res = (await fetch(url)) as Response;
        await res.json();
      `,
      languageOptions: tsLanguageOptions,
    },
    {
      name: "TS `satisfies` on awaited fetch, then consumed",
      code: `
        const res = (await fetch(url)) satisfies Response;
        await res.text();
      `,
      languageOptions: tsLanguageOptions,
    },
    {
      name: "TS non-null assertion on awaited fetch, then consumed",
      code: `
        const res = (await fetch(url))!;
        await res.arrayBuffer();
      `,
      languageOptions: tsLanguageOptions,
    },

    // -----------------------------------------------------------------
    // Async callbacks and conditional consumption inside .then
    // -----------------------------------------------------------------
    {
      name: "chained: .then async arrow callback consumes via await",
      code: `
        fetch(url).then(async res => await res.text());
      `,
    },
    {
      name: "chained: .then async block-body callback consumes via await",
      code: `
        fetch(url).then(async res => {
          const data = await res.json();
          return data;
        });
      `,
    },
    {
      name: "chained: .then callback consumes via ternary (both branches)",
      code: `
        fetch(url).then(res => res.ok ? res.json() : res.text());
      `,
    },
  ],

  invalid: [
    // -----------------------------------------------------------------
    // Fire-and-forget
    // -----------------------------------------------------------------
    {
      name: "bare fetch() statement",
      code: `fetch(url);`,
      errors: unconsumed,
    },
    {
      name: "bare `await fetch()` statement",
      code: `await fetch(url);`,
      errors: unconsumed,
    },

    // -----------------------------------------------------------------
    // Response kept but body never touched
    // -----------------------------------------------------------------
    {
      name: "only reads status",
      code: `
        const res = await fetch(url);
        console.log(res.status);
      `,
      errors: unconsumed,
    },
    {
      name: "only reads ok",
      code: `
        const res = await fetch(url);
        if (res.ok) {
          console.log("success");
        }
      `,
      errors: unconsumed,
    },
    {
      name: "only reads headers",
      code: `
        const res = await fetch(url);
        const contentType = res.headers.get("Content-Type");
      `,
      errors: unconsumed,
    },

    // -----------------------------------------------------------------
    // Member forms of global fetch
    // -----------------------------------------------------------------
    {
      name: "globalThis.fetch not consumed",
      code: `await globalThis.fetch(url);`,
      errors: unconsumed,
    },
    {
      name: "window.fetch not consumed",
      code: `await window.fetch(url);`,
      errors: unconsumed,
    },
    {
      name: "self.fetch not consumed",
      code: `await self.fetch(url);`,
      errors: unconsumed,
    },

    // -----------------------------------------------------------------
    // Assignment
    // -----------------------------------------------------------------
    {
      name: "assigned via =, never consumed",
      code: `
        let res;
        res = await fetch(url);
        console.log(res.status);
      `,
      errors: unconsumed,
    },

    // -----------------------------------------------------------------
    // Destructuring without body
    // -----------------------------------------------------------------
    {
      name: "destructuring without body leaks the stream",
      code: `
        const { status } = await fetch(url);
      `,
      errors: unconsumed,
    },

    // -----------------------------------------------------------------
    // Chained .then callback that does not consume
    // -----------------------------------------------------------------
    {
      name: "chained: .then reads only a header",
      code: `
        fetch(url, { signal })
          .then(response => response.headers.get("X-Shard") || undefined)
          .catch(() => undefined);
      `,
      errors: unconsumed,
    },
    {
      name: "arrow-wrapped fetch passed to a factory function",
      code: `
        factory(() => fetch(url));
      `,
      errors: unconsumed,
    },
    {
      name: "chained: .then only reads response.ok",
      code: `
        fetch(url)
          .then(response => setAvailable(response.ok))
          .catch(() => setAvailable(false));
      `,
      errors: unconsumed,
    },
    {
      name: "chained: .then callback throws without consuming the body",
      code: `
        fetch(url).then(res => {
          throw new Error("status: " + res.status);
        });
      `,
      errors: unconsumed,
    },
    {
      name: "chained: .then uses a parameterless callback",
      code: `
        fetch(url).then(() => console.log("done"));
      `,
      errors: unconsumed,
    },
    {
      name: "chained: .then callback only logs the status",
      code: `
        fetch(url).then(res => console.log(res.status));
      `,
      errors: unconsumed,
    },
    {
      name: "chained: .then callback returns a transformed shape (body lost)",
      code: `
        fetch(url).then(res => ({ ok: res.ok, status: res.status }));
      `,
      errors: unconsumed,
    },
    {
      name: "chained: only .catch attached (success path leaks the body)",
      code: `
        fetch(url).catch(() => undefined);
      `,
      errors: unconsumed,
    },
    {
      name: "chained: only .finally attached",
      code: `
        fetch(url).finally(() => cleanup());
      `,
      errors: unconsumed,
    },
    {
      name: "awaited fetch accessed only for status",
      code: `
        const status = (await fetch(url)).status;
      `,
      errors: unconsumed,
    },
    {
      name: "arrow-wrapped fetch passed to new (constructor argument)",
      code: `
        new Observable(() => fetch(url));
      `,
      errors: unconsumed,
    },
    {
      name: "destructuring parameter in .then callback leaks the body",
      code: `
        fetch(url).then(({ ok }) => setAvailable(ok));
      `,
      errors: unconsumed,
    },

    // -----------------------------------------------------------------
    // TypeScript type expressions must not hide an unconsumed fetch
    // -----------------------------------------------------------------
    {
      name: "TS `as` cast on awaited fetch, only status read",
      code: `
        const res = (await fetch(url)) as Response;
        console.log(res.status);
      `,
      languageOptions: tsLanguageOptions,
      errors: unconsumed,
    },
    {
      name: "TS `satisfies` on awaited fetch, only status read",
      code: `
        const res = (await fetch(url)) satisfies Response;
        console.log(res.status);
      `,
      languageOptions: tsLanguageOptions,
      errors: unconsumed,
    },
    {
      name: "TS non-null assertion on awaited fetch, only status read",
      code: `
        const res = (await fetch(url))!;
        console.log(res.status);
      `,
      languageOptions: tsLanguageOptions,
      errors: unconsumed,
    },
  ],
});

console.log("All tests passed!");
