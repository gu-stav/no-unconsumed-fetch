import type { ESLint, Linter } from "eslint";
import noUnconsumedFetch = require("./rules/no-unconsumed-fetch");

const meta = {
  name: "no-unconsumed-fetch",
  version: "1.0.0-alpha.0",
} as const;

const rules = {
  "no-unconsumed-fetch": noUnconsumedFetch,
} satisfies ESLint.Plugin["rules"];

const plugin: ESLint.Plugin & { configs: Record<string, Linter.Config> } = {
  meta,
  rules,
  configs: Object.create(null),
};

plugin.configs.recommended = {
  name: "no-unconsumed-fetch/recommended",
  plugins: {
    "no-unconsumed-fetch": plugin,
  },
  rules: {
    "no-unconsumed-fetch/no-unconsumed-fetch": "error",
  },
};

export = plugin;
