import type { Rule, Scope } from "eslint";
import type {
  CallExpression,
  Identifier,
  ObjectPattern,
  Property,
  VariableDeclarator,
} from "estree";

/**
 * ESLint rule: no-unconsumed-fetch
 *
 * Flags fetch() calls whose Response body is never consumed or cancelled.
 *
 * Consumed = one of the body-reading methods (json/text/arrayBuffer/blob/
 * formData) is called on the response, or the stream is handed off (body.cancel,
 * body.getReader, body.pipeTo/pipeThrough, for-await on body). The rule also
 * accepts patterns where responsibility is transferred (returned, thrown,
 * yielded, passed as a callee argument, stored in an array/object).
 */

const BODY_READ_METHODS = new Set([
  "json",
  "text",
  "arrayBuffer",
  "blob",
  "formData",
]);

const STREAM_CONSUME_METHODS = new Set([
  "cancel",
  "getReader",
  "pipeTo",
  "pipeThrough",
]);

const GLOBAL_OBJECTS = new Set(["globalThis", "window", "self"]);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * True when `node` is a call to the global `fetch`, including member forms
 * (`globalThis.fetch`, `window.fetch`, `self.fetch`). A locally-declared or
 * imported `fetch` is not flagged.
 */
function isGlobalFetchCall(node: CallExpression, scope: Scope.Scope): boolean {
  const callee = node.callee;

  if (callee.type === "Identifier" && callee.name === "fetch") {
    // Make sure this resolves to a global, not an imported/declared binding.
    const variable = resolveToVariable(scope, callee);
    return !variable || isGlobalVariable(variable);
  }

  if (
    callee.type === "MemberExpression" &&
    !callee.computed &&
    callee.property.type === "Identifier" &&
    callee.property.name === "fetch" &&
    callee.object.type === "Identifier" &&
    GLOBAL_OBJECTS.has(callee.object.name)
  ) {
    return true;
  }

  return false;
}

function resolveToVariable(
  scope: Scope.Scope,
  identifier: Identifier,
): Scope.Variable | null {
  for (let current: Scope.Scope | null = scope; current; current = current.upper) {
    const ref = current.references.find((r) => r.identifier === identifier);
    if (ref) return ref.resolved;
  }
  return null;
}

function findVariable(
  scope: Scope.Scope,
  name: string,
): Scope.Variable | null {
  for (let current: Scope.Scope | null = scope; current; current = current.upper) {
    const variable = current.set.get(name);
    if (variable) return variable;
  }
  return null;
}

/** Implicit globals have no `defs` entries; real declarations always do. */
function isGlobalVariable(variable: Scope.Variable): boolean {
  return variable.defs.length === 0;
}

/** Walk outward through wrappers that pass the value along unchanged. */
function resolveOuterExpression(node: Rule.Node): Rule.Node {
  let current = node;
  while (current.parent) {
    const parent = current.parent;
    if (parent.type === "AwaitExpression" && parent.argument === current) {
      current = parent;
      continue;
    }
    if (parent.type === "ChainExpression" && parent.expression === current) {
      current = parent;
      continue;
    }
    break;
  }
  return current;
}

/**
 * True when `node` sits in a position whose value is handed to someone else
 * (returned, thrown, yielded, passed as an argument, stored in an
 * array/object, used as an arrow body, etc.).
 */
function isValueHandedOff(node: Rule.Node): boolean {
  const parent = node.parent;
  if (!parent) return false;

  switch (parent.type) {
    case "ReturnStatement":
    case "YieldExpression":
    case "ThrowStatement":
    case "ArrayExpression":
    case "SpreadElement":
      return true;

    case "Property":
      // `{ foo: <expr> }` — consumed, but `{ <expr>: value }` is a computed key
      // on an object literal which also counts as handing the value off.
      return true;

    case "CallExpression":
    case "NewExpression":
      return parent.arguments.includes(node as CallExpression);

    case "ArrowFunctionExpression":
      return parent.body === node;

    case "ConditionalExpression":
      return (
        (parent.consequent === node || parent.alternate === node) &&
        isValueHandedOff(parent)
      );

    case "LogicalExpression":
    case "SequenceExpression":
      return isValueHandedOff(parent);

    default:
      return false;
  }
}

/**
 * True when the identifier is used in a way that consumes the response body.
 */
function isConsumingUsage(identifier: Rule.Node): boolean {
  const parent = identifier.parent;
  if (!parent || parent.type !== "MemberExpression" || parent.object !== identifier) {
    return false;
  }
  if (parent.computed || parent.property.type !== "Identifier") return false;

  const propName = parent.property.name;
  const grandparent = parent.parent;

  // res.json(), res.text(), ...
  if (BODY_READ_METHODS.has(propName)) {
    return isCallee(parent, grandparent);
  }

  // res.body.* / res.body (for await)
  if (propName === "body") {
    if (!grandparent) return false;

    // res.body.cancel(), res.body.getReader(), res.body.pipeTo(...)
    if (
      grandparent.type === "MemberExpression" &&
      grandparent.object === parent &&
      !grandparent.computed &&
      grandparent.property.type === "Identifier" &&
      STREAM_CONSUME_METHODS.has(grandparent.property.name)
    ) {
      return isCallee(grandparent, grandparent.parent);
    }

    // for await (const x of res.body) {...}
    if (
      grandparent.type === "ForOfStatement" &&
      grandparent.await === true &&
      grandparent.right === parent
    ) {
      return true;
    }
  }

  return false;
}

function isCallee(
  callee: Rule.Node,
  maybeCall: Rule.Node | undefined,
): boolean {
  if (!maybeCall) return false;
  if (maybeCall.type === "CallExpression" && maybeCall.callee === callee) {
    return true;
  }
  // res?.json() / res.body?.cancel() — the call is wrapped in a ChainExpression.
  if (maybeCall.type === "ChainExpression") {
    return isCallee(callee, maybeCall.parent);
  }
  return false;
}

/**
 * Walk references of a variable that was assigned a fetch result. Returns
 * true if any reference indicates the body was consumed or the response was
 * handed off to someone else.
 */
function isVariableConsumed(
  scope: Scope.Scope,
  name: string,
): boolean {
  const variable = findVariable(scope, name);
  if (!variable) return false;

  for (const ref of variable.references) {
    const id = ref.identifier as Rule.Node;
    if (!id.parent) continue;
    // Ignore the write reference (the declaration/assignment itself).
    if (ref.writeExpr) continue;
    if (isConsumingUsage(id) || isValueHandedOff(id)) return true;
  }
  return false;
}

/**
 * For `const { body } = await fetch(url)` patterns: treat as consumed if
 * `body` is extracted. Nothing else on a Response holds the stream open.
 */
function destructuresBody(pattern: ObjectPattern): boolean {
  for (const prop of pattern.properties) {
    if (prop.type === "RestElement") return true; // `...rest` captures body
    const p = prop as Property;
    if (p.computed) continue;
    const key = p.key;
    const name =
      key.type === "Identifier"
        ? key.name
        : key.type === "Literal" && typeof key.value === "string"
          ? key.value
          : null;
    if (name === "body") return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// Rule definition
// ---------------------------------------------------------------------------

const rule: Rule.RuleModule = {
  meta: {
    type: "problem",
    docs: {
      description:
        "Disallow fetch() calls whose response body is not consumed or cancelled",
      recommended: true,
      url: "https://github.com/gu-stav/no-unconsumed-fetch#readme",
    },
    schema: [],
    defaultOptions: [],
    messages: {
      unconsumed:
        "The response body of this fetch() call is not consumed. " +
        "Call res.json(), res.text(), res.body.cancel(), or another " +
        "body-consuming method to avoid connection pool exhaustion and " +
        "resource leaks.",
    },
  },

  create(context) {
    const { sourceCode } = context;

    return {
      CallExpression(node) {
        if (!isGlobalFetchCall(node, sourceCode.getScope(node))) return;

        const resolved = resolveOuterExpression(node as Rule.Node);
        const parent = resolved.parent;

        // 1) Bare expression statement: `fetch(url);` or `await fetch(url);`
        if (!parent || parent.type === "ExpressionStatement") {
          context.report({ node, messageId: "unconsumed" });
          return;
        }

        // 2) Handed off (returned, thrown, passed as arg, stored in collection).
        if (isValueHandedOff(resolved)) return;

        // 3) Assigned to a variable binding.
        if (parent.type === "VariableDeclarator" && parent.init === resolved) {
          handleBinding(parent, node);
          return;
        }

        // 4) Assigned via assignment expression: `res = await fetch(url);`
        if (
          parent.type === "AssignmentExpression" &&
          parent.operator === "=" &&
          parent.right === resolved
        ) {
          if (parent.left.type === "Identifier") {
            const scope = sourceCode.getScope(parent);
            if (!isVariableConsumed(scope, parent.left.name)) {
              context.report({ node, messageId: "unconsumed" });
            }
            return;
          }
          // Member or destructured assignment: assume the callee/property owner
          // takes responsibility.
          return;
        }

        // 5) Direct chaining: fetch(url).then(...), (await fetch(url)).json()
        if (parent.type === "MemberExpression" && parent.object === resolved) {
          return;
        }

        // Anything else: we don't recognise the pattern — don't flag by default
        // (prefer false negatives over false positives).
      },
    };

    function handleBinding(declarator: VariableDeclarator, fetchNode: CallExpression) {
      const target = declarator.id;

      if (target.type === "Identifier") {
        const scope = sourceCode.getScope(declarator);
        if (!isVariableConsumed(scope, target.name)) {
          context.report({ node: fetchNode, messageId: "unconsumed" });
        }
        return;
      }

      if (target.type === "ObjectPattern") {
        if (destructuresBody(target)) return;
        context.report({ node: fetchNode, messageId: "unconsumed" });
        return;
      }

      // ArrayPattern or anything else: a Response isn't iterable, so this is
      // almost certainly a bug — but leave it alone rather than double-flagging.
    }
  },
};

export = rule;
