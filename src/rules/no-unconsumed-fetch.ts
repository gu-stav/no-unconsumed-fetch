import type { Rule, Scope } from "eslint";
import type {
  ArrowFunctionExpression,
  CallExpression,
  FunctionExpression,
  Identifier,
  MemberExpression,
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

const BODY_READ_METHODS = new Set(["json", "text", "arrayBuffer", "blob", "formData"]);

const STREAM_CONSUME_METHODS = new Set(["cancel", "getReader", "pipeTo", "pipeThrough"]);

const GLOBAL_OBJECTS = new Set(["globalThis", "window", "self"]);

type FunctionLike = ArrowFunctionExpression | FunctionExpression;

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

/**
 * Returns the static string name of a member access (both `x.foo` and
 * `x["foo"]`), or null when the name cannot be determined statically
 * (computed numeric/template/expression keys).
 */
function getStaticMemberName(member: MemberExpression): string | null {
  if (!member.computed) {
    return member.property.type === "Identifier" ? member.property.name : null;
  }
  if (member.property.type === "Literal" && typeof member.property.value === "string") {
    return member.property.value;
  }
  return null;
}

function resolveToVariable(scope: Scope.Scope, identifier: Identifier): Scope.Variable | null {
  for (let current: Scope.Scope | null = scope; current; current = current.upper) {
    const ref = current.references.find((r) => r.identifier === identifier);
    if (ref) return ref.resolved;
  }
  return null;
}

function findVariable(scope: Scope.Scope, name: string): Scope.Variable | null {
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
    // TypeScript-only wrappers that do not affect runtime value:
    //   `fetch(url) as Response`    (TSAsExpression)
    //   `fetch(url) satisfies ...`  (TSSatisfiesExpression)
    //   `fetch(url)!`               (TSNonNullExpression)
    //   `<Response>fetch(url)`      (TSTypeAssertion, legacy angle-bracket form)
    const parentType = (parent as { type: string }).type;
    if (
      (parentType === "TSAsExpression" ||
        parentType === "TSSatisfiesExpression" ||
        parentType === "TSNonNullExpression" ||
        parentType === "TSTypeAssertion") &&
      (parent as { expression?: unknown }).expression === current
    ) {
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
 * array/object, etc.). Transparent wrappers like `await`, `?.`, and TS type
 * expressions are walked through first.
 */
function isValueHandedOff(node: Rule.Node): boolean {
  const outer = resolveOuterExpression(node);
  const parent = outer.parent;
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
      return parent.arguments.some((arg) => arg === outer);

    case "ArrowFunctionExpression":
      return parent.body === outer;

    case "ConditionalExpression":
      return (
        (parent.consequent === outer || parent.alternate === outer) && isValueHandedOff(parent)
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
 * Walks past transparent wrappers (`as`, `!`, `satisfies`) before inspecting.
 * Accepts both dot access (`res.json()`) and bracket access with a string
 * literal (`res["json"]()`).
 */
function isConsumingUsage(identifier: Rule.Node): boolean {
  const outer = resolveOuterExpression(identifier);
  const parent = outer.parent;
  if (!parent || parent.type !== "MemberExpression" || parent.object !== outer) {
    return false;
  }

  const propName = getStaticMemberName(parent);
  if (propName === null) return false;

  const grandparent = parent.parent;

  // res.json(), res.text(), ...
  if (BODY_READ_METHODS.has(propName)) {
    return isCallee(parent, grandparent);
  }

  // res.body.* / res.body (for await)
  if (propName === "body") {
    if (!grandparent) return false;

    // res.body.cancel(), res.body.getReader(), res.body.pipeTo(...)
    if (grandparent.type === "MemberExpression" && grandparent.object === parent) {
      const methodName = getStaticMemberName(grandparent);
      if (methodName !== null && STREAM_CONSUME_METHODS.has(methodName)) {
        return isCallee(grandparent, grandparent.parent);
      }
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

function isCallee(callee: Rule.Node, maybeCall: Rule.Node | undefined): boolean {
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
 * If `identifier` is the source of a simple aliasing assignment
 * (`const target = identifier` or `target = identifier`), return the target
 * binding name. Otherwise null.
 */
function getAliasTargetName(identifier: Rule.Node): string | null {
  const outer = resolveOuterExpression(identifier);
  const parent = outer.parent;
  if (!parent) return null;

  if (
    parent.type === "VariableDeclarator" &&
    parent.init === outer &&
    parent.id.type === "Identifier"
  ) {
    return parent.id.name;
  }

  if (
    parent.type === "AssignmentExpression" &&
    parent.operator === "=" &&
    parent.right === outer &&
    parent.left.type === "Identifier"
  ) {
    return parent.left.name;
  }

  return null;
}

/**
 * Walk references of a variable that was assigned a fetch result. Returns
 * true if any reference indicates the body was consumed or the response was
 * handed off to someone else. Follows simple aliases (`const r2 = res`) so
 * that consumption of the alias counts as consumption of the original.
 */
function isVariableConsumed(
  scope: Scope.Scope,
  name: string,
  visited: Set<string> = new Set(),
): boolean {
  if (visited.has(name)) return false;
  visited.add(name);

  const variable = findVariable(scope, name);
  if (!variable) return false;

  for (const ref of variable.references) {
    const id = ref.identifier as Rule.Node;
    if (!id.parent) continue;
    // Ignore the write reference (the declaration/assignment itself).
    if (ref.writeExpr) continue;
    if (isConsumingUsage(id) || isValueHandedOff(id)) return true;

    const aliasName = getAliasTargetName(id);
    if (aliasName !== null && isVariableConsumed(scope, aliasName, visited)) {
      return true;
    }
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

/**
 * True when the callback body consumes or hands off its first parameter
 * (the response). Used to validate `.then(cb)` chains.
 *
 * If the callback has no first parameter, we cannot reason about consumption
 * and conservatively treat it as not consuming. Chains like
 * `fetch(url).then(() => {}).catch(...)` therefore leak.
 */
function isThenCallbackConsuming(scope: Scope.Scope, cb: FunctionLike): boolean {
  const firstParam = cb.params[0];
  if (!firstParam || firstParam.type !== "Identifier") {
    // Destructuring parameter — if it picks up `body` or uses a rest element,
    // consider it consumed. Otherwise the response body is unreachable.
    if (firstParam && firstParam.type === "ObjectPattern") {
      return destructuresBody(firstParam);
    }
    return false;
  }
  return isVariableConsumed(scope, firstParam.name);
}

/**
 * For a `fetch(...).then(...)` (or chained) expression, find the `.then`
 * CallExpression that immediately consumes the fetch. Returns the call node,
 * or null if `fetch` is not followed by `.then(...)`.
 */
function findThenCall(memberExpressionChild: Rule.Node): CallExpression | null {
  const parent = memberExpressionChild.parent;
  if (
    !parent ||
    parent.type !== "MemberExpression" ||
    parent.object !== memberExpressionChild ||
    parent.computed ||
    parent.property.type !== "Identifier" ||
    parent.property.name !== "then"
  ) {
    return null;
  }
  const call = parent.parent;
  if (!call || call.type !== "CallExpression" || call.callee !== parent) return null;
  return call;
}

// ---------------------------------------------------------------------------
// Rule definition
// ---------------------------------------------------------------------------

const rule: Rule.RuleModule = {
  meta: {
    type: "problem",
    docs: {
      description: "Disallow fetch() calls whose response body is not consumed or cancelled",
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

        // 2) Arrow-wrapped fetch passed as a callback to another function.
        //    e.g. `defer(() => fetch(url))`. The external consumer usually
        //    does not drain the body — this is a common leak pattern.
        //    Exception: if the arrow body chains a consuming `.then`, fetch's
        //    parent is a MemberExpression instead, so we never reach here.
        if (parent.type === "ArrowFunctionExpression" && parent.body === resolved) {
          const arrowParent = parent.parent;
          if (
            arrowParent &&
            (arrowParent.type === "CallExpression" || arrowParent.type === "NewExpression") &&
            arrowParent.arguments.includes(parent)
          ) {
            context.report({ node, messageId: "unconsumed" });
            return;
          }
          // Otherwise: arrow is stored/returned as a factory. Trust the caller.
          return;
        }

        // 3) Handed off (returned, thrown, passed as arg, stored in collection).
        if (isValueHandedOff(resolved)) return;

        // 4) Assigned to a variable binding.
        if (parent.type === "VariableDeclarator" && parent.init === resolved) {
          handleBinding(parent, node);
          return;
        }

        // 5) Assigned via assignment expression: `res = await fetch(url);`
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

        // 6) Direct chaining: fetch(url).then(...), (await fetch(url)).json()
        if (parent.type === "MemberExpression" && parent.object === resolved) {
          const thenCall = findThenCall(resolved);
          if (thenCall) {
            const cb = thenCall.arguments[0];
            if (cb && (cb.type === "ArrowFunctionExpression" || cb.type === "FunctionExpression")) {
              const cbScope = sourceCode.getScope(cb as unknown as Rule.Node);
              if (!isThenCallbackConsuming(cbScope, cb)) {
                context.report({ node, messageId: "unconsumed" });
              }
              return;
            }
            // `.then(cb)` where cb is not an inline function (named identifier,
            // null, missing, etc.) — we can't verify the body is consumed, so
            // flag it. Callers can silence by wrapping: `.then(r => helper(r))`.
            context.report({ node, messageId: "unconsumed" });
            return;
          }
          // Chained with something other than `.then` (e.g. `.catch`, `.status`
          // on an awaited fetch). Without a `.then` in the chain the body is
          // never consumed. For awaited forms like `(await fetch(url)).json()`
          // or `(await fetch(url))["json"]()` the member itself is the
          // consumer — check it directly.
          if (isConsumingUsage(resolved)) {
            return;
          }
          context.report({ node, messageId: "unconsumed" });
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
