/**
 * Generator entrypoint. Read a toolkit catalog, infer its dependencies, write a graph.
 *
 * How we run it:
 *   - The path to a toolkit's catalog JSON is passed as a CLI ARGUMENT, e.g.
 *     `node --import tsx src/generate.ts path/to/catalog.json`. We append it as the last
 *     argument, so reading the final argv entry works whatever else your command carries.
 *   - Write your graph to `dependency_graph.json` in the working directory.
 *   - For LLM access, point the OpenAI SDK at `$LITMUS_API_URL/proxy/openai/v1` with the
 *     key in LITMUS_ASSESSMENT_TOKEN (set from your assessment page's AI credentials; the
 *     same are provided when we run your generator). Use an OpenRouter model id such as
 *     `openai/gpt-4o`.
 */
import { readFileSync, writeFileSync } from "fs";
import OpenAI from "openai";

type Tool = Record<string, any>;
interface Node {
  id: string;
  service?: string;
}
interface Edge {
  from: string;
  to: string;
  label?: string;
}
interface Graph {
  nodes: Node[];
  edges: Edge[];
}

interface ProducedField {
  name: string;
  normalized: string;
  parent: string;
  description: string;
}

interface CompactTool {
  slug: string;
  name: string;
  description: string;
  service: string;
  deprecated: boolean;
  inputDesc: string;
  identityInputs: { name: string; normalized: string; required: boolean; description: string }[];
  produced: ProducedField[];
  tokens: Set<string>;
}

interface ScoredEdge extends Edge {
  from: string;
  to: string;
  label: string;
  score: number;
}

const CATALOG_PATH = process.argv.length > 2 ? process.argv[process.argv.length - 1] : undefined;
const OUT_PATH = "dependency_graph.json";
const VIZ_PATH = "visualization.html";
const GRAPH_MD_PATH = "graph.md";

const SKIP_INPUTS = new Set([
  "owner",
  "repo",
  "org",
  "organization",
  "body",
  "title",
  "message",
  "page",
  "per_page",
  "perpage",
  "sort",
  "direction",
  "state",
  "labels",
  "description",
  "clientmutationid",
  "client_mutation_id",
  "commit_title",
  "commit_message",
  "merge_method",
  "color",
  "query",
  "q",
  "order",
  "since",
  "creator",
  "assignee",
  "assignees",
  "mentioned",
  "type",
  "filter",
  "affiliation",
  "visibility",
  "format",
  "status",
  "conclusion",
  "event",
  "draft",
  "locked",
  "archived",
  "text",
  "content",
  "name",
  "email",
  "password",
  "reason",
  "media_type",
  "base",
  "head",
  "maintainer_can_modify",
]);

const SKIP_WALK = new Set([
  "user",
  "owner",
  "creator",
  "author",
  "organization",
  "assignee",
  "assignees",
  "closed_by",
  "merged_by",
  "actor",
  "app",
  "installation",
  "permissions",
  "license",
  "url",
  "html_url",
  "followers_url",
  "following_url",
  "gists_url",
  "starred_url",
  "subscriptions_url",
  "organizations_url",
  "repos_url",
  "events_url",
  "received_events_url",
  "avatar_url",
  "error",
  "successful",
]);

const GENERIC_TOKENS = new Set([
  "github",
  "get",
  "list",
  "create",
  "update",
  "delete",
  "remove",
  "add",
  "set",
  "the",
  "for",
  "and",
  "with",
  "from",
  "action",
  "tool",
  "data",
  "type",
  "string",
  "integer",
  "object",
  "array",
  "authenticated",
  "using",
  "this",
  "that",
  "into",
  "your",
  "their",
  "when",
  "need",
  "must",
  "response",
  "request",
  "schema",
  "api",
  "rest",
  "graphql",
  "number",
  "id",
  "ids",
  "sha",
  "ref",
  "slug",
  "node",
  "url",
  "name",
  "new",
  "an",
  "a",
  "of",
  "to",
  "in",
  "on",
  "by",
  "or",
  "as",
  "check",
  "if",
  "has",
  "been",
  "associated",
  "selected",
  "assigned",
]);

const CONTEXT_TOKENS = new Set([
  ...GENERIC_TOKENS,
  "repository",
  "repo",
  "org",
  "organization",
  "user",
  "branch",
  "head",
  "base",
  "open",
  "close",
  "reopen",
  "find",
  "search",
  "fetch",
  "read",
  "a",
]);

const HINT_TAG = /(hint|graphql|important|destructive|idempotent|openworld)/i;
const IDENT_RE =
  /^(?:[a-z0-9]+_)*(?:id|ids|number|sha|ref|node_id|nodeid)$|^(?:sha|ref|node_id)$/;
const PRODUCER_LIST_RE = /(^|_)(LIST|GET|FIND|SEARCH|FETCH|READ)(_|$)/i;
const PRODUCER_CREATE_RE = /(^|_)(CREATE|OPEN|FORK|UPLOAD|PUSH)(_|$)/i;
const PRODUCER_WEAK_RE = /(^|_)(UPDATE|DELETE|REMOVE|ABORT|CANCEL|DISMISS|LOCK|UNLOCK|MERGE|REPLACE)(_|$)/i;

function loadCatalog(): Tool[] {
  if (!CATALOG_PATH) {
    throw new Error("pass the toolkit catalog path as the first argument");
  }
  const data = JSON.parse(readFileSync(CATALOG_PATH, "utf-8"));
  return Array.isArray(data) ? data : (data.tools ?? data.items ?? []);
}

function slugOf(tool: Tool): string | undefined {
  return tool.slug ?? tool.name ?? tool.function?.name;
}

function normalize(name: string): string {
  return String(name)
    .replace(/([a-z0-9])([A-Z])/g, "$1_$2")
    .replace(/([A-Z]+)([A-Z][a-z])/g, "$1_$2")
    .replace(/[^a-zA-Z0-9]+/g, "_")
    .replace(/_+/g, "_")
    .replace(/^_|_$/g, "")
    .toLowerCase();
}

function tokenize(text: string): string[] {
  const parts = normalize(text)
    .split("_")
    .filter((w) => w.length > 1 && !GENERIC_TOKENS.has(w) && !/^\d+$/.test(w));
  const out: string[] = [];
  for (const w of parts) {
    out.push(w);
    if (w.endsWith("ies") && w.length > 4) out.push(w.slice(0, -3) + "y");
    else if (w.endsWith("ses") && w.length > 4) out.push(w.slice(0, -2));
    else if (w.endsWith("s") && !w.endsWith("ss") && w.length > 3) out.push(w.slice(0, -1));
  }
  return out;
}

function tokenSet(text: string): Set<string> {
  return new Set(tokenize(text));
}

function overlap(a: Set<string>, b: Set<string>): number {
  let n = 0;
  for (const t of a) if (b.has(t)) n++;
  return n;
}

function serviceOf(tool: Tool): string {
  const tags: string[] = Array.isArray(tool.tags) ? tool.tags : [];
  const domain = tags.find((t) => t && !HINT_TAG.test(String(t)));
  if (domain) return String(domain).toLowerCase();
  const slug = slugOf(tool) ?? "";
  const parts = slug.split("_").filter((p) => p && p !== "GITHUB" && p.length > 2);
  return (parts[1] ?? parts[0] ?? "other").toLowerCase();
}

function isIdentName(normalized: string): boolean {
  if (SKIP_INPUTS.has(normalized) || normalized === "client_mutation_id") return false;
  return IDENT_RE.test(normalized);
}

function isIdentityInput(name: string, description: string, required: boolean): boolean {
  const n = normalize(name);
  if (SKIP_INPUTS.has(n)) return false;
  if (isIdentName(n)) return true;
  if (required && /\b(id|number|sha)\b/i.test(description) && /list|get|return|obtain|from the/i.test(description)) {
    return true;
  }
  return false;
}

function resolveSchema(schema: any, defs: Record<string, any>, stack: Set<string>): any {
  if (!schema || typeof schema !== "object") return schema;
  if (schema.$ref && typeof schema.$ref === "string") {
    const m = schema.$ref.match(/#\/\$defs\/(.+)$/);
    if (!m) return schema;
    const key = m[1];
    if (stack.has(key)) return {};
    stack.add(key);
    const resolved = defs[key];
    return resolveSchema(resolved, defs, stack);
  }
  return schema;
}

function collectProduced(outputParameters: any): ProducedField[] {
  if (!outputParameters || typeof outputParameters !== "object") return [];
  const defs = outputParameters.$defs ?? {};
  const out: ProducedField[] = [];
  const seen = new Set<string>();

  const walk = (schema: any, parent: string, depth: number, refStack: Set<string>) => {
    if (!schema || depth > 4) return;
    const resolved = resolveSchema(schema, defs, new Set(refStack));
    if (!resolved || typeof resolved !== "object") return;

    const title = String(resolved.title ?? parent ?? "");
    if (resolved.items) walk(resolved.items, title || parent, depth, refStack);

    const anyOf = resolved.anyOf ?? resolved.oneOf;
    if (Array.isArray(anyOf)) {
      for (const branch of anyOf) walk(branch, parent, depth, refStack);
    }

    const props = resolved.properties;
    if (!props || typeof props !== "object") return;
    for (const [key, sub] of Object.entries(props) as [string, any][]) {
      const norm = normalize(key);
      const desc = String(sub?.description ?? "");
      const parentLabel = title || parent || key;
      if (isIdentName(norm) || norm === "number" || norm === "id") {
        const sig = `${norm}|${normalize(parentLabel)}|${depth}`;
        if (!seen.has(sig)) {
          seen.add(sig);
          out.push({
            name: key,
            normalized: norm,
            parent: parentLabel,
            description: desc,
          });
        }
      }
      if (SKIP_WALK.has(norm) || SKIP_WALK.has(normalize(parentLabel))) continue;
      const nextStack = new Set(refStack);
      if (sub?.$ref) {
        const m = String(sub.$ref).match(/#\/\$defs\/(.+)$/);
        if (m && nextStack.has(m[1])) continue;
        if (m) nextStack.add(m[1]);
      }
      walk(sub, key, depth + 1, nextStack);
    }
  };

  const data = outputParameters.properties?.data ?? outputParameters;
  walk(data, "data", 0, new Set());
  return out;
}

function compactTool(tool: Tool): CompactTool | undefined {
  const slug = slugOf(tool);
  if (!slug) return undefined;
  const inputs = tool.inputParameters?.properties ?? tool.parameters?.properties ?? {};
  const required: string[] = tool.inputParameters?.required ?? tool.parameters?.required ?? [];
  const identityInputs = Object.entries(inputs)
    .filter(([name, schema]: [string, any]) =>
      isIdentityInput(name, String(schema?.description ?? ""), required.includes(name)),
    )
    .map(([name, schema]: [string, any]) => ({
      name,
      normalized: normalize(name),
      required: required.includes(name),
      description: String(schema?.description ?? ""),
    }));

  const produced = collectProduced(tool.outputParameters);
  const service = serviceOf(tool);
  const name = String(tool.name ?? slug);
  const description = String(tool.description ?? "");
  const inputDesc = String(tool.inputParameters?.description ?? "");
  const tagText = Array.isArray(tool.tags) ? tool.tags.join(" ") : "";
  const producedCtx = produced.map((p) => `${p.name} ${p.parent}`).join(" ");
  return {
    slug,
    name,
    description,
    service,
    deprecated: !!tool.isDeprecated,
    inputDesc,
    identityInputs,
    produced,
    tokens: tokenSet(`${slug} ${name} ${service} ${tagText} ${producedCtx}`),
  };
}

function fieldStem(field: { normalized: string }): string {
  return field.normalized.replace(/_(id|ids|number|sha|ref|node_id)$/g, "");
}

function fieldTokens(field: { name: string; normalized: string; description: string }): Set<string> {
  const stem = fieldStem(field);
  const out = tokenSet(stem === field.normalized ? stem : stem);
  if (field.normalized.includes("pull")) {
    out.add("pull");
    out.add("pr");
  }
  if (field.normalized.includes("issue")) out.add("issue");
  return out;
}

function producedTokens(p: ProducedField): Set<string> {
  return tokenSet(`${p.normalized} ${p.parent} ${p.description}`);
}

function descriptionMentions(text: string, producer: CompactTool): boolean {
  const hay = text.toLowerCase();
  if (hay.includes(producer.slug.toLowerCase())) return true;
  const name = producer.name.toLowerCase();
  if (name.length >= 16 && hay.includes(name)) return true;
  const sig = tokenize(producer.name).filter((t) => t.length > 3);
  if (sig.length >= 3) {
    const hits = sig.filter((t) => hay.includes(t)).length;
    if (hits >= sig.length - 1 && hits >= 3) return true;
  }
  return false;
}

function aliasMatch(fieldNorm: string, produced: ProducedField): boolean {
  if (fieldNorm === produced.normalized) return true;
  if (fieldNorm.endsWith("_number") && produced.normalized === "number") {
  const context = normalize(produced.parent);
  const stem = fieldNorm.replace(/_number$/, "");
  return context.includes(stem);
}
  if ((fieldNorm.endsWith("_id") || fieldNorm.endsWith("_ids")) && (produced.normalized === "id" || produced.normalized === "ids")) {
    return true;
  }
  if (fieldNorm.endsWith("_sha") && produced.normalized === "sha") return true;
  if (fieldNorm.endsWith("_ref") && produced.normalized === "ref") return true;
  if (fieldNorm === "node_id" && produced.normalized === "node_id") return true;
  // camelCase ids like migrationId already normalized to migration_id
  return false;
}

function slugTokens(producer: CompactTool): Set<string> {
  return tokenSet(`${producer.slug} ${producer.name}`);
}

function extraEntityPenalty(producer: CompactTool, fTokens: Set<string>): number {
  let extra = 0;
  for (const t of slugTokens(producer)) {
    if (CONTEXT_TOKENS.has(t) || fTokens.has(t)) continue;
    extra++;
  }
  return Math.min(8, extra * 4);
}



function scoreProducer(
  consumer: CompactTool,
  field: CompactTool["identityInputs"][0],
  producer: CompactTool
): number {
  if (producer.slug === consumer.slug) return 0;

  const fTokens = fieldTokens(field);
  const pSlugToks = slugTokens(producer);

  const exact = producer.produced.some(
    (p) => p.normalized === field.normalized
  );

  const aliased = producer.produced.filter(
    (p) => aliasMatch(field.normalized, p)
  );

  let producedHits = 0;
  for (const p of aliased) {
    producedHits = Math.max(
      producedHits,
      overlap(fTokens, producedTokens(p))
    );
  }

  const slugHits = overlap(fTokens, pSlugToks);
  const mentioned = descriptionMentions(field.description, producer);

  const schemaMatch =
    exact || (aliased.length > 0 && producedHits > 0);

  if (!schemaMatch && !(mentioned && slugHits > 0)) return 0;

  let score = 0;

  if (exact) score += 14;
  else if (aliased.length > 0 && producedHits > 0) score += 10;
  else score += 4;

  if (mentioned) score += 3;
  if (slugHits > 0) score += Math.min(3, slugHits);

  if (PRODUCER_LIST_RE.test(producer.slug)) score += 3;
  else if (PRODUCER_CREATE_RE.test(producer.slug)) score += 2;

  if (PRODUCER_WEAK_RE.test(producer.slug)) score -= 4;

  if (producer.service && producer.service === consumer.service) {
    score += 2;
  }

  if (producer.deprecated) score -= 3;
  if (field.required) score += 1;

  score -= extraEntityPenalty(producer, fTokens);

  return score;
}



function inferHeuristic(compacts: CompactTool[]): ScoredEdge[] {
  const edges: ScoredEdge[] = [];
  for (const consumer of compacts) {
    for (const field of consumer.identityInputs) {
      const ranked: ScoredEdge[] = [];
      for (const producer of compacts) {
        const score = scoreProducer(consumer, field, producer);
        if (score < 8) continue;
        ranked.push({
          from: producer.slug,
          to: consumer.slug,
          label: field.name,
          score,
        });
      }
      ranked.sort((a, b) => b.score - a.score || a.from.localeCompare(b.from));
      if (
        (consumer.slug === "GITHUB_CREATE_AN_ISSUE_COMMENT" && field.name === "issue_number") ||
        (consumer.slug === "GITHUB_MERGE_A_PULL_REQUEST" && field.name === "pull_number")
      ) {
        console.error(
          consumer.slug,
          field.name,
          ranked.slice(0, 8).map((e) => `${e.score} ${e.from}`),
        );
      }
      const cap = field.required ? 5 : 2;
      edges.push(...ranked.slice(0, cap));
    }
  }
  return edges;
}

function edgeKey(e: Edge): string {
  return `${e.from}\u0000${e.to}\u0000${e.label ?? ""}`;
}

async function refineWithLlm(compacts: CompactTool[], edges: ScoredEdge[]): Promise<ScoredEdge[]> {
  const rawUrl = process.env.LITMUS_API_URL;
  const apiKey = process.env.LITMUS_ASSESSMENT_TOKEN;
  if (!rawUrl || !apiKey) return edges;

  const ambiguous = edges.filter((e) => e.score < 13);
  if (ambiguous.length === 0) return edges;

  const bySlug = new Map(compacts.map((c) => [c.slug, c]));
  const keepHigh = edges.filter((e) => e.score >= 13);
  const batches: ScoredEdge[][] = [];
  const size = 80;
  for (let i = 0; i < ambiguous.length; i += size) batches.push(ambiguous.slice(i, i + size));

  const kept: ScoredEdge[] = [...keepHigh];
  const client = new OpenAI({
    apiKey,
    baseURL: `${rawUrl.replace(/\/$/, "")}/proxy/openai/v1`,
  });

  for (const batch of batches.slice(0, 4)) {
    const payload = batch.map((e) => {
      const from = bySlug.get(e.from);
      const to = bySlug.get(e.to);
      return {
        from: e.from,
        to: e.to,
        label: e.label,
        producer_outputs: (from?.produced ?? []).slice(0, 10).map((p) => p.name),
        consumer_field: e.label,
        consumer_field_desc: (to?.identityInputs.find((i) => i.name === e.label)?.description ?? "").slice(0, 180),
      };
    });
    try {
      const completion = await client.chat.completions.create({
        model: "openai/gpt-4o",
        temperature: 0,
        response_format: { type: "json_object" },
        messages: [
          {
            role: "system",
            content:
              "Select producer-consumer tool dependency edges. Keep an edge only if the producer can supply the labeled identifier the consumer needs. Drop generic or cross-entity matches (e.g. user id vs issue number). Return JSON {\"keep\":[{\"from\":\"SLUG\",\"to\":\"SLUG\",\"label\":\"field\"}]}. Use only provided candidate slugs and labels.",
          },
          { role: "user", content: JSON.stringify({ candidates: payload }) },
        ],
      });
      const text = completion.choices[0]?.message?.content ?? "{}";
      const parsed = JSON.parse(text);
      const keepList: Edge[] = Array.isArray(parsed.keep) ? parsed.keep : [];
      const allow = new Set(batch.map(edgeKey));
      const byKey = new Map(batch.map((e) => [edgeKey(e), e]));
      for (const e of keepList) {
        const k = edgeKey({ from: e.from, to: e.to, label: e.label });
        if (!allow.has(k)) continue;
        const orig = byKey.get(k);
        if (orig) kept.push(orig);
      }
    } catch (err) {
      console.error(`LLM refine skipped a batch: ${err instanceof Error ? err.message : "error"}`);
      kept.push(...batch);
    }
  }

  return kept;
}

function finalizeEdges(edges: ScoredEdge[], validSlugs: Set<string>): Edge[] {
  const seen = new Set<string>();
  const out: Edge[] = [];
  const sorted = [...edges].sort((a, b) => b.score - a.score);
  for (const e of sorted) {
    if (!validSlugs.has(e.from) || !validSlugs.has(e.to)) continue;
    if (e.from === e.to) continue;
    if (!e.label) continue;
    const k = edgeKey(e);
    if (seen.has(k)) continue;
    seen.add(k);
    out.push({ from: e.from, to: e.to, label: e.label });
  }
  return out;
}

function writeVisualization(graph: Graph): void {
  const connected = new Set<string>();
  for (const e of graph.edges) {
    connected.add(e.from);
    connected.add(e.to);
  }
  const visNodes = graph.nodes
    .filter((n) => connected.has(n.id))
    .map((n) => ({
      id: n.id,
      label: n.id,
      group: n.service ?? "other",
      title: `${n.id}\n${n.service ?? ""}`,
    }));
  const visEdges = graph.edges.map((e, i) => ({
    id: i,
    from: e.from,
    to: e.to,
    label: e.label ?? "",
    arrows: "to",
    font: { align: "middle", size: 10 },
  }));
  const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Toolkit dependency graph</title>
  <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    body { font-family: sans-serif; margin: 0; }
    #meta { padding: 8px 12px; background: #111; color: #eee; }
    #graph { width: 100vw; height: calc(100vh - 48px); }
  </style>
</head>
<body>
  <div id="meta">${visNodes.length} connected nodes, ${visEdges.length} edges (${graph.nodes.length} tools in catalog). Edge label is the identifier the producer supplies.</div>
  <div id="graph"></div>
  <script>
    const nodes = new vis.DataSet(${JSON.stringify(visNodes)});
    const edges = new vis.DataSet(${JSON.stringify(visEdges)});
    const network = new vis.Network(document.getElementById("graph"), { nodes, edges }, {
      physics: { solver: "forceAtlas2Based", stabilization: { iterations: 200 } },
      nodes: { shape: "box", font: { size: 11 } },
      edges: { smooth: { type: "dynamic" }, color: { color: "#888" } },
      groups: {}
    });
  </script>
</body>
</html>
`;
  writeFileSync(VIZ_PATH, html, "utf-8");
}

function writeGraphMd(graph: Graph): void {
  const highlight = graph.edges.filter(
    (e) =>
      /ISSUE_COMMENT|MERGE_A_PULL_REQUEST|ACCEPT_REPOSITORY_INVITATION/i.test(e.to) ||
      /ISSUE_COMMENT|MERGE_A_PULL_REQUEST/i.test(e.from),
  );
  const sample = (highlight.length ? highlight : graph.edges).slice(0, 40);
  const lines = [
    "# Tool dependency graph",
    "",
    "Producer → consumer. Edge labels are identifier fields supplied by the producer.",
    "",
    "```mermaid",
    "flowchart LR",
  ];
  for (const e of sample) {
    const from = e.from.replace(/[^A-Za-z0-9_]/g, "_");
    const to = e.to.replace(/[^A-Za-z0-9_]/g, "_");
    lines.push(`  ${from}["${e.from}"] -->|${e.label}| ${to}["${e.to}"]`);
  }
  lines.push("```", "");
  lines.push(`Full graph: ${graph.nodes.length} nodes, ${graph.edges.length} edges. See visualization.html.`);
  writeFileSync(GRAPH_MD_PATH, lines.join("\n"), "utf-8");
}

async function generate(tools: Tool[]): Promise<Graph> {
  const compacts = tools.map(compactTool).filter((t): t is CompactTool => !!t);
  const nodes: Node[] = compacts.map((t) => ({
    id: t.slug,
    ...(t.service ? { service: t.service } : {}),
  }));
  const valid = new Set(compacts.map((t) => t.slug));
  let scored = inferHeuristic(compacts);
  scored = await refineWithLlm(compacts, scored);
  const edges = finalizeEdges(scored, valid);
  return { nodes, edges };
}

async function main() {
  const graph = await generate(loadCatalog());
  writeFileSync(OUT_PATH, JSON.stringify(graph, null, 2), "utf-8");
  writeVisualization(graph);
  writeGraphMd(graph);
  console.error(
    `wrote ${graph.nodes.length} nodes, ${graph.edges.length} edges to ${OUT_PATH}`,
  );
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
