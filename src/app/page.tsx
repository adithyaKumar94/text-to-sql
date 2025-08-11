// src/app/page.tsx
import {
  createClient,
  type PostgrestSingleResponse,
  type PostgrestError,
} from "@supabase/supabase-js";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// ----- ENV (server-side only) -----
const SUPABASE_URL =
  process.env.NEXT_PUBLIC_SUPABASE_URL ?? process.env.SUPABASE_URL!;
const SUPABASE_ANON_KEY = process.env.SUPABASE_ANON_KEY!;
const VOYAGE_API_KEY = process.env.VOYAGE_API_KEY!;
const GROQ_API_KEY = process.env.GROQ_API_KEY!;

// Server-side Supabase client (scoped to clinical)
const sb = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
  db: { schema: "clinical" },
});

// ---------- types & helpers ----------
type SchemaRow = { table: string; columns: string[] };
type RowObject = Record<string, unknown>;
type RpcRes<T> = PostgrestSingleResponse<T>;

function cleanSql(s: string): string {
  let out = s ?? "";
  out = out.replace(/^```(?:\w+)?\s*/i, "").replace(/\s*```$/i, ""); // code fences
  out = out.replace(/^\s*--\s*param:.*$/gim, ""); // hint comments
  out = out.replace(/;+(\s*)$/g, "$1"); // trailing semicolons
  return out.trim();
}

function errMessage(e: unknown): string {
  if (e instanceof Error) return e.message;
  if (typeof e === "string") return e;
  try {
    return JSON.stringify(e);
  } catch {
    return "Unknown error";
  }
}

function buildPrompt(
  question: string,
  ctxs: Array<{ content: string }>,
  schemaList: SchemaRow[]
): string {
  const ctx = (ctxs || [])
    .map((c, i) => `### Context ${i + 1}\n${c.content}`)
    .join("\n\n");
  const schemaWhitelist = (schemaList || [])
    .map((t) => `- ${t.table}: ${t.columns.join(", ")}`)
    .join("\n");

  return `
Return ONLY valid Postgres SQL. Do NOT wrap in markdown fences. One statement only (CTEs ok). No trailing semicolon.

Use ONLY these tables/columns (anything else is invalid):
${schemaWhitelist}

Extra context:
${ctx}

Hard rules:
- Fully-qualify clinical tables (e.g., clinical.patients).
- There is NO column "name" and NO column "next_appointment".
- Use clinical.patients.full_name for patient names.
- "Next appointment" = the smallest clinical.appointments.starts_at >= NOW() with status='scheduled' per patient (use a LATERAL subquery or the view clinical.v_patient_next_appointment).
- Prefer CTEs.
- Avoid parameters ($1, $2...). Use NOW() and LIMIT 5 if unsure.

User question: "${question}"
`.trim();
}

async function sleep(ms: number): Promise<void> {
  await new Promise<void>((r) => setTimeout(r, ms));
}

async function voyageEmbed(
  text: string,
  type: "query" | "document" = "query"
): Promise<number[]> {
  const call = async (): Promise<number[]> => {
    const res = await fetch("https://api.voyageai.com/v1/embeddings", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${VOYAGE_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "voyage-3.5",
        input: [text],
        input_type: type,
        output_dimension: 1024,
      }),
    });
    const json = (await res.json()) as {
      data?: Array<{ embedding: number[] }>;
      detail?: unknown;
    };
    if (!res.ok || !json?.data?.[0]?.embedding) {
      throw new Error(
        "Embedding failed: " +
          (json ? JSON.stringify(json) : String(res.status))
      );
    }
    return json.data[0].embedding;
  };

  try {
    return await call();
  } catch (e) {
    const m = errMessage(e).toLowerCase();
    if (
      m.includes("429") ||
      m.includes("rate") ||
      m.includes("reduced rate") ||
      m.includes("payment method")
    ) {
      await sleep(25000); // single backoff for Voyage free-tier
      return await call();
    }
    throw new Error("Embedding failed: " + errMessage(e));
  }
}

async function groqSQL(prompt: string): Promise<string> {
  const res = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${GROQ_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "llama-3.1-8b-instant",
      temperature: 0,
      messages: [
        { role: "system", content: "Return only SQL. No explanations." },
        { role: "user", content: prompt },
      ],
    }),
  });
  const json = (await res.json()) as {
    choices?: Array<{ message?: { content?: string } }>;
  };
  const sql = json?.choices?.[0]?.message?.content?.trim();
  if (!sql) throw new Error("No SQL from LLM");
  return sql;
}

async function fetchLiveSchema(): Promise<SchemaRow[]> {
  const { data, error }: RpcRes<SchemaRow[]> = await sb.rpc("run_sql_ro", {
    q: `
      SELECT table_name AS table,
             array_agg(column_name ORDER BY ordinal_position) AS columns
      FROM information_schema.columns
      WHERE table_schema = 'clinical'
      GROUP BY table_name
      ORDER BY table_name
    `,
  });
  if (error) throw new Error(error.message);
  return data ?? [];
}

async function retrieveContext(
  qVec: number[],
  topK = 6
): Promise<Array<{ content: string }>> {
  const { data, error }: RpcRes<Array<{ content: string }>> = await sb.rpc(
    "match_schema",
    {
      query_embedding: qVec,
      match_count: topK,
    }
  );
  if (error) throw new Error("match_schema: " + error.message);
  return data ?? [];
}

async function runRPC(
  sql: string
): Promise<PostgrestSingleResponse<RowObject[]>> {
  return sb.rpc("run_sql_ro", { q: sql });
}

// ---------- Page (Server Component) ----------
export default async function Page({
  searchParams,
}: {
  searchParams: Promise<{ q?: string | string[] }>;
}) {
  const sp = await searchParams;
  const question = (Array.isArray(sp?.q) ? sp.q[0] : sp?.q ?? "").trim();

  let sql = "";
  let repaired = false;
  let rows: RowObject[] = [];
  let errorMsg = "";

  if (question) {
    try {
      const schema = await fetchLiveSchema();
      const qVec = await voyageEmbed(question, "query");
      const ctx = await retrieveContext(qVec, 6);

      // 1) generate & sanitize
      const prompt = buildPrompt(question, ctx, schema);
      sql = cleanSql(await groqSQL(prompt));

      // heuristic guard for common bogus columns
      if (/\b[^_.]name\b/i.test(sql) || /\bnext_appointment\b/i.test(sql)) {
        sql = `
WITH next_appt AS (
  SELECT p.id, p.full_name,
         (SELECT a.starts_at
          FROM clinical.appointments a
          WHERE a.patient_id = p.id
            AND a.status = 'scheduled'
            AND a.starts_at >= NOW()
          ORDER BY a.starts_at
          LIMIT 1) AS next_starts_at
  FROM clinical.patients p
)
SELECT id, full_name, next_starts_at
FROM next_appt
ORDER BY next_starts_at NULLS LAST
LIMIT 5
        `.trim();
      }

      // 2) run
      const first = await runRPC(sql);
      rows = first.data ?? [];
      let rpcErr: PostgrestError | null = first.error;

      // 3) one-shot repair on SQL/schema errors
      const repairableCodes = new Set(["42703", "42P01", "42601", "42883"]);
      if (rpcErr && rpcErr.code && repairableCodes.has(rpcErr.code)) {
        const repairPrompt = `
Previous SQL had an error:

SQL:
${sql}

DB error: ${rpcErr.message}

Fix the SQL. Use ONLY these tables/columns:
${schema.map((t) => `- ${t.table}: ${t.columns.join(", ")}`).join("\n")}

Rules:
- Fully-qualify clinical tables.
- No unknown columns like "name" or "next_appointment".
- One statement only; no markdown fences; no trailing semicolon.
- Prefer CTEs; use NOW() and LIMIT 5 if unsure.

User question: "${question}"
        `.trim();

        const repairedSql = cleanSql(await groqSQL(repairPrompt));
        const second = await runRPC(repairedSql);

        if (!second.error) {
          sql = repairedSql;
          rows = second.data ?? [];
          repaired = true;
          rpcErr = null;
        } else {
          rpcErr = second.error;
        }
      }

      if (rpcErr) {
        errorMsg = rpcErr.message ?? "Query failed";
      }
    } catch (e) {
      errorMsg = errMessage(e);
    }
  }

  return (
    <main
      style={{
        maxWidth: 900,
        margin: "40px auto",
        padding: "0 16px",
        fontFamily: "Inter, system-ui, sans-serif",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          marginBottom: 8,
        }}
      >
        <img src="/logo-eg.svg" alt="Company logo" width={32} height={32} />
        <h1 style={{ fontSize: 28, fontWeight: 700, margin: 0 }}>
          - HealthCare
        </h1>
      </div>
      <h1 style={{ fontSize: 28, fontWeight: 700, margin: 0 }}>
        Clinical SQL Chat
      </h1>
      <p style={{ color: "#555", marginBottom: 18 }}>
        Ask a question about your <code>clinical</code> schema. I’ll generate
        safe SQL and run it via Supabase.
      </p>

      <form method="GET" style={{ display: "flex", gap: 8, marginBottom: 16 }}>
        <input
          name="q"
          defaultValue={question}
          placeholder="e.g., List 5 patients with their next appointment after today"
          style={{
            flex: 1,
            padding: "12px 14px",
            borderRadius: 12,
            border: "1px solid #e5e7eb",
          }}
        />
        <button
          type="submit"
          style={{
            padding: "12px 16px",
            borderRadius: 12,
            border: "1px solid transparent",
            background: "#2563eb",
            color: "#fff",
            fontWeight: 600,
          }}
        >
          Ask
        </button>
      </form>

      {question ? (
        <>
          <div style={{ marginBottom: 12, color: "#444" }}>
            <strong>Question:</strong> {question}
          </div>

          {sql && (
            <div style={{ marginBottom: 12 }}>
              <div style={{ fontWeight: 600, marginBottom: 6 }}>
                Generated SQL{repaired ? " (after fix)" : ""}:
              </div>
              <pre
                style={{
                  whiteSpace: "pre-wrap",
                  padding: 12,
                  background: "#0b1020",
                  color: "#e2e8f0",
                  borderRadius: 8,
                }}
              >
                {sql}
              </pre>
            </div>
          )}

          {errorMsg && (
            <div style={{ color: "#b91c1c", marginTop: 8, marginBottom: 12 }}>
              Error: {errorMsg}
            </div>
          )}

          {rows.length > 0 ? (
            <div
              style={{
                overflowX: "auto",
                border: "1px solid #e5e7eb",
                borderRadius: 8,
              }}
            >
              <table style={{ borderCollapse: "collapse", width: "100%" }}>
                <thead>
                  <tr>
                    {Object.keys(rows[0] as RowObject).map((k) => (
                      <th
                        key={k}
                        style={{
                          textAlign: "left",
                          padding: "8px 10px",
                          borderBottom: "1px solid #e5e7eb",
                        }}
                      >
                        {k}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.slice(0, 100).map((r, i) => (
                    <tr key={i}>
                      {Object.keys(rows[0] as RowObject).map((k) => (
                        <td
                          key={k}
                          style={{
                            padding: "8px 10px",
                            borderBottom: "1px solid #f3f4f6",
                          }}
                        >
                          {String((r as RowObject)[k])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
              <div style={{ fontSize: 12, color: "#666", padding: 8 }}>
                Showing {Math.min(rows.length, 100)} row(s).
              </div>
            </div>
          ) : question && !errorMsg ? (
            <div style={{ color: "#444", marginTop: 8 }}>No rows.</div>
          ) : null}
        </>
      ) : (
        <div style={{ color: "#666" }}>
          Tip: try “List 5 patients with their next appointment after today”.
        </div>
      )}
    </main>
  );
}
