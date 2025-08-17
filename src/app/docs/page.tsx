// src/app/docs/page.tsx
import {
  createClient,
  type PostgrestSingleResponse,
} from "@supabase/supabase-js";
import { redirect } from "next/navigation";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/* ========= ENV ========= */
const SUPABASE_URL =
  process.env.NEXT_PUBLIC_SUPABASE_URL ?? process.env.SUPABASE_URL!;
const SUPABASE_ANON_KEY = process.env.SUPABASE_ANON_KEY!;
const VOYAGE_API_KEY = process.env.VOYAGE_API_KEY!;
const GROQ_API_KEY = process.env.GROQ_API_KEY!;

/* ========= SB client (clinical) ========= */
const sb = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
  db: { schema: "clinical" },
});

/* ========= Types & helpers ========= */
type RowObject = Record<string, unknown>;
type DocRow = {
  id: string;
  title: string;
  filename: string;
  created_at: string;
};

function errMessage(e: unknown) {
  if (e instanceof Error) return e.message;
  if (typeof e === "string") return e;
  try {
    return JSON.stringify(e);
  } catch {
    return "Unknown error";
  }
}

function chunkText(text: string, max = 2000, overlap = 200) {
  const chunks: string[] = [];
  let i = 0;
  while (i < text.length) {
    const end = Math.min(i + max, text.length);
    chunks.push(text.slice(i, end));
    if (end === text.length) break;
    i = end - overlap;
  }
  return chunks;
}

async function sleep(ms: number) {
  await new Promise<void>((r) => setTimeout(r, ms));
}

async function embedVoyage(input: string[]): Promise<number[][]> {
  const res = await fetch("https://api.voyageai.com/v1/embeddings", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${VOYAGE_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "voyage-3.5",
      input,
      input_type: "document",
      output_dimension: 1024,
    }),
  });
  const json = (await res.json()) as {
    data?: Array<{ embedding: number[] }>;
    detail?: unknown;
  };
  if (!res.ok || !json?.data) {
    throw new Error(
      "Embedding failed: " + (json ? JSON.stringify(json) : String(res.status))
    );
  }
  return json.data.map((d) => d.embedding);
}

async function embedWithRetry(texts: string[]) {
  try {
    return await embedVoyage(texts);
  } catch (e) {
    const m = errMessage(e).toLowerCase();
    if (
      m.includes("429") ||
      m.includes("rate") ||
      m.includes("payment method") ||
      m.includes("reduced rate")
    ) {
      // one backoff for Voyage free-tier
      await sleep(25000);
      return await embedVoyage(texts);
    }
    throw e;
  }
}

function buildQAContext(
  chunks: Array<{ content: string; similarity: number }>
) {
  return chunks
    .map(
      (c, i) =>
        `### Chunk ${i + 1} (sim=${c.similarity.toFixed(2)})\n${c.content}`
    )
    .join("\n\n");
}

function buildQAPrompt(question: string, context: string) {
  return `
You are a helpful assistant. Answer strictly using the context. If unsure, say you don't know.

${context}

Question: "${question}"

Answer in concise, plain language. Cite which chunk numbers you used.
`.trim();
}

async function groqAnswer(prompt: string): Promise<string> {
  const res = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${GROQ_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "llama-3.1-8b-instant",
      temperature: 0.2,
      messages: [
        { role: "system", content: "Answer using ONLY the provided context." },
        { role: "user", content: prompt },
      ],
    }),
  });
  const json = (await res.json()) as {
    choices?: Array<{ message?: { content?: string } }>;
  };
  const out = json?.choices?.[0]?.message?.content?.trim();
  if (!out) throw new Error("No answer from model");
  return out;
}

/* ========= Server actions ========= */
async function listDocs(): Promise<DocRow[]> {
  // use your read-only runner to avoid table grants
  const { data, error }: PostgrestSingleResponse<DocRow[]> = await sb.rpc(
    "run_sql_ro",
    {
      q: `
      SELECT id, title, filename, created_at
      FROM clinical.documents
      ORDER BY created_at DESC
      LIMIT 50
    `,
    }
  );
  if (error) throw new Error(error.message);
  return data ?? [];
}

async function doIngest(formData: FormData) {
  "use server";
  const file = formData.get("file");
  const titleFromForm = (formData.get("title") as string) || "";

  if (!(file instanceof File)) {
    throw new Error("No file uploaded");
  }
  const filename = file.name;
  const mime = file.type || "text/plain";
  const size = file.size;

  // Keep it simple: support .txt /.md here
  const ok =
    filename.toLowerCase().endsWith(".txt") ||
    filename.toLowerCase().endsWith(".md");
  if (!ok) {
    throw new Error("Only .txt and .md supported in this demo");
  }

  const buf = Buffer.from(await file.arrayBuffer());
  const text = buf.toString("utf8");
  const chunks = chunkText(text, 2000, 200);

  // 1) create document row
  const title = titleFromForm || filename;
  const { data: docIdRows, error: docErr } = await sb.rpc("add_document", {
    p_title: title,
    p_filename: filename,
    p_mime: mime,
    p_size: size,
  });
  if (docErr) throw new Error(docErr.message);
  const docId = (docIdRows as unknown as string) || docIdRows;

  // 2) embed in batches (Voyage)
  const vectors = await embedWithRetry(chunks);

  // 3) insert chunks via RPC
  for (let i = 0; i < chunks.length; i++) {
    const { error } = await sb.rpc("add_doc_chunk", {
      p_doc_id: docId as string,
      p_content: chunks[i],
      p_embedding: vectors[i],
    });
    if (error) throw new Error(error.message);
  }

  redirect(`/docs?ingested=1&title=${encodeURIComponent(title)}&doc=${docId}`);
}

async function retrieveAndAnswer(params: {
  q: string;
  doc?: string;
  topK?: number;
}) {
  const q = params.q.trim();
  const docId = params.doc?.trim() || null;
  const topK = Math.max(1, Math.min(params.topK ?? 6, 12));

  // embed question
  const qvec = (await embedWithRetry([q]))[0];

  // retrieve
  const { data, error } = await sb.rpc("match_doc_chunks", {
    query_embedding: qvec,
    match_count: topK,
    p_doc_id: docId,
  });
  if (error) throw new Error(error.message);

  const hits =
    (data as Array<{ doc_id: string; content: string; similarity: number }>) ||
    [];
  const context = buildQAContext(hits);
  const prompt = buildQAPrompt(q, context);
  const answer = await groqAnswer(prompt);

  return { hits, answer };
}

/* ========= Page ========= */
export default async function Page({
  searchParams,
}: {
  searchParams: Promise<{
    q?: string | string[];
    doc?: string | string[];
    ingested?: string;
    title?: string;
  }>;
}) {
  const sp = await searchParams;
  const q = (Array.isArray(sp?.q) ? sp.q[0] : sp?.q || "").trim();
  const doc = (Array.isArray(sp?.doc) ? sp.doc[0] : sp?.doc || "").trim();
  const ingested =
    (Array.isArray(sp?.ingested) ? sp.ingested[0] : sp?.ingested) === "1";
  const ingestedTitle = Array.isArray(sp?.title)
    ? sp.title[0]
    : sp?.title || "";

  const docs = await listDocs();

  let qa: {
    hits: Array<{ doc_id: string; content: string; similarity: number }>;
    answer: string;
  } | null = null;
  let errorMsg = "";

  if (q) {
    try {
      qa = await retrieveAndAnswer({ q, doc: doc || undefined, topK: 6 });
    } catch (e) {
      errorMsg = errMessage(e);
    }
  }

  return (
    <main
      style={{
        maxWidth: 960,
        margin: "40px auto",
        padding: "0 16px",
        fontFamily: "Inter, system-ui, sans-serif",
      }}
    >
      {/* Header */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 12,
          marginBottom: 16,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <img src="/logo-eg.svg" alt="Company logo" width={32} height={32} />
          <h1 style={{ fontSize: 24, fontWeight: 700, margin: 0 }}>
            Document Q&A
          </h1>
        </div>
        <div style={{ color: "#64748b", fontWeight: 600 }}>- healthcare</div>
      </div>

      {/* Ingest form */}
      <section
        style={{
          border: "1px solid #e5e7eb",
          borderRadius: 12,
          padding: 16,
          marginBottom: 18,
          background: "#fafafa",
        }}
      >
        <h2 style={{ fontSize: 18, marginTop: 0 }}>
          Upload a document (.txt/.md)
        </h2>
        <form
          action={doIngest}
          encType="multipart/form-data"
          style={{ display: "grid", gap: 10 }}
        >
          <input type="text" name="title" placeholder="Optional title" />
          <input
            type="file"
            name="file"
            accept=".txt,.md,text/plain,text/markdown"
            required
          />
          <button
            type="submit"
            style={{
              padding: "10px 14px",
              borderRadius: 10,
              border: "1px solid transparent",
              background: "#2563eb",
              color: "#fff",
              fontWeight: 600,
              width: 140,
            }}
          >
            Ingest
          </button>
        </form>
        {ingested && (
          <div style={{ marginTop: 10, color: "#15803d" }}>
            ✅ Ingested: <strong>{ingestedTitle}</strong>
          </div>
        )}
      </section>

      {/* Ask form */}
      <section
        style={{
          border: "1px solid #e5e7eb",
          borderRadius: 12,
          padding: 16,
          marginBottom: 18,
          background: "#fff",
        }}
      >
        <h2 style={{ fontSize: 18, marginTop: 0 }}>Ask your documents</h2>
        <form
          method="GET"
          style={{ display: "flex", gap: 8, flexWrap: "wrap" }}
        >
          <input
            name="q"
            defaultValue={q}
            placeholder="e.g., What does the policy say about cancellations?"
            style={{
              flex: 1,
              minWidth: 260,
              padding: "10px 12px",
              borderRadius: 10,
              border: "1px solid #e5e7eb",
            }}
          />
          <select
            name="doc"
            defaultValue={doc || ""}
            style={{
              padding: "10px 12px",
              borderRadius: 10,
              border: "1px solid #e5e7eb",
            }}
          >
            <option value="">All documents</option>
            {docs.map((d) => (
              <option key={d.id} value={d.id}>
                {d.title || d.filename} —{" "}
                {new Date(d.created_at).toLocaleDateString()}
              </option>
            ))}
          </select>
          <button
            type="submit"
            style={{
              padding: "10px 14px",
              borderRadius: 10,
              border: "1px solid transparent",
              background: "#2563eb",
              color: "#fff",
              fontWeight: 600,
            }}
          >
            Ask
          </button>
        </form>
      </section>

      {/* Results */}
      {q ? (
        errorMsg ? (
          <div style={{ color: "#b91c1c" }}>Error: {errorMsg}</div>
        ) : qa ? (
          <section
            style={{
              border: "1px solid #e5e7eb",
              borderRadius: 12,
              padding: 16,
              background: "#fff",
            }}
          >
            <div style={{ marginBottom: 8, color: "#444" }}>
              <strong>Question:</strong> {q}
            </div>

            <div style={{ whiteSpace: "pre-wrap", marginBottom: 16 }}>
              <strong>Answer:</strong> {qa.answer}
            </div>

            <details>
              <summary style={{ cursor: "pointer" }}>Show top chunks</summary>
              <ol>
                {qa.hits.map((h, i) => (
                  <li key={i} style={{ margin: "8px 0" }}>
                    <div style={{ fontSize: 12, color: "#64748b" }}>
                      sim={h.similarity.toFixed(2)} doc={h.doc_id}
                    </div>
                    <div style={{ whiteSpace: "pre-wrap" }}>{h.content}</div>
                  </li>
                ))}
              </ol>
            </details>
          </section>
        ) : null
      ) : null}
    </main>
  );
}
