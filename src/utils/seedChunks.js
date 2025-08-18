// seedChunks_batch.js
import "dotenv/config";
import { createClient } from "@supabase/supabase-js";

const sb = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY,
  { db: { schema: "clinical" } }
);

const CHUNKS = [
  "Table doctors(id uuid pk, full_name text, specialization text).",
  "Table patients(id uuid pk, full_name text, gender, dob).",
  "Table appointments(id uuid pk, patient_id->patients.id, doctor_id->doctors.id, starts_at timestamptz, status).",
  "Next appointment = smallest starts_at >= now() with status = scheduled.",
];

async function embedBatch(texts) {
  const res = await fetch("https://api.voyageai.com/v1/embeddings", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${process.env.VOYAGE_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "voyage-3.5",
      input: texts, // <— batch all at once
      input_type: "document",
      output_dimension: 1024,
    }),
  });
  const json = await res.json();
  if (!res.ok) {
    console.error("Voyage error", res.status, json);
    throw new Error("Embedding request failed");
  }
  const out = json?.data?.map((d) => d.embedding) ?? [];
  if (out.length !== texts.length)
    throw new Error("Mismatched embedding count");
  return out;
}

const embeddings = await embedBatch(CHUNKS);
const rows = CHUNKS.map((content, i) => ({
  content,
  embedding: embeddings[i],
}));

const { error } = await sb.from("schema_chunks").insert(rows);
if (error) {
  console.error("Supabase insert error:", error);
  process.exit(1);
}
console.log("Seeded", rows.length, "chunks in one go.");
