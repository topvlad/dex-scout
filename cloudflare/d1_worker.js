const ALLOWED_TABLES = new Set(["app_storage", "runtime_state", "locks", "tg_state", "job_heartbeats", "job_runs"]);
const ALLOWED_OPS = new Set(["eq", "lt", "gt"]);

function json(data, status = 200) {
  return new Response(JSON.stringify(data), { status, headers: { "content-type": "application/json" } });
}

function authOk(req, env) {
  const h = req.headers.get("authorization") || "";
  return h === `Bearer ${env.D1_PROXY_TOKEN || ""}`;
}

function cond(col, op) {
  if (!col) return { sql: "", params: [] };
  if (!ALLOWED_OPS.has(op)) throw new Error("bad_filter_op");
  const opSql = op === "eq" ? "=" : op === "lt" ? "<" : ">";
  return { sql: ` where ${col} ${opSql} ?`, params: [] };
}

export default {
  async fetch(req, env) {
    const url = new URL(req.url);
    if (url.pathname === "/health") return json({ ok: true });
    if (!authOk(req, env)) return json({ ok: false, code: "unauthorized" }, 401);

    const parts = url.pathname.split("/").filter(Boolean);
    try {
      if (parts[0] === "v1" && parts[1] === "storage" && parts[2]) {
        const key = decodeURIComponent(parts[2]);
        if (req.method === "GET") {
          const r = await env.DB.prepare("select key, content, updated_at from app_storage where key = ? limit 1").bind(key).first();
          return json({ ok: true, found: !!r, key, content: r?.content || null, updated_at: r?.updated_at || "" });
        }
        if (req.method === "PUT") {
          const body = await req.json();
          const content = String(body?.content || "");
          await env.DB.prepare("insert into app_storage (key, content, updated_at) values (?, ?, ?) on conflict(key) do update set content=excluded.content, updated_at=excluded.updated_at").bind(key, content, new Date().toISOString()).run();
          return json({ ok: true, key });
        }
        if (req.method === "DELETE") {
          const out = await env.DB.prepare("delete from app_storage where key = ?").bind(key).run();
          return json({ ok: true, deleted: out.meta?.changes || 0 });
        }
      }
      if (parts[0] === "v1" && parts[1] === "storage-size" && parts[2] && req.method === "GET") {
        const key = decodeURIComponent(parts[2]);
        const r = await env.DB.prepare("select length(content) as bytes from app_storage where key = ? limit 1").bind(key).first();
        return json({ ok: true, key, bytes: Number(r?.bytes || 0) });
      }
      if (parts[0] === "v1" && parts[1] === "storage-sizes" && req.method === "GET") {
        const limit = Math.max(1, Math.min(1000, Number(url.searchParams.get("limit") || 200)));
        const rows = (await env.DB.prepare("select key, length(content) as bytes, updated_at from app_storage order by bytes desc limit ?").bind(limit).all()).results || [];
        return json({ ok: true, rows });
      }
      if (parts[0] === "v1" && parts[1] === "table" && parts[2]) {
        const table = parts[2];
        if (!ALLOWED_TABLES.has(table)) return json({ ok: false, code: "table_not_allowed" }, 400);
        if (req.method === "GET") {
          const select = url.searchParams.get("select") || "*";
          const limit = Math.max(1, Math.min(2000, Number(url.searchParams.get("limit") || 1)));
          const col = url.searchParams.get("filter_col");
          const op = url.searchParams.get("filter_op") || "eq";
          const val = url.searchParams.get("filter_value") || "";
          const where = cond(col, op);
          const sql = `select ${select} from ${table}${where.sql} limit ?`;
          const stmt = env.DB.prepare(sql);
          const params = col ? [val, limit] : [limit];
          const rows = (await stmt.bind(...params).all()).results || [];
          return json({ ok: true, rows });
        }
        if (req.method === "POST" && parts[3] === "upsert") {
          const body = await req.json();
          const payload = body?.payload || {};
          const onConflict = String(body?.on_conflict || "");
          const cols = Object.keys(payload);
          const vals = Object.values(payload);
          const updates = cols.filter((c) => c !== onConflict).map((c) => `${c}=excluded.${c}`).join(",");
          const placeholders = cols.map(() => "?").join(",");
          const sql = `insert into ${table} (${cols.join(",")}) values (${placeholders}) on conflict(${onConflict}) do update set ${updates}`;
          await env.DB.prepare(sql).bind(...vals).run();
          return json({ ok: true });
        }
        if (req.method === "DELETE") {
          const selectCol = url.searchParams.get("select_col") || "run_id";
          const col = url.searchParams.get("filter_col") || "";
          const op = url.searchParams.get("filter_op") || "eq";
          const val = url.searchParams.get("filter_value") || "";
          const where = cond(col, op);
          const sql = `delete from ${table}${where.sql} returning ${selectCol}`;
          const rows = (await env.DB.prepare(sql).bind(...(col ? [val] : [])).all()).results || [];
          return json({ ok: true, deleted: rows.length });
        }
      }
      return json({ ok: false, code: "not_found" }, 404);
    } catch (e) {
      return json({ ok: false, code: "worker_error", detail: String(e?.message || e) }, 500);
    }
  },
};
