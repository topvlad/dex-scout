const ALLOWED_TABLES = new Set(["app_storage", "runtime_state", "locks", "tg_state", "job_heartbeats", "job_runs"]);
const ALLOWED_OPS = new Set(["eq", "lt", "gt"]);
const JSON_TEXT_FIELDS = new Set(["state_json", "meta_json"]);

function json(data, status = 200) {
  return new Response(JSON.stringify(data), { status, headers: { "content-type": "application/json" } });
}

function authOk(req, env) {
  const h = req.headers.get("authorization") || "";
  return h === `Bearer ${env.D1_PROXY_TOKEN || ""}`;
}

function isSafeIdentifier(name) {
  return /^[A-Za-z_][A-Za-z0-9_]*$/.test(String(name || ""));
}

function normalizeSelect(selectRaw) {
  const select = String(selectRaw || "*").trim();
  if (select === "*") return "*";
  const cols = select.split(",").map((s) => s.trim()).filter(Boolean);
  if (!cols.length || cols.some((c) => !isSafeIdentifier(c))) throw new Error("unsafe_select");
  return cols.join(",");
}

function cond(col, op) {
  if (!col) return { sql: "", params: [] };
  if (!isSafeIdentifier(col)) throw new Error("unsafe_filter_col");
  if (!ALLOWED_OPS.has(op)) throw new Error("bad_filter_op");
  const opSql = op === "eq" ? "=" : op === "lt" ? "<" : ">";
  return { sql: ` where ${col} ${opSql} ?` };
}

function parseJsonFields(row) {
  if (!row || typeof row !== "object") return row;
  const out = { ...row };
  for (const k of Object.keys(out)) {
    if (JSON_TEXT_FIELDS.has(k) && typeof out[k] === "string") {
      try { out[k] = JSON.parse(out[k]); } catch (_) {}
    }
  }
  return out;
}

export default {
  async fetch(req, env) {
    const url = new URL(req.url);
    if (url.pathname === "/health") return json({ ok: true });
    if (!authOk(req, env)) return json({ ok: false, code: "unauthorized" }, 401);

    const parts = url.pathname.split("/").filter(Boolean);
    try {
      if (parts[0] === "v1" && parts[1] === "storage" && parts.length > 2) {
        const key = decodeURIComponent(parts.slice(2).join("/"));
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
      if (parts[0] === "v1" && parts[1] === "storage-size" && parts.length > 2 && req.method === "GET") {
        const key = decodeURIComponent(parts.slice(2).join("/"));
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
          const select = normalizeSelect(url.searchParams.get("select") || "*");
          const limit = Math.max(1, Math.min(2000, Number(url.searchParams.get("limit") || 1)));
          const col = url.searchParams.get("filter_col");
          const op = url.searchParams.get("filter_op") || "eq";
          const val = url.searchParams.get("filter_value") || "";
          const where = cond(col, op);
          const sql = `select ${select} from ${table}${where.sql} limit ?`;
          const rows = (await env.DB.prepare(sql).bind(...(col ? [val, limit] : [limit])).all()).results || [];
          return json({ ok: true, rows: rows.map(parseJsonFields) });
        }
        if (req.method === "POST" && parts[3] === "upsert") {
          const body = await req.json();
          const payload = body?.payload || {};
          const onConflict = String(body?.on_conflict || "");
          if (!isSafeIdentifier(onConflict)) return json({ ok: false, code: "unsafe_on_conflict" }, 400);
          const cols = Object.keys(payload);
          if (!cols.length || cols.some((c) => !isSafeIdentifier(c))) return json({ ok: false, code: "unsafe_payload_column" }, 400);
          const vals = cols.map((c) => {
            const val = payload[c];
            if (JSON_TEXT_FIELDS.has(c) && (Array.isArray(val) || (val && typeof val === "object"))) return JSON.stringify(val);
            return val;
          });
          const updates = cols.filter((c) => c !== onConflict).map((c) => `${c}=excluded.${c}`).join(",");
          const sql = `insert into ${table} (${cols.join(",")}) values (${cols.map(() => "?").join(",")}) on conflict(${onConflict}) do update set ${updates}`;
          await env.DB.prepare(sql).bind(...vals).run();
          return json({ ok: true });
        }
        if (req.method === "DELETE") {
          const selectCol = String(url.searchParams.get("select_col") || "run_id");
          if (!isSafeIdentifier(selectCol)) return json({ ok: false, code: "unsafe_select_col" }, 400);
          const col = url.searchParams.get("filter_col") || "";
          const op = url.searchParams.get("filter_op") || "eq";
          const val = url.searchParams.get("filter_value") || "";
          const where = cond(col, op);
          const rows = (await env.DB.prepare(`delete from ${table}${where.sql} returning ${selectCol}`).bind(...(col ? [val] : [])).all()).results || [];
          return json({ ok: true, deleted: rows.length });
        }
      }
      return json({ ok: false, code: "not_found" }, 404);
    } catch (e) {
      return json({ ok: false, code: "worker_error", detail: String(e?.message || e) }, 500);
    }
  },
};
