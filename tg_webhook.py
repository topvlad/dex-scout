from typing import Any, Dict

from fastapi import FastAPI, Request

from app import (
    MON_FIELDS,
    PORTFOLIO_FIELDS,
    addr_key,
    addr_store,
    load_monitoring,
    load_portfolio,
    normalize_chain_name,
    now_utc_str,
    save_monitoring,
    save_portfolio,
)

app = FastAPI()


def add_contract_to_portfolio(chain: str, ca: str) -> None:
    chain = normalize_chain_name(chain or "")
    ca = addr_store(chain, ca or "")
    if not chain or not ca:
        return

    rows = load_portfolio()
    key = addr_key(chain, ca)
    for r in rows:
        row_chain = normalize_chain_name(r.get("chain") or "")
        row_ca = addr_store(row_chain, r.get("base_token_address") or "")
        if addr_key(row_chain, row_ca) == key:
            r["active"] = "1"
            save_portfolio(rows)
            return

    row: Dict[str, Any] = {k: "" for k in PORTFOLIO_FIELDS}
    row.update(
        {
            "ts_utc": now_utc_str(),
            "chain": chain,
            "base_symbol": "UNKNOWN",
            "base_token_address": ca,
            "active": "1",
            "note": "added_from_tg_callback",
        }
    )
    rows.append(row)
    save_portfolio(rows)


def add_contract_to_monitoring(chain: str, ca: str) -> None:
    chain = normalize_chain_name(chain or "")
    ca = addr_store(chain, ca or "")
    if not chain or not ca:
        return

    rows = load_monitoring()
    key = addr_key(chain, ca)
    for r in rows:
        row_chain = normalize_chain_name(r.get("chain") or "")
        row_ca = addr_store(row_chain, r.get("base_addr") or "")
        if addr_key(row_chain, row_ca) == key:
            r["active"] = "1"
            r["status"] = r.get("status") or "WATCH"
            r["last_seen"] = now_utc_str()
            save_monitoring(rows)
            return

    row: Dict[str, Any] = {k: "" for k in MON_FIELDS}
    row.update(
        {
            "ts_added": now_utc_str(),
            "last_seen": now_utc_str(),
            "chain": chain,
            "base_symbol": "UNKNOWN",
            "base_addr": ca,
            "active": "1",
            "entry_status": "WATCH",
            "status": "WATCH",
            "note": "added_from_tg_callback",
        }
    )
    rows.append(row)
    save_monitoring(rows)


def remove_contract_everywhere(chain: str, ca: str) -> None:
    chain = normalize_chain_name(chain or "")
    ca = addr_store(chain, ca or "")
    if not chain or not ca:
        return

    key = addr_key(chain, ca)

    p_rows = load_portfolio()
    for r in p_rows:
        row_chain = normalize_chain_name(r.get("chain") or "")
        row_ca = addr_store(row_chain, r.get("base_token_address") or "")
        if addr_key(row_chain, row_ca) == key:
            r["active"] = "0"
            r["note"] = "removed_from_tg_callback"
    save_portfolio(p_rows)

    m_rows = load_monitoring()
    for r in m_rows:
        row_chain = normalize_chain_name(r.get("chain") or "")
        row_ca = addr_store(row_chain, r.get("base_addr") or "")
        if addr_key(row_chain, row_ca) == key:
            r["active"] = "0"
            r["status"] = "ARCHIVED"
            r["ts_archived"] = now_utc_str()
            r["archived_reason"] = "removed_from_tg_callback"
    save_monitoring(m_rows)


@app.post("/tg_webhook")
async def tg_webhook(req: Request):
    data = await req.json()
    cb = data.get("callback_query")
    if not cb:
        return {"ok": True}

    payload = cb.get("data", "")
    parts = payload.split("|")
    if len(parts) != 3:
        return {"ok": True}

    action, chain, ca = parts
    print(f"[TG CALLBACK] action={action} chain={chain} ca={ca}", flush=True)

    if action == "pf_add":
        add_contract_to_portfolio(chain, ca)
        print(f"[TG CALLBACK] applied action={action} chain={chain} ca={ca}", flush=True)
    elif action == "mon_add":
        add_contract_to_monitoring(chain, ca)
        print(f"[TG CALLBACK] applied action={action} chain={chain} ca={ca}", flush=True)
    elif action == "remove":
        remove_contract_everywhere(chain, ca)
        print(f"[TG CALLBACK] applied action={action} chain={chain} ca={ca}", flush=True)

    return {"ok": True}
