import json
import subprocess


def _node_eval(script: str) -> dict:
    proc = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, check=True)
    return json.loads(proc.stdout.strip())


def test_d1_worker_rejects_unsafe_identifiers():
    script = r'''
import workerMod from "./cloudflare/d1_worker.js";
const db = { prepare(){ return { bind(){ return { all: async()=>({results:[]}), first: async()=>null, run: async()=>({meta:{changes:0}}) }; } }; } };
const env = { D1_PROXY_TOKEN: "t", DB: db };
const req = new Request("https://x/v1/table/runtime_state?select=state_json;drop&limit=1", {headers:{authorization:"Bearer t"}});
const res = await workerMod.fetch(req, env);
console.log(JSON.stringify({status: res.status}));
'''
    out = _node_eval(script)
    assert out["status"] == 500 or out["status"] == 400


def test_d1_worker_accepts_safe_select_list_and_json_roundtrip():
    script = r'''
import workerMod from "./cloudflare/d1_worker.js";
const db = {
  prepare(sql){
    return {
      bind(...args){
        return {
          all: async()=>({results:[{state_json:'{"a":1}', updated_ts:'x'}]}),
          first: async()=>null,
          run: async()=>({meta:{changes:1}, sql, args}),
        };
      }
    };
  }
};
const env = { D1_PROXY_TOKEN: "t", DB: db };
const req = new Request("https://x/v1/table/runtime_state?select=state_json,updated_ts&limit=1", {headers:{authorization:"Bearer t"}});
const res = await workerMod.fetch(req, env);
const body = await res.json();
console.log(JSON.stringify({status: res.status, parsed: body.rows[0].state_json.a}));
'''
    out = _node_eval(script)
    assert out["status"] == 200
    assert out["parsed"] == 1
