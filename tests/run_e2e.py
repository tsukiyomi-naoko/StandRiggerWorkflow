"""Queue a workflow (API format) on a running ComfyUI and wait for it.

    python tests/run_e2e.py workflow.api.json [http://127.0.0.1:8188]

Exits non-zero on validation or execution errors and prints node errors.
"""
import json, sys, time, urllib.request, urllib.error, uuid

path = sys.argv[1]
url = (sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8188").rstrip("/")
prompt = json.load(open(path, encoding="utf-8"))
body = json.dumps({"prompt": prompt, "client_id": str(uuid.uuid4())}).encode()
req = urllib.request.Request(url + "/prompt", data=body, headers={"Content-Type": "application/json"})
try:
    res = json.load(urllib.request.urlopen(req))
except urllib.error.HTTPError as e:
    print("VALIDATION FAILED:", e.read().decode()[:4000]); sys.exit(2)
pid = res["prompt_id"]
print("queued", pid)
t0 = time.time()
while time.time() - t0 < 1800:
    h = json.load(urllib.request.urlopen(f"{url}/history/{pid}"))
    if pid in h:
        st = h[pid]["status"]
        print("status:", st.get("status_str"), "completed:", st.get("completed"), f"{time.time() - t0:.1f}s")
        for m in st.get("messages", []):
            if m[0] == "execution_error":
                e = m[1]
                print("ERROR in node", e.get("node_id"), e.get("node_type"), e.get("exception_message"))
                print("".join(e.get("traceback", []))[-3000:])
                sys.exit(1)
        outs = h[pid].get("outputs", {})
        for nid, o in outs.items():
            if "text" in o:
                print(nid, "text:", o["text"])
        sys.exit(0 if st.get("completed") else 1)
    time.sleep(2)
print("timeout"); sys.exit(3)
