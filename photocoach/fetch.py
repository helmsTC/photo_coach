"""Resumable multi-connection downloader for the training archives.

    python -m photocoach.fetch <url> <dest> [--have=<bytes already valid from a sequential download>]
"""
import json
import os
import sys
import threading
import time
import urllib.request

CHUNK = 48 << 20


def fetch(url, dest, have=0, workers=6):
    req = urllib.request.Request(url, method="HEAD")
    size = int(urllib.request.urlopen(req, timeout=30).headers["Content-Length"])
    state_p = dest + ".chunks.json"
    n = (size + CHUNK - 1) // CHUNK
    done = set(json.load(open(state_p))) if os.path.exists(state_p) else {i for i in range(n) if (i + 1) * CHUNK <= have}
    mode = "r+b" if os.path.exists(dest) else "w+b"
    f = open(dest, mode)
    f.truncate(size)
    todo = [i for i in range(n) if i not in done]
    lock = threading.Lock()
    t0, got = time.time(), [0]

    def work():
        while True:
            with lock:
                if not todo:
                    return
                i = todo.pop(0)
            a, b = i * CHUNK, min(size, (i + 1) * CHUNK) - 1
            for attempt in range(50):
                try:
                    r = urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"}), timeout=30)
                    pos = a
                    while True:
                        buf = r.read(1 << 20)
                        if not buf:
                            break
                        os.pwrite(f.fileno(), buf, pos)
                        pos += len(buf)
                        got[0] += len(buf)
                    if pos != b + 1:
                        raise IOError("short read")
                    with lock:
                        done.add(i)
                        if len(done) % 8 == 0 or not todo:
                            json.dump(sorted(done), open(state_p, "w"))
                            print(f"{os.path.basename(dest)}: {len(done)}/{n} chunks, {got[0] / (time.time() - t0) / 1e6:.1f} MB/s", flush=True)
                    break
                except Exception as e:
                    time.sleep(min(30, 2 + attempt))
            else:
                print(f"chunk {i} failed permanently", flush=True)

    th = [threading.Thread(target=work) for _ in range(workers)]
    [t.start() for t in th]
    [t.join() for t in th]
    json.dump(sorted(done), open(state_p, "w"))
    f.close()
    ok = len(done) == n
    print(f"{dest}: {'complete' if ok else 'INCOMPLETE'} ({size} bytes)", flush=True)
    return ok


if __name__ == "__main__":
    have = next((int(a.split("=")[1]) for a in sys.argv[3:] if a.startswith("--have=")), 0)
    sys.exit(0 if fetch(sys.argv[1], sys.argv[2], have) else 1)
