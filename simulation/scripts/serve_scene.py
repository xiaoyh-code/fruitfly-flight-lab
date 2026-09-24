#!/usr/bin/env python3
"""Loopback-only 3DGS viewer; serves only explicit local scene assets."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
import argparse
import shutil

ROOT=Path(__file__).resolve().parents[1]
FILES={
    "/": (ROOT/"dashboard/scene.html","text/html; charset=utf-8"),
    "/scene/robot_hall.ply":(ROOT/"models/gaussian/robot_hall.ply","application/octet-stream"),
    "/lib/three.module.js":(ROOT/"dashboard/lib/three.module.js","text/javascript"),
    "/lib/gaussian-splats-3d.module.js":(ROOT/"dashboard/lib/gaussian-splats-3d.module.js","text/javascript"),
    "/lib/crazyflie-scene.js":(ROOT/"dashboard/lib/crazyflie-scene.js","text/javascript"),
    "/lib/hall-replay.js":(ROOT/"dashboard/lib/hall-replay.js","text/javascript"),
    "/assets/crazyflie.json":(ROOT/"dashboard/assets/crazyflie.json","application/json; charset=utf-8"),
    "/data/hall-replay.json":(ROOT/"outputs/scene-replay/hall-replay.json","application/json; charset=utf-8"),
}
FILES["/library/manifest.json"] = (ROOT / "models/gaussian/library/manifest.json", "application/json; charset=utf-8")
for filename in ("sakura1.ply", "kitune1.ply", "kouen2.ply", "dajing-preview.ply", "dajing-detail.ply", "dajing-full.ply"):
    FILES["/library/" + filename] = (ROOT / "models/gaussian/library" / filename, "application/octet-stream")
FILES["/library/dajing-LICENSE.txt"] = (ROOT / "models/gaussian/library/dajing-LICENSE.txt", "text/plain; charset=utf-8")

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.serve_asset()

    def do_HEAD(self):
        self.serve_asset(head_only=True)

    def serve_asset(self, head_only=False):
        route=urlsplit(self.path).path
        if route not in FILES:
            self.send_error(404);return
        path,kind=FILES[route]
        try:stream=path.open("rb")
        except FileNotFoundError:self.send_error(503,"Scene asset not ready");return
        with stream:
            self.send_response(200);self.send_header("Content-Type",kind)
            self.send_header("Content-Length",str(path.stat().st_size))
            self.send_header("Cache-Control","no-cache")
            self.send_header("X-Content-Type-Options","nosniff")
            self.end_headers()
            if not head_only:
                try:shutil.copyfileobj(stream,self.wfile,length=1024*1024)
                except (BrokenPipeError,ConnectionResetError):pass

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--port",type=int,default=8766)
    args=parser.parse_args()
    server=ThreadingHTTPServer(("127.0.0.1",args.port),Handler)
    print(f"3DGS scene viewer: http://127.0.0.1:{args.port}",flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()

if __name__=="__main__":main()
