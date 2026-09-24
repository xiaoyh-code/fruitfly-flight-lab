"""Protocol-only regression for an abandoned browser poll after page reload.

The synthetic PNG below verifies transport, never visual rendering. Actual 3DGS
render inspection is recorded separately in outputs/realistic/obstacle*.png.
"""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import struct
import sys
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import zlib

import pytest


SPEC = importlib.util.spec_from_file_location(
    "render_bridge_regression", Path(__file__).resolve().parents[1] / "scripts/serve_render_bridge.py"
)
BRIDGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BRIDGE  # dataclasses resolves annotations through the module registry.
SPEC.loader.exec_module(BRIDGE)


def synthetic_png():
    def chunk(kind, payload):
        return (struct.pack(">I", len(payload)) + kind + payload
                + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", 320, 240, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress((b"\0" + b"\x11\x22\x33" * 320) * 240))
            + chunk(b"IEND", b""))


def test_abandoned_poll_is_reoffered_and_stale_lease_cannot_complete_frame(monkeypatch):
    monkeypatch.setattr(BRIDGE, "OFFER_LEASE_SECONDS", 0.03)
    monkeypatch.setattr(BRIDGE, "FRAME_TIMEOUT", 3.0)
    server = BRIDGE.ThreadingHTTPServer(("127.0.0.1", 0), BRIDGE.Handler)
    server.daemon_threads = True
    server.bridge = BRIDGE.Bridge()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"

    def get_json(path):
        with urlopen(base + path, timeout=2) as response:
            return json.load(response)

    def post(path, payload, content_type="application/json"):
        body = json.dumps(payload).encode() if isinstance(payload, dict) else payload
        with urlopen(Request(base + path, data=body, headers={"Content-Type": content_type}), timeout=5) as response:
            return response.status, response.headers, response.read()

    def frame_path(offer):
        return f"/frame?id={offer['id']}&lease_token={offer['lease_token']}"

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(post, "/render", {
                "id": "reload-frame-1", "cam_pos": [0, 0, 1.5],
                "cam_forward": [1, 0, 0], "cam_up": [0, 0, 1],
            })
            abandoned = get_json("/next")  # Old tab received the offer, then vanished without a claim.
            replacement = get_json("/next")
            assert replacement["id"] == abandoned["id"] == "reload-frame-1"
            assert replacement["lease_token"] != abandoned["lease_token"]
            assert not pending.done()

            with pytest.raises(HTTPError) as stale_claim:
                post("/claim", abandoned)
            assert stale_claim.value.code == 409

            status, _, body = post("/claim", replacement)
            assert status == 200 and json.loads(body)["claimed"] is True
            with pytest.raises(HTTPError) as duplicate_claim:
                post("/claim", replacement)
            assert duplicate_claim.value.code == 409

            png = synthetic_png()
            with pytest.raises(HTTPError) as stale_frame:
                post(frame_path(abandoned), png, "image/png")
            assert stale_frame.value.code == 409
            assert not pending.done()

            # A corrupt PNG from the current lease must not unblock the caller.
            with pytest.raises(HTTPError) as corrupt_frame:
                post(frame_path(replacement), png[:-4], "image/png")
            assert corrupt_frame.value.code == 400
            assert not pending.done()

            assert post(frame_path(replacement), png, "image/png")[0] == 200
            status, headers, received = pending.result(timeout=2)
            assert status == 200 and headers["X-Frame-Id"] == replacement["id"]
            assert headers["Content-Type"] == "image/png" and received == png

            with pytest.raises(HTTPError) as duplicate_frame:
                post(frame_path(replacement), png, "image/png")
            assert duplicate_frame.value.code == 409
            health = get_json("/health")
            assert health["completed"] == 1 and health["failures"] == 0
            assert health["reoffers"] == 1 and health["pending"] == 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
