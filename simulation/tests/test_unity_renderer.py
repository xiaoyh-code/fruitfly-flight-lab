"""Protocol checks protect policy evaluation from stale or corrupt frames."""
import json
from pathlib import Path
import socket
import struct
import sys
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import cv2
import numpy as np
from fruitfly_sim.unity_renderer import UnitySplatRenderer


class BridgeTest(unittest.TestCase):
    def response(self, response_id=1):
        listener = socket.socket()
        listener.bind(("127.0.0.1",0)); listener.listen(1)
        port = listener.getsockname()[1]
        def serve():
            connection, _ = listener.accept()
            with connection:
                line = connection.makefile("rb").readline()
                request = json.loads(line)
                self.assertEqual(request["id"], 1)
                # Pure BGR blue must emerge as RGB blue, not red.
                image = np.zeros((240,320,3), np.uint8); image[:,:,0]=255
                png = cv2.imencode(".png",image)[1].tobytes()
                header = json.dumps(dict(id=response_id,width=320,height=240,format="png")).encode()
                packet = struct.pack("<I",len(header))+header+struct.pack("<I",len(png))+png
                # Deliberately fragmented stream; recv is not packet aligned.
                for start in range(0,len(packet),71):
                    connection.sendall(packet[start:start+71])
            listener.close()
        thread=threading.Thread(target=serve); thread.start()
        return port, thread

    def test_fragmented_image_and_rgb(self):
        port, thread = self.response()
        with UnitySplatRenderer(port=port) as client:
            image = client.render([0,0,1],[1,0,0],[0,0,1])
            np.testing.assert_array_equal(image[0,0],[0,0,255])
        thread.join()

    def test_reject_stale_frame(self):
        port, thread = self.response(response_id=0)
        with UnitySplatRenderer(port=port) as client:
            with self.assertRaisesRegex(RuntimeError,"mismatched"):
                client.render([0,0,1],[1,0,0],[0,0,1])
        thread.join()


if __name__ == "__main__": unittest.main()
