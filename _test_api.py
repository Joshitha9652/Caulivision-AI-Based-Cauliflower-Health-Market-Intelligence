import json
import urllib.request

import cv2
import numpy as np

img = np.zeros((400, 400, 3), np.uint8)
img[:] = (220, 230, 235)
cv2.circle(img, (200, 200), 140, (240, 245, 248), -1)
ok, buf = cv2.imencode(".jpg", img)
data = buf.tobytes()
boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
body = (
    f"--{boundary}\r\n"
    'Content-Disposition: form-data; name="image"; filename="t.jpg"\r\n'
    "Content-Type: image/jpeg\r\n\r\n"
).encode() + data + f"\r\n--{boundary}--\r\n".encode()
req = urllib.request.Request(
    "http://127.0.0.1:5000/api/analyze",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
)
resp = json.loads(urllib.request.urlopen(req).read().decode())
print(resp["disease"]["name"], resp["quality"]["grade"], resp["advice"]["source"])
print(len(resp["advice"]["precautions"]), "precautions")
print(resp["advice"]["pesticides"][0]["name"])
