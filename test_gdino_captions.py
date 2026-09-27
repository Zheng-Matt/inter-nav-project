import base64
import cv2
import requests

IMAGE = "/data/why/inter-nav-tvtest/tv_test.png"
URL = "http://127.0.0.1:12181/gdino"

img = cv2.imread(IMAGE)
if img is None:
    raise RuntimeError(f"Cannot read image: {IMAGE}")

ok, buf = cv2.imencode(".jpg", img)
if not ok:
    raise RuntimeError("Failed to encode image")

image_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

tests = {
    "A_monitor": "monitor",
    "B_monitor_screen": "monitor . screen",
    "C_tv_group": "television . monitor . screen",
    "D_full_context": (
        "chair . table . sofa . door . cabinet . bed . lamp . "
        "television . refrigerator . plant . person . monitor . "
        "screen . picture . mirror . window . bookshelf . curtain . microwave"
    ),
}

for name, caption in tests.items():
    print("\n" + "=" * 80)
    print(name)
    print("caption:", caption)

    r = requests.post(
        URL,
        json={
            "image": image_b64,
            "caption": caption,
        },
        timeout=60,
    )

    print("HTTP:", r.status_code)
    r.raise_for_status()

    result = r.json()

    boxes = result.get("boxes", [])
    scores = result.get("scores", result.get("logits", []))
    labels = result.get("labels", result.get("phrases", []))

    print("num_boxes:", len(boxes))

    for i, box in enumerate(boxes):
        label = labels[i] if i < len(labels) else None
        score = scores[i] if i < len(scores) else None
        print(
            f"[{i}] label={label!r} "
            f"score={score} "
            f"bbox={box}"
        )
