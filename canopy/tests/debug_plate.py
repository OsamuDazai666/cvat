"""Compare plate-reader pieces on one plates task frame with the plates_v1 label (run on Windows)."""
import base64, json, sqlite3, sys, urllib.request
sys.path[:0] = ["D:/canopy-evidence/tools/plates", "D:/canopy-evidence/tools"]
import cv2
from read import OCR, read_ocr, parse_raw
con = sqlite3.connect("file:D:/canopy-evidence/curated/index.sqlite?mode=ro", uri=True)
ids = [r for r in con.execute("select id, img_path from items where kind='plate' and img_path is not null and gate='gate1' order by ts limit 10")]
def post(p, b):
    r = urllib.request.Request("http://127.0.0.1:9090" + p, data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r).read())
ocr = OCR("http://127.0.0.1:9012/v1", "glm", "Text Recognition:")
for iid, rel in ids[:int(sys.argv[1]) if len(sys.argv) > 1 else 4]:
    path = "D:/canopy-evidence/training/" + rel
    img = cv2.imread(path); b64 = base64.b64encode(open(path, "rb").read()).decode()
    gold = con.execute("select x1,y1,x2,y2,text,extra from labels where item_id=? and source='plates_v1' order by obj", (iid,)).fetchall()
    det = post("/plate/detect", {"image": b64, "conf": 0.4})["boxes"]
    print(iid, "plates_v1:", [(g[4], json.loads(g[5]).get("origin"), json.loads(g[5]).get("status"), [round(v) for v in g[:4]]) for g in gold])
    for d in det:
        o = post("/plate/ocr", {"image": b64, "boxes": [d["bbox"]]})["reads"][0]
        g = read_ocr(ocr, img, d["bbox"])
        print("   det", [round(v) for v in d["bbox"]], d["score"], "| onnx", o["text"], o.get("region"), o.get("region_p"),
              "parsed", parse_raw(o["text"]), "| glm", g["raw"].replace("\n", " "), g["number"], g["code"], g["origin"])
