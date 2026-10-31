"""Round-trip test for psd_writer: write a PSD, read it back with psd-tools and
(if node + ag-psd are available) with ag-psd, the parser StandRig uses."""
import json, os, subprocess, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "custom_nodes", "ComfyUI-StandRig-Bridge"))
from psd_writer import PsdLayer, write_psd, packbits_rows  # noqa: E402


def unpackbits(data: bytes, n: int) -> bytes:
    out, i = bytearray(), 0
    while len(out) < n:
        h = data[i]; i += 1
        if h < 128:
            out += data[i:i + h + 1]; i += h + 1
        elif h > 128:
            out += bytes([data[i]]) * (257 - h); i += 1
    return bytes(out)


def test_packbits_roundtrip():
    rng = np.random.default_rng(0)
    rows = [np.zeros(300, np.uint8), rng.integers(0, 255, 517).astype(np.uint8),
            np.repeat(rng.integers(0, 255, 40).astype(np.uint8), rng.integers(1, 9, 40)),
            np.array([5], np.uint8), np.array([1, 1], np.uint8), np.array([1, 2, 2, 2, 3, 3], np.uint8)]
    for r in rows:
        counts, data = packbits_rows(r[None, :])
        assert unpackbits(data, r.size) == r.tobytes(), r


def make_layers():
    w, h = 64, 48
    body = np.zeros((h, w, 4), np.uint8); body[10:40, 20:44] = (200, 50, 50, 255)
    face = np.zeros((20, 16, 4), np.uint8); face[2:18, 2:14] = (250, 220, 200, 255)
    mouth_a = np.zeros((6, 6, 4), np.uint8); mouth_a[1:5, 1:5] = (120, 0, 30, 255)
    return w, h, [
        PsdLayer("Expressions/Mouth A (あ)", mouth_a, left=29, top=20, visible=False),
        PsdLayer("Head/Face", face, left=24, top=6),
        PsdLayer("Body", body, opacity=0.5),
    ]


def test_psd_tools_and_agpsd():
    w, h, layers = make_layers()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.psd")
        write_psd(p, w, h, layers)
        from psd_tools import PSDImage
        psd = PSDImage.open(p)
        top_down = list(psd)[::-1]  # psd-tools iterates bottom-up
        names = [(l.name, l.is_group(), l.visible) for l in top_down]
        assert names == [("Expressions", True, True), ("Head", True, True), ("Body", False, True)], names
        exp = top_down[0]
        mouth = list(exp)[0]
        assert mouth.name == "Mouth A (あ)" and not mouth.visible
        assert mouth.bbox == (30, 21, 34, 25), mouth.bbox  # trimmed to opaque pixels
        face = list(top_down[1])[0]
        assert face.bbox == (26, 8, 38, 24)
        assert abs(top_down[2].opacity - 128) <= 1
        arr = np.asarray(face.topil())
        assert tuple(arr[5, 5]) == (250, 220, 200, 255)

        agdir = os.environ.get("AGPSD_DIR")
        if agdir:
            out = subprocess.check_output(["node", os.path.join(HERE, "read_psd_agpsd.mjs"), p], cwd=agdir)
            tree = json.loads(out)
            top = tree["children"]  # ag-psd lists children bottom-up
            assert [c["name"] for c in top] == ["Body", "Head", "Expressions"], top
            m = top[2]["children"][0]
            assert m["name"] == "Mouth A (あ)" and m["hidden"] is True
            assert (m["left"], m["top"], m["right"], m["bottom"]) == (30, 21, 34, 25)
            assert m["firstOpaqueRGBA"] == [120, 0, 30, 255]
            assert abs(top[0]["opacity"] - 0.5) < 0.01
            print("ag-psd OK")


if __name__ == "__main__":
    test_packbits_roundtrip(); test_psd_tools_and_agpsd(); print("psd_writer tests passed")
