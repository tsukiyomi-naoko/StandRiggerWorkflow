"""Draw a simple synthetic full-body character (no third-party artwork) for CPU tests."""
import sys
from PIL import Image, ImageDraw

W, H = 900, 1400
im = Image.new("RGB", (W, H), (255, 255, 255))
d = ImageDraw.Draw(im)
cx = W // 2
d.ellipse([cx - 190, 110, cx + 190, 560], fill=(70, 60, 140))           # back hair
d.rectangle([cx - 40, 430, cx + 40, 520], fill=(250, 220, 200))         # neck
d.polygon([(cx - 170, 500), (cx + 170, 500), (cx + 200, 900), (cx - 200, 900)], fill=(60, 120, 200))  # top
d.rectangle([cx - 250, 520, cx - 180, 860], fill=(250, 220, 200))       # arm R (character's right)
d.rectangle([cx + 180, 520, cx + 250, 860], fill=(250, 220, 200))       # arm L
d.polygon([(cx - 200, 880), (cx + 200, 880), (cx + 260, 1080), (cx - 260, 1080)], fill=(40, 40, 60))  # skirt
d.rectangle([cx - 120, 1080, cx - 50, 1330], fill=(240, 240, 240), outline=(0, 0, 0))  # legs
d.rectangle([cx + 50, 1080, cx + 120, 1330], fill=(240, 240, 240), outline=(0, 0, 0))
d.ellipse([cx - 150, 160, cx + 150, 470], fill=(250, 222, 205), outline=(40, 30, 30), width=3)  # face
for s in (-1, 1):
    ex = cx + s * 65
    d.ellipse([ex - 35, 280, ex + 35, 345], fill=(255, 255, 255), outline=(30, 20, 20), width=3)  # eye white
    d.ellipse([ex - 20, 288, ex + 20, 340], fill=(40, 120, 160))                                  # iris
    d.line([ex - 40, 255, ex + 35, 250 - s * 2], fill=(50, 40, 40), width=6)                      # brow
d.arc([cx - 30, 380, cx + 30, 410], 10, 170, fill=(150, 40, 60), width=4)                          # mouth
d.pieslice([cx - 170, 120, cx + 170, 330], 180, 360, fill=(90, 80, 170))                           # bangs
im.save(sys.argv[1] if len(sys.argv) > 1 else "standrig_test_character.png")
