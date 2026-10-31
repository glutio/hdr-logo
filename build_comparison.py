import base64
import hashlib
from html.parser import HTMLParser
from pathlib import Path


root = Path(__file__).resolve().parent
page = (root / "comparison.template.html").read_text(encoding="utf-8")
assets = {
    "__SDR__": root / "start_logo_sdr_preview.png",
    "__HDR__": root / "start_logo_hdr.jpg",
    "__CLEAN__": root / "start_logo_clean.png",
}
for marker, path in assets.items():
    page = page.replace(marker, base64.b64encode(path.read_bytes()).decode("ascii"))
assert not any(marker in page for marker in assets)


class VerifyPage(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images = []
        self.downloads = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "img":
            self.images.append(attrs["src"])
        if tag == "a" and "download" in attrs:
            self.downloads.append(attrs["href"])


output = root / "index.html"
output.write_text(page, encoding="utf-8")
parser = VerifyPage()
parser.feed(output.read_text(encoding="utf-8"))
assert len(parser.images) == 2 and len(parser.downloads) == 3
assert parser.images == parser.downloads[:2]
for embedded, original in zip(parser.downloads, assets.values()):
    restored = base64.b64decode(embedded.split(",", 1)[1], validate=True)
    assert restored == original.read_bytes(), "Embedded image bytes changed"
    print(f"Verified {original.name}: SHA256 {hashlib.sha256(restored).hexdigest()}")
print(f"Created {output} ({output.stat().st_size:,} bytes)")
