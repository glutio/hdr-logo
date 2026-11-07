import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path


root = Path(__file__).resolve().parent
page = (root / "comparison.template.html").read_text(encoding="utf-8")
release = root / "logo-final-2026-11-06"
manifest = json.loads((release / "assets" / "manifest.json").read_text(encoding="utf-8"))
assets = {
    "__STANDARD__": "logo.png",
    "__FLAT__": "logo-flat-color.png",
    "__RIM_HDR__": "logo-inner-rim-hdr.jpg",
    "__SYMBOLS_HDR__": "logo-inner-rim-symbols-hdr.jpg",
}
for marker, filename in assets.items():
    path = release / "assets" / filename
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == manifest["assets_sha256"][filename], f"Final asset changed: {filename}"
    page = page.replace(marker, f"{release.name}/assets/{filename}?v={digest[:12]}")
archive = root / f"{release.name}.zip"
archive_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
page = page.replace("__PACKAGE__", f"{archive.name}?v={archive_digest[:12]}")
assert not any(marker in page for marker in assets)
assert "__PACKAGE__" not in page


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
assert len(parser.images) == 4 and len(parser.downloads) == 5
assert parser.images == parser.downloads[1:]
for url in parser.downloads:
    path = root.joinpath(*url.split("?", 1)[0].split("/"))
    assert path.is_file(), f"Broken download: {url}"
    print(f"Verified {path.name}: SHA256 {hashlib.sha256(path.read_bytes()).hexdigest()}")
print(f"Created {output} ({output.stat().st_size:,} bytes)")
