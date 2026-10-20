"""Sanity checks for saudi-places.navp: decodes it and looks up well-known places."""
import sys
import zlib

data = zlib.decompress(open(sys.argv[1], "rb").read(), -15).decode("utf-8")
rows = [line.split("\t") for line in data.split("\n")]
assert all(len(r) == 6 for r in rows), "malformed line"
for r in rows:
    lat, lon = float(r[3]), float(r[4])
    assert -90 <= lat <= 90 and -180 <= lon <= 180
print(f"{len(rows):,} entries; with a settlement: {sum(1 for r in rows if r[5]) / len(rows) * 100:.0f}%")
for query in ("مطار الملك خالد", "الرياض", "جدة", "النخيل", "مستشفى"):
    hits = [r for r in rows if query in r[0]]
    print(f"{query}: {len(hits):,} — e.g. " + " | ".join(f"{r[0]} ({r[2]}, {r[5]})" for r in hits[:3]))
    assert hits, f"no result for {query}"
