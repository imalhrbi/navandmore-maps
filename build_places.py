"""Builds Nav&More's offline place index (saudi-places.navp) from an OpenStreetMap extract.

The file is raw DEFLATE (RFC 1951, what Apple's `NSData.decompressed(using: .zlib)` reads) of
UTF-8 text, one place per line, tab-separated:

    name    english name    category    latitude    longitude    settlement

`category` is "key=value" from OpenStreetMap (e.g. amenity=hospital, place=neighbourhood,
street); `settlement` is the nearest city, town or village, so results read "حي النخيل · الرياض".
Named streets appear once per name and area, not once per OpenStreetMap way.
"""
import math
import sys
import zlib

import osmium

KEYS = ("place", "amenity", "shop", "tourism", "leisure", "healthcare", "office", "aeroway",
        "railway", "public_transport", "historic", "natural", "building", "craft", "landuse")
SKIP = {"amenity": {"parking_space", "bench", "waste_basket", "vending_machine", "parking_entrance",
                    "bicycle_parking", "toilets", "drinking_water", "shelter", "waste_disposal"},
        "building": {"yes", "house", "residential", "apartments", "roof", "garage", "shed", "construction"},
        "landuse": {"residential", "farmland", "grass", "construction", "industrial", "commercial", "retail"},
        "natural": {"tree", "scrub", "sand", "bare_rock", "water", "coastline", "wood"},
        "railway": {"rail", "abandoned", "level_crossing", "switch", "buffer_stop", "signal"},
        "aeroway": {"taxiway", "runway", "apron", "gate", "holding_position", "parking_position"}}
STREETS = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
           "living_street", "service"}
SETTLEMENTS = {"city": 0, "town": 1, "village": 2, "hamlet": 3}


def clean(text):
    return " ".join((text or "").replace("\t", " ").split())


class Places(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.rows = {}
        self.streets = {}
        self.settlements = []  # (rank, lat, lon, name)

    def category(self, tags):
        for key in KEYS:
            value = tags.get(key)
            if value and value not in SKIP.get(key, ()) and value not in ("no", "disused"):
                return f"{key}={value}"
        return None

    def add(self, tags, lat, lon):
        name = clean(tags.get("name:ar") or tags.get("name"))
        if not name:
            return
        place = tags.get("place")
        if place in SETTLEMENTS:
            self.settlements.append((SETTLEMENTS[place], lat, lon, name))
        category = self.category(tags)
        if not category:
            return
        english = clean(tags.get("name:en") or (tags.get("name") if tags.get("name") != name else ""))
        key = (name, category, round(lat, 3), round(lon, 3))
        self.rows.setdefault(key, (name, english, category, lat, lon))

    def node(self, n):
        if n.tags and n.location.valid():
            self.add(n.tags, n.location.lat, n.location.lon)

    def way(self, w):
        tags = w.tags
        points = [(n.location.lat, n.location.lon) for n in w.nodes if n.location.valid()]
        if not points:
            return
        lat = sum(p[0] for p in points) / len(points)
        lon = sum(p[1] for p in points) / len(points)
        highway = tags.get("highway")
        name = clean(tags.get("name:ar") or tags.get("name"))
        if highway in STREETS and name:
            # One entry per street name and ~2 km area: a long road is many ways in OSM.
            key = (name, round(lat * 50), round(lon * 50))
            english = clean(tags.get("name:en") or "")
            self.streets.setdefault(key, (name, english, "street", lat, lon))
            return
        self.add(tags, lat, lon)


def main(source, output):
    places = Places()
    places.apply_file(source, locations=True, idx="flex_mem")
    rows = list(places.rows.values()) + list(places.streets.values())
    print(f"places {len(places.rows):,}  streets {len(places.streets):,}  settlements {len(places.settlements):,}", flush=True)

    # Nearest settlement on a coarse grid; cities count from farther away than villages.
    grid = {}
    for rank, lat, lon, name in places.settlements:
        grid.setdefault((int(lat * 4), int(lon * 4)), []).append((rank, lat, lon, name))
    reach = {0: 30_000, 1: 12_000, 2: 5_000, 3: 2_500}

    def settlement(lat, lon, own):
        best, best_score = "", math.inf
        gy, gx = int(lat * 4), int(lon * 4)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                for rank, slat, slon, name in grid.get((gy + dy, gx + dx), ()):
                    if name == own:
                        continue
                    d = math.hypot((slat - lat) * 110_574, (slon - lon) * 111_320 * math.cos(math.radians(lat)))
                    if d <= reach[rank] and d / reach[rank] < best_score:
                        best, best_score = name, d / reach[rank]
        return best

    lines = []
    for name, english, category, lat, lon in rows:
        lines.append(f"{name}\t{english}\t{category}\t{lat:.6f}\t{lon:.6f}\t{settlement(lat, lon, name)}")
    text = "\n".join(lines).encode("utf-8")
    compressor = zlib.compressobj(9, zlib.DEFLATED, -15)
    with open(output, "wb") as out:
        out.write(compressor.compress(text) + compressor.flush())
    print(f"written {output}: {len(lines):,} entries, {len(text) / 1e6:.1f} MB text", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
