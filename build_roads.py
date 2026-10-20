"""Builds Nav&More's offline road network (saudi-roads.navr) from an OpenStreetMap extract.

The file is a compact, memory-mappable driving graph used by the app for offline routing
and speed limits. All integers are little-endian; every section starts on an 8-byte boundary.

Header (64 bytes): b"NAVR", version, nodeCount, segmentCount, edgeCount, geometryCount,
nameCount, nameByteCount, cellCount, cellEntryCount, then zero padding.
Sections, in order:
  nodes        nodeCount × (int32 lat·1e7, int32 lon·1e7)          road junctions and ends
  firstEdge    (nodeCount + 1) × uint32                            CSR index into edges
  edges        edgeCount × (uint32 target, uint32 segment|0x80000000 when travelled backwards)
  segments     segmentCount × 28 bytes: uint32 a, b, geometryStart, nameIndex, geometryCount;
               float32 length (m); uint8 speed (km/h, for travel time), maxspeed (km/h, 0 = unknown),
               roadClass, flags (1 forward, 2 backward, 4 roundabout, 8 link/ramp, 16 toll)
  geometry     geometryCount × (int32 lat·1e7, int32 lon·1e7)      points between a and b
  nameOffsets  (nameCount + 1) × uint32, then nameByteCount UTF-8 bytes
  cellKeys     cellCount × uint32 (sorted), cellStart (cellCount + 1) × uint32,
  cellSegments cellEntryCount × uint32                             0.01° grid → segments
"""
import math
import re
import struct
import sys
from array import array

import numpy as np
import osmium

CLASSES = ["motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
           "secondary", "secondary_link", "tertiary", "tertiary_link", "unclassified",
           "residential", "living_street", "service", "track", "road"]
DEFAULT_SPEED = {"motorway": 110, "motorway_link": 60, "trunk": 90, "trunk_link": 50,
                 "primary": 70, "primary_link": 40, "secondary": 60, "secondary_link": 35,
                 "tertiary": 50, "tertiary_link": 30, "unclassified": 40, "residential": 30,
                 "living_street": 10, "service": 15, "track": 20, "road": 30}
CELL = 100  # cells per degree (0.01°, about 1.1 km)


def maxspeed_kmh(value):
    if not value:
        return 0
    match = re.match(r"^\s*(\d+(?:\.\d+)?)\s*(mph)?", value)
    if not match:
        return 0
    speed = float(match.group(1)) * (1.609 if match.group(2) else 1)
    return int(round(speed)) if 5 <= speed <= 160 else 0


class Roads(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.refs = array("q")
        self.lat = array("i")
        self.lon = array("i")
        self.starts = array("I")
        self.attrs = []  # (class, speed, maxspeed, flags, name)

    def way(self, w):
        tags = w.tags
        highway = tags.get("highway")
        if highway not in DEFAULT_SPEED or tags.get("area") == "yes":
            return
        access = tags.get("motor_vehicle") or tags.get("motorcar") or tags.get("vehicle") or tags.get("access")
        if access in ("no", "private", "agricultural", "forestry", "delivery"):
            return
        refs, lats, lons = [], [], []
        for node in w.nodes:
            if not node.location.valid():
                continue
            refs.append(node.ref)
            lats.append(int(round(node.location.lat * 1e7)))
            lons.append(int(round(node.location.lon * 1e7)))
        if len(refs) < 2:
            return
        oneway = tags.get("oneway")
        roundabout = tags.get("junction") in ("roundabout", "circular")
        implied = highway in ("motorway", "motorway_link") or roundabout
        if oneway in ("yes", "true", "1") or (implied and oneway != "no"):
            flags = 1
        elif oneway in ("-1", "reverse"):
            flags = 2
        else:
            flags = 3
        if roundabout:
            flags |= 4
        if highway.endswith("_link"):
            flags |= 8
        if tags.get("toll") == "yes":
            flags |= 16
        limit = maxspeed_kmh(tags.get("maxspeed"))
        speed = DEFAULT_SPEED[highway]
        if limit:
            speed = max(5, min(int(limit * 0.9), 130))
        name = tags.get("name:ar") or tags.get("name") or tags.get("ref") or ""
        self.starts.append(len(self.refs))
        self.refs.extend(refs)
        self.lat.extend(lats)
        self.lon.extend(lons)
        self.attrs.append((CLASSES.index(highway), speed, min(limit, 255), flags, name))


def main(source, output):
    roads = Roads()
    roads.apply_file(source, locations=True, idx="flex_mem")
    refs = np.frombuffer(roads.refs, dtype=np.int64)
    lat = np.frombuffer(roads.lat, dtype=np.int32)
    lon = np.frombuffer(roads.lon, dtype=np.int32)
    starts = np.append(np.frombuffer(roads.starts, dtype=np.uint32).astype(np.int64), len(refs))
    print(f"ways {len(roads.attrs):,}  node refs {len(refs):,}", flush=True)

    # A node is a graph vertex when ways share it or a way ends there.
    unique, inverse, counts = np.unique(refs, return_inverse=True, return_counts=True)
    vertex = counts[inverse] >= 2
    vertex[starts[:-1]] = True
    vertex[starts[1:] - 1] = True
    vertex_ids = np.full(len(unique), -1, dtype=np.int64)
    vertex_unique = np.unique(inverse[vertex])
    vertex_ids[vertex_unique] = np.arange(len(vertex_unique))
    node_of = vertex_ids[inverse]
    first_pos = np.full(len(unique), -1, dtype=np.int64)
    first_pos[inverse[::-1]] = np.arange(len(refs))[::-1]
    node_lat = lat[first_pos[vertex_unique]]
    node_lon = lon[first_pos[vertex_unique]]

    # Metres between consecutive points (meaningless across way boundaries, never used there).
    rlat = np.radians(lat / 1e7)
    rlon = np.radians(lon / 1e7)
    dlat = np.diff(rlat)
    dlon = np.diff(rlon)
    h = np.sin(dlat / 2) ** 2 + np.cos(rlat[:-1]) * np.cos(rlat[1:]) * np.sin(dlon / 2) ** 2
    step = 2 * 6371008.8 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
    cumulative = np.concatenate([[0.0], np.cumsum(step)])

    names = {"": 0}
    seg = {k: array("I") for k in ("a", "b", "gs", "gc", "name")}
    seg_len = array("f")
    seg_meta = array("B")  # speed, maxspeed, class, flags per segment
    geo_lat, geo_lon = array("i"), array("i")
    vertex_list = vertex.tolist()
    node_list = node_of.tolist()
    for w, (cls, speed, limit, flags, name) in enumerate(roads.attrs):
        name_index = names.setdefault(name, len(names))
        begin, end = int(starts[w]), int(starts[w + 1])
        segment_start = begin
        for i in range(begin + 1, end):
            if not vertex_list[i]:
                continue
            a, b = node_list[segment_start], node_list[i]
            length = float(cumulative[i] - cumulative[segment_start])
            if a != b or length > 1:
                seg["a"].append(a); seg["b"].append(b)
                seg["gs"].append(len(geo_lat)); seg["gc"].append(i - segment_start - 1)
                seg["name"].append(name_index)
                geo_lat.extend(lat[segment_start + 1:i]); geo_lon.extend(lon[segment_start + 1:i])
                seg_len.append(max(length, 0.1))
                seg_meta.extend((speed, limit, cls, flags))
            segment_start = i
    segments = len(seg_len)
    print(f"vertices {len(node_lat):,}  segments {segments:,}  geometry {len(geo_lat):,}", flush=True)

    a = np.frombuffer(seg["a"], dtype=np.uint32)
    b = np.frombuffer(seg["b"], dtype=np.uint32)
    meta = np.frombuffer(seg_meta, dtype=np.uint8).reshape(-1, 4)
    forward = (meta[:, 3] & 1) != 0
    backward = (meta[:, 3] & 2) != 0
    ids = np.arange(segments, dtype=np.uint32)
    src = np.concatenate([a[forward], b[backward]])
    dst = np.concatenate([b[forward], a[backward]])
    ref = np.concatenate([ids[forward], ids[backward] | np.uint32(0x80000000)])
    order = np.argsort(src, kind="stable")
    src, dst, ref = src[order], dst[order], ref[order]
    first_edge = np.searchsorted(src, np.arange(len(node_lat) + 1)).astype(np.uint32)
    print(f"directed edges {len(src):,}", flush=True)

    # Spatial grid: every cell a segment passes through, sampled about every 0.005°.
    gs = np.frombuffer(seg["gs"], dtype=np.uint32)
    gc = np.frombuffer(seg["gc"], dtype=np.uint32)
    glat = np.frombuffer(geo_lat, dtype=np.int32)
    glon = np.frombuffer(geo_lon, dtype=np.int32)
    keys, owners = array("q"), array("I")
    for s in range(segments):
        start, count = int(gs[s]), int(gc[s])
        pts_lat = [int(node_lat[a[s]])] + glat[start:start + count].tolist() + [int(node_lat[b[s]])]
        pts_lon = [int(node_lon[a[s]])] + glon[start:start + count].tolist() + [int(node_lon[b[s]])]
        cells = set()
        for (y0, x0), (y1, x1) in zip(zip(pts_lat, pts_lon), zip(pts_lat[1:], pts_lon[1:])):
            n = max(1, int(max(abs(y1 - y0), abs(x1 - x0)) / 50000) + 1)  # 0.005° = 50000e-7
            for k in range(n + 1):
                y = (y0 + (y1 - y0) * k / n) / 1e7
                x = (x0 + (x1 - x0) * k / n) / 1e7
                cells.add((int(math.floor(y * CELL)) + 9000) * 36000 + int(math.floor(x * CELL)) + 18000)
        for cell in cells:
            keys.append(cell); owners.append(s)
    keys_np = np.frombuffer(keys, dtype=np.int64)
    owners_np = np.frombuffer(owners, dtype=np.uint32)
    order = np.lexsort((owners_np, keys_np))
    keys_np, owners_np = keys_np[order], owners_np[order]
    cell_keys, cell_first = np.unique(keys_np, return_index=True)
    cell_start = np.append(cell_first, len(keys_np)).astype(np.uint32)
    print(f"grid cells {len(cell_keys):,}  entries {len(keys_np):,}", flush=True)

    name_list = sorted(names, key=names.get)
    encoded = [n.encode("utf-8") for n in name_list]
    name_offsets = np.cumsum([0] + [len(e) for e in encoded]).astype(np.uint32)
    name_bytes = b"".join(encoded)

    segment_records = np.zeros(segments, dtype=np.dtype([
        ("a", "<u4"), ("b", "<u4"), ("gs", "<u4"), ("name", "<u4"), ("gc", "<u4"),
        ("length", "<f4"), ("speed", "u1"), ("maxspeed", "u1"), ("cls", "u1"), ("flags", "u1")]))
    segment_records["a"], segment_records["b"] = a, b
    segment_records["gs"], segment_records["gc"] = gs, gc
    segment_records["name"] = np.frombuffer(seg["name"], dtype=np.uint32)
    segment_records["length"] = np.frombuffer(seg_len, dtype=np.float32)
    segment_records["speed"], segment_records["maxspeed"] = meta[:, 0], meta[:, 1]
    segment_records["cls"], segment_records["flags"] = meta[:, 2], meta[:, 3]
    assert segment_records.dtype.itemsize == 28

    with open(output, "wb") as out:
        def section(data):
            out.write(data)
            out.write(b"\0" * (-out.tell() % 8))
        header = struct.pack("<4s9I", b"NAVR", 1, len(node_lat), segments, len(src), len(glat),
                             len(name_list), len(name_bytes), len(cell_keys), len(keys_np))
        section(header + b"\0" * (64 - len(header)))
        section(np.column_stack([node_lat, node_lon]).astype("<i4").tobytes())
        section(first_edge.astype("<u4").tobytes())
        section(np.column_stack([dst, ref]).astype("<u4").tobytes())
        section(segment_records.tobytes())
        section(np.column_stack([glat, glon]).astype("<i4").tobytes())
        section(name_offsets.astype("<u4").tobytes())
        section(name_bytes)
        section(cell_keys.astype("<u4").tobytes())
        section(cell_start.tobytes())
        section(owners_np.astype("<u4").tobytes())
    print("written", output, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
