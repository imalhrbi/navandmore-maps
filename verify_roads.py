"""Sanity checks for saudi-roads.navr: parses every section and routes a few real trips."""
import heapq
import math
import struct
import sys

import numpy as np


def load(path):
    data = np.fromfile(path, dtype=np.uint8)
    magic, version, nodes, segments, edges, geometry, names, name_bytes, cells, entries = \
        struct.unpack_from("<4s9I", data, 0)
    assert magic == b"NAVR" and version == 1, (magic, version)
    offset = 64
    out = {}

    def take(name, count, dtype, width=1):
        nonlocal offset
        size = count * np.dtype(dtype).itemsize * width
        out[name] = data[offset:offset + size].view(dtype).reshape(-1, width) if width > 1 else data[offset:offset + size].view(dtype)
        offset += size + (-size % 8)

    take("nodes", nodes, "<i4", 2)
    take("first", nodes + 1, "<u4")
    take("edges", edges, "<u4", 2)
    segment_type = np.dtype([("a", "<u4"), ("b", "<u4"), ("gs", "<u4"), ("name", "<u4"), ("gc", "<u4"),
                             ("length", "<f4"), ("speed", "u1"), ("maxspeed", "u1"), ("cls", "u1"), ("flags", "u1")])
    size = segments * 28
    out["segments"] = data[offset:offset + size].view(segment_type)
    offset += size + (-size % 8)
    take("geometry", geometry, "<i4", 2)
    take("nameOffsets", names + 1, "<u4")
    take("nameBytes", name_bytes, "u1")
    take("cellKeys", cells, "<u4")
    take("cellStart", cells + 1, "<u4")
    take("cellSegments", entries, "<u4")
    assert offset == len(data), (offset, len(data))
    assert out["first"][-1] == edges
    assert np.all(np.diff(out["cellKeys"].astype(np.int64)) > 0)
    print(f"nodes {nodes:,} segments {segments:,} edges {edges:,} geometry {geometry:,} names {names:,} cells {cells:,}")
    known = out["segments"]["maxspeed"] > 0
    print(f"segments with a posted speed limit: {known.mean() * 100:.1f}%")
    return out


def nearest(net, lat, lon):
    pts = net["nodes"] / 1e7
    d = (pts[:, 0] - lat) ** 2 + ((pts[:, 1] - lon) * math.cos(math.radians(lat))) ** 2
    return int(np.argmin(d))


def route(net, origin, destination):
    first, edges, segs = net["first"], net["edges"], net["segments"]
    seconds = segs["length"] / (segs["speed"].astype(np.float64) / 3.6)
    start, goal = nearest(net, *origin), nearest(net, *destination)
    best = {start: 0.0}
    meters = {start: 0.0}
    heap = [(0.0, start)]
    while heap:
        cost, node = heapq.heappop(heap)
        if node == goal:
            return cost, meters[node]
        if cost > best.get(node, math.inf):
            continue
        for e in range(first[node], first[node + 1]):
            target, ref = int(edges[e, 0]), int(edges[e, 1])
            s = ref & 0x7FFFFFFF
            candidate = cost + seconds[s]
            if candidate < best.get(target, math.inf):
                best[target] = candidate
                meters[target] = meters[node] + float(segs["length"][s])
                heapq.heappush(heap, (candidate, target))
    raise SystemExit(f"no route from {origin} to {destination}")


if __name__ == "__main__":
    net = load(sys.argv[1])
    trips = {
        "Riyadh → Jeddah": ((24.7136, 46.6753), (21.5433, 39.1728)),
        "Riyadh (Olaya → King Khalid Airport)": ((24.6900, 46.6850), (24.9576, 46.6988)),
        "Dammam → Al Khobar": ((26.4207, 50.0888), (26.2794, 50.2083)),
    }
    for title, (a, b) in trips.items():
        seconds, meters = route(net, a, b)
        straight = 6371 * math.acos(min(1, math.sin(math.radians(a[0])) * math.sin(math.radians(b[0])) +
            math.cos(math.radians(a[0])) * math.cos(math.radians(b[0])) * math.cos(math.radians(b[1] - a[1]))))
        print(f"{title}: {meters / 1000:.0f} km by road ({straight:.0f} km straight), {seconds / 3600:.1f} h")
        assert straight * 1000 <= meters <= straight * 1000 * 2.5, "implausible route length"
