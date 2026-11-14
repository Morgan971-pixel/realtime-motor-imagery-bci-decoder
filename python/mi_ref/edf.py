"""A strict reader (and a small writer, for tests) for EDF and EDF+ files.

Layout: a 256-byte fixed header, then 256 bytes per signal stored field by field (all labels,
then all units, ...), then the data records. Every record holds, for each signal in order,
that signal's samples-per-record count of little-endian int16 values. A digital value d becomes
the physical value (d - dmin) * (pmax - pmin) / (dmax - dmin) + pmin.

EDF+ adds a signal labelled "EDF Annotations" whose bytes are time-stamped annotation lists
(TALs): onset [0x15 duration] 0x14 text 0x14 [text 0x14 ...] 0x00, then zero padding. The
first TAL of every record has an empty text and gives the record's start time.

Refused with ValueError, never fixed silently: a version other than "0"; a header size other
than 256 * (signals + 1); a discontinuous file (EDF+D); an unknown record count (-1) or a
non-positive record duration; a header number that is not plain decimal ASCII; a signal with
no samples per record, dmin >= dmax or pmin == pmax; two signals with the same label; a file
size other than header + records * record size, so truncation and trailing bytes are both
caught; a digital value outside [dmin, dmax]; an annotation list that does not parse, is not
UTF-8, or whose record start time disagrees with record index * record duration.
"""
import math
import re
from dataclasses import dataclass

import numpy as np

ANNOTATIONS_LABEL = "EDF Annotations"
_INT = re.compile(r"[+-]?\d+")
_NUM = re.compile(r"[+-]?(\d+(\.\d*)?|\.\d+)")
_TAL = re.compile(rb"([+-]\d+(?:\.\d+)?)(?:\x15(\d+(?:\.\d+)?))?\x14((?:[^\x00\x14]*\x14)*)\x00")
_FIELDS = (("label", 16), ("transducer", 80), ("unit", 8), ("pmin", 8), ("pmax", 8),
           ("dmin", 8), ("dmax", 8), ("prefilter", 80), ("nsamp", 8), ("reserved", 32))


@dataclass(frozen=True)
class Annotation:
    onset: float  # seconds from the start of the file
    duration: float | None
    text: str


@dataclass(frozen=True)
class Edf:
    labels: tuple[str, ...]  # ordinary signals only, trailing spaces removed
    units: tuple[str, ...]
    fs: tuple[float, ...]  # samples per second, per signal
    signals: tuple[np.ndarray, ...]  # physical values, float64
    annotations: tuple[Annotation, ...]  # record time-keeping entries left out


def _ascii(raw: bytes, what: str) -> str:
    try:
        return raw.decode("ascii").rstrip(" ")
    except UnicodeDecodeError:
        raise ValueError(f"{what} is not ASCII") from None


def _int(raw: bytes, what: str) -> int:
    text = _ascii(raw, what).strip()
    if not _INT.fullmatch(text):
        raise ValueError(f"{what} must be an integer, got {text!r}")
    return int(text)


def _num(raw: bytes, what: str) -> float:
    text = _ascii(raw, what).strip()
    if not _NUM.fullmatch(text):
        raise ValueError(f"{what} must be a decimal number, got {text!r}")
    return float(text)


def _tals(raw: bytes, record: int, start: float) -> list[Annotation]:
    """The annotations in one record's annotation bytes, after checking its time-keeping TAL."""
    out: list[Annotation] = []
    pos, first = 0, True
    while pos < len(raw) and raw[pos] != 0:
        m = _TAL.match(raw, pos)
        if m is None:
            raise ValueError(f"record {record}: malformed annotation at byte {pos}")
        try:
            texts = [t.decode("utf-8") for t in m[3].split(b"\x14")[:-1]]
        except UnicodeDecodeError:
            raise ValueError(f"record {record}: annotation text is not UTF-8") from None
        onset = float(m[1])
        duration = float(m[2]) if m[2] is not None else None
        if first:
            if not texts or texts[0] != "":
                raise ValueError(f"record {record}: first annotation is not a time-keeping entry")
            if abs(onset - start) > 1e-9:
                raise ValueError(f"record {record}: starts at {onset} s, expected {start} s")
            texts = texts[1:]
            first = False
        out.extend(Annotation(onset, duration, t) for t in texts)
        pos = m.end()
    if first:
        raise ValueError(f"record {record}: no time-keeping annotation")
    if any(raw[pos:]):
        raise ValueError(f"record {record}: nonzero bytes after the last annotation")
    return out


def read_edf(data: bytes) -> Edf:
    if len(data) < 256:
        raise ValueError(f"file is {len(data)} bytes, shorter than the 256-byte EDF header")
    if _ascii(data[0:8], "version") != "0":
        raise ValueError(f"version must be '0', got {data[0:8]!r}")
    header_bytes = _int(data[184:192], "header size")
    reserved = _ascii(data[192:236], "reserved field")
    n_records = _int(data[236:244], "record count")
    duration = _num(data[244:252], "record duration")
    ns = _int(data[252:256], "signal count")
    if ns < 1:
        raise ValueError(f"signal count must be >= 1, got {ns}")
    if header_bytes != 256 * (ns + 1):
        raise ValueError(f"header size {header_bytes} != 256 * ({ns} + 1)")
    if reserved.startswith("EDF+D"):
        raise ValueError("discontinuous EDF+D files are not supported")
    if n_records < 1:
        raise ValueError(f"record count must be >= 1, got {n_records}")
    if not (math.isfinite(duration) and duration > 0):
        raise ValueError(f"record duration must be > 0, got {duration}")
    if len(data) < header_bytes:
        raise ValueError(f"file is {len(data)} bytes, shorter than its {header_bytes}-byte header")

    fields: dict[str, list[bytes]] = {}
    pos = 256
    for name, width in _FIELDS:
        fields[name] = [data[pos + i * width:pos + (i + 1) * width] for i in range(ns)]
        pos += ns * width
    labels = [_ascii(raw, f"signal {i} label") for i, raw in enumerate(fields["label"])]
    if len(set(labels)) != ns:
        raise ValueError(f"two signals share a label: {labels}")
    nsamp = [_int(raw, f"{labels[i]} samples per record") for i, raw in enumerate(fields["nsamp"])]
    if min(nsamp) < 1:
        raise ValueError("every signal needs at least one sample per record")
    record_values = sum(nsamp)
    expected = header_bytes + n_records * record_values * 2
    if len(data) != expected:
        raise ValueError(f"file is {len(data)} bytes, expected {expected} for {n_records} records")

    records = np.frombuffer(data, dtype="<i2", offset=header_bytes).reshape(n_records, record_values)
    starts = np.concatenate([[0], np.cumsum(nsamp)])
    out_labels, units, fs, signals, annotations = [], [], [], [], []
    for i, label in enumerate(labels):
        block = records[:, starts[i]:starts[i + 1]]
        if label == ANNOTATIONS_LABEL:
            for r in range(n_records):
                annotations.extend(_tals(block[r].tobytes(), r, r * duration))
            continue
        dmin, dmax = (_int(fields[k][i], f"{label} {k}") for k in ("dmin", "dmax"))
        pmin, pmax = (_num(fields[k][i], f"{label} {k}") for k in ("pmin", "pmax"))
        if dmin >= dmax:
            raise ValueError(f"{label}: digital minimum {dmin} >= maximum {dmax}")
        if pmin == pmax:
            raise ValueError(f"{label}: physical minimum equals maximum ({pmin})")
        digital = block.reshape(-1).astype(np.float64)
        if digital.min() < dmin or digital.max() > dmax:
            raise ValueError(f"{label}: digital value outside [{dmin}, {dmax}]")
        out_labels.append(label)
        units.append(_ascii(fields["unit"][i], f"{label} unit"))
        fs.append(nsamp[i] / duration)
        signals.append((digital - dmin) * ((pmax - pmin) / (dmax - dmin)) + pmin)
    return Edf(tuple(out_labels), tuple(units), tuple(fs), tuple(signals), tuple(annotations))


def write_edf(signals: dict[str, np.ndarray], fs: float, annotations: list[tuple[float, float, str]] = (),
              *, record_s: float = 1.0, unit: str = "uV", digital: int = 8092, ann_bytes: int = 160) -> bytes:
    """A minimal EDF+C file (gain 1: physical == digital), for tests and synthetic fixtures.

    Each annotation (onset, duration, text) goes in the record holding its onset.
    """
    n = {len(x) for x in signals.values()}
    per_record = round(fs * record_s)
    if len(n) != 1 or n.pop() % per_record:
        raise ValueError("signals need one common length, a whole number of records")
    n_records = len(next(iter(signals.values()))) // per_record
    labels = [*signals, ANNOTATIONS_LABEL]
    ns = len(labels)
    tals: list[bytes] = [b"+%g\x14\x14\x00" % (r * record_s) for r in range(n_records)]
    for onset, dur, text in annotations:
        tals[int(onset // record_s)] += b"+%g\x15%g\x14%s\x14\x00" % (onset, dur, text.encode())
    if max(map(len, tals)) > ann_bytes:
        raise ValueError("annotations do not fit in ann_bytes per record")

    def col(values: list[str], width: int) -> bytes:
        return b"".join(v.encode("ascii").ljust(width)[:width] for v in values)

    head = b"".join([
        b"0".ljust(8), b"X X X X".ljust(80), b"Startdate X X X X".ljust(80), b"01.01.00", b"00.00.00",
        str(256 * (ns + 1)).encode().ljust(8), b"EDF+C".ljust(44), str(n_records).encode().ljust(8),
        (b"%g" % record_s).ljust(8), str(ns).encode().ljust(4),
    ])
    ordinary = len(signals)
    head += col(labels, 16) + col([""] * ns, 80) + col([unit] * ordinary + [""], 8)
    head += col([str(-digital)] * ordinary + ["-1"], 8) + col([str(digital)] * ordinary + ["1"], 8)
    head += col([str(-digital)] * ordinary + ["-32768"], 8) + col([str(digital)] * ordinary + ["32767"], 8)
    head += col([""] * ns, 80) + col([str(per_record)] * ordinary + [str(ann_bytes // 2)], 8) + col([""] * ns, 32)
    body = bytearray()
    arrays = [np.asarray(x).astype("<i2") for x in signals.values()]
    for r in range(n_records):
        for x in arrays:
            body += x[r * per_record:(r + 1) * per_record].tobytes()
        body += tals[r].ljust(ann_bytes, b"\x00")
    return head + bytes(body)
