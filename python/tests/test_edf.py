import numpy as np
import pytest

from mi_ref.edf import Annotation, read_edf, write_edf

FIELD_WIDTHS = {"label": 16, "transducer": 80, "unit": 8, "pmin": 8, "pmax": 8,
                "dmin": 8, "dmax": 8, "prefilter": 80, "nsamp": 8, "reserved": 32}


def _sample_file(**kw) -> bytes:
    t = np.arange(320)
    signals = {"C3..": (t % 50) - 25, "Cz..": np.zeros(320, dtype=int), "Po7.": -(t % 7)}
    return write_edf(signals, 160.0, [(0.0, 4.2, "T0"), (0.5, 4.1, "T1"), (1.25, 4.1, "T2")], **kw)


def _patch(data: bytes, offset: int, width: int, text: str) -> bytes:
    return data[:offset] + text.encode().ljust(width) + data[offset + width:]


def _patch_signal(data: bytes, field: str, signal: int, text: str) -> bytes:
    ns = int(data[252:256])
    offset = 256
    for name, width in FIELD_WIDTHS.items():
        if name == field:
            return _patch(data, offset + signal * width, width, text)
        offset += ns * width
    raise KeyError(field)


def _ann_offset(data: bytes, record: int) -> int:
    """Byte offset of a record's annotation bytes in a _sample_file (3 signals, 160 samples each)."""
    return 256 * 5 + record * (3 * 160 * 2 + 160) + 3 * 160 * 2


def test_round_trip_keeps_labels_values_rate_and_annotations():
    e = read_edf(_sample_file())
    assert e.labels == ("C3..", "Cz..", "Po7.")
    assert e.units == ("uV", "uV", "uV")
    assert e.fs == (160.0, 160.0, 160.0)
    t = np.arange(320)
    np.testing.assert_array_equal(e.signals[0], (t % 50) - 25)
    np.testing.assert_array_equal(e.signals[2], -(t % 7))
    assert e.signals[0].dtype == np.float64
    assert e.annotations == (Annotation(0.0, 4.2, "T0"), Annotation(0.5, 4.1, "T1"), Annotation(1.25, 4.1, "T2"))


def test_physical_scaling_uses_both_ranges():
    data = _sample_file()
    for field, value in (("pmin", "-1"), ("pmax", "3"), ("dmin", "-2"), ("dmax", "2")):
        data = _patch_signal(data, field, 1, value)  # Cz: digital 0 -> (0 + 2) * 4 / 4 - 1 = 1
    np.testing.assert_array_equal(read_edf(data).signals[1], np.ones(320))


def test_annotation_without_duration_and_with_two_texts():
    data = _sample_file()
    tal = b"+0\x14\x14\x00+0.75\x14A\x14B\x14\x00"
    off = _ann_offset(data, 0)
    data = data[:off] + tal.ljust(160, b"\x00") + data[off + 160:]
    assert read_edf(data).annotations[:2] == (Annotation(0.75, None, "A"), Annotation(0.75, None, "B"))


@pytest.mark.parametrize("offset,width,text,message", [
    (0, 8, "1", "version"),
    (184, 8, "1280x", "header size must be an integer"),
    (184, 8, "1024", "header size 1024"),
    (192, 44, "EDF+D", "discontinuous"),
    (236, 8, "-1", "record count"),
    (244, 8, "0", "record duration"),
    (244, 8, "1e0", "record duration must be a decimal number"),
    (252, 4, "0", "signal count"),
])
def test_refuses_bad_fixed_header(offset, width, text, message):
    with pytest.raises(ValueError, match=message):
        read_edf(_patch(_sample_file(), offset, width, text))


@pytest.mark.parametrize("field,signal,text,message", [
    ("label", 1, "C3..", "share a label"),
    ("nsamp", 0, "0", "at least one sample"),
    ("dmin", 0, "8092", "digital minimum"),
    ("pmax", 0, "-8092", "physical minimum equals maximum"),
    ("dmax", 0, "20", "digital value outside"),
    ("pmin", 2, "low", "must be a decimal number"),
])
def test_refuses_bad_signal_header(field, signal, text, message):
    with pytest.raises(ValueError, match=message):
        read_edf(_patch_signal(_sample_file(), field, signal, text))


def test_refuses_short_truncated_and_padded_files():
    data = _sample_file()
    with pytest.raises(ValueError, match="shorter than the 256-byte"):
        read_edf(data[:100])
    with pytest.raises(ValueError, match="expected"):
        read_edf(data[:-1])
    with pytest.raises(ValueError, match="expected"):
        read_edf(data + b"\x00\x00")


def test_refuses_non_ascii_header_text():
    with pytest.raises(ValueError, match="not ASCII"):
        read_edf(_patch_signal(_sample_file(), "unit", 0, "µV"))


@pytest.mark.parametrize("tal,message", [
    (b"+0\x14\x14\x00+0.5\x14T1\x00", "malformed annotation"),
    (b"+0\x14\x14\x00+0.5\x14\xff\x14\x00", "not UTF-8"),
    (b"+0.5\x14T1\x14\x00", "not a time-keeping entry"),
    (b"+3\x14\x14\x00", "expected 0"),
    (b"", "no time-keeping"),
    (b"+0\x14\x14\x00\x00\x00x", "nonzero bytes"),
])
def test_refuses_bad_annotations(tal, message):
    data = _sample_file()
    off = _ann_offset(data, 0)
    with pytest.raises(ValueError, match=message):
        read_edf(data[:off] + tal.ljust(160, b"\x00") + data[off + 160:])


def test_second_record_start_time_is_checked():
    data = _sample_file()
    off = _ann_offset(data, 1)
    with pytest.raises(ValueError, match="record 1: starts at 0.0 s, expected 1.0 s"):
        read_edf(data[:off] + b"+0\x14\x14\x00".ljust(160, b"\x00") + data[off + 160:])


def test_writer_refuses_partial_records_and_overfull_annotations():
    with pytest.raises(ValueError, match="whole number of records"):
        write_edf({"C3": np.zeros(100)}, 160.0)
    with pytest.raises(ValueError, match="do not fit"):
        write_edf({"C3": np.zeros(160)}, 160.0, [(0.1, 1.0, "x" * 200)])
