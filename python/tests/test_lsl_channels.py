import os
import uuid

import pytest

from mi_ref import lsl_live
from mi_ref.golden import GOLDEN_DIR
from mi_ref.lsl_live import channel_layout, parse_channels
from mi_ref.recording import EEG_CHANNELS

try:
    import pylsl
except ImportError:
    if os.environ.get("REQUIRE_LSL") == "1":
        raise
    pytest.skip("pylsl is not installed", allow_module_level=True)

# a Unicorn-like layout: 8 EEG channels, then motion, battery and counter (labels not official)
UNICORN = [f"EEG {i}" for i in range(1, 9)] + [
    "Accelerometer X", "Accelerometer Y", "Accelerometer Z",
    "Gyroscope X", "Gyroscope Y", "Gyroscope Z", "Battery Level", "Counter"]


def _info(labels=None, n=None):
    n = len(labels) if labels is not None else n
    info = pylsl.StreamInfo("test", "EEG", n, 250, pylsl.cf_double64, str(uuid.uuid4()))
    if labels is not None:
        channels = info.desc().append_child("channels")
        for label in labels:
            channels.append_child("channel").append_child_value("label", label)
    return info


def test_parse_channels_indices_and_ranges():
    assert parse_channels("0-7") == list(range(8))
    assert parse_channels("3,0-2, 4,5,6,7") == [3, 0, 1, 2, 4, 5, 6, 7]


def test_parse_channels_labels():
    spec = ",".join(f"{name}=EEG {i}" for i, name in enumerate(EEG_CHANNELS, 1))
    assert parse_channels(spec) == {name: f"EEG {i}" for i, name in enumerate(EEG_CHANNELS, 1)}


@pytest.mark.parametrize("spec,message", [
    ("0-6", "needs 8 indices"), ("0-8", "needs 8 indices"), ("0-6,6", "repeats an index"),
    ("7-0", "bad channel index or range"), ("a,1", "bad channel index or range"), ("-1,0-6", "bad channel"),
    ("Fz=EEG 1", "missing C3"), ("Fz=EEG 1,Fz=EEG 2", "mapped twice"), ("Xx=EEG 1", "unknown model channel"),
    ("Fz=", "expected MODEL=LABEL"),
])
def test_parse_channels_refuses(spec, message):
    with pytest.raises(ValueError, match=message):
        parse_channels(spec)


def test_indices_pick_eeg_from_a_unicorn_like_stream_and_ignore_the_rest():
    assert channel_layout(_info(UNICORN), parse_channels("0-7")) == (list(range(8)), None)
    with pytest.raises(ValueError, match="invalid LSL channel layout"):
        channel_layout(_info(UNICORN))  # without --channels the 6a rules still refuse it


def test_labels_pick_eeg_in_model_order_from_shuffled_positions():
    labels = UNICORN[8:12] + UNICORN[:8][::-1] + UNICORN[12:]
    mapping = parse_channels(",".join(f"{name}=EEG {i}" for i, name in enumerate(EEG_CHANNELS, 1)))
    eeg, stim = channel_layout(_info(labels), mapping)
    assert [labels[i] for i in eeg] == [f"EEG {i}" for i in range(1, 9)]
    assert stim is None


def test_indices_work_on_an_unlabelled_stream_and_a_stim_channel_is_kept():
    assert channel_layout(_info(n=17), parse_channels("0-7")) == (list(range(8)), None)
    assert channel_layout(_info(UNICORN + ["stim"]), parse_channels("0-7")) == (list(range(8)), 16)


@pytest.mark.parametrize("labels,spec,message", [
    (UNICORN, "9-16", "index 16 is out of range for a 16-channel stream"),
    (UNICORN[:15] + ["stim"], "8-15", "channel 15 \\(stim\\) is a stim/marker channel"),
    (UNICORN + ["stim", "Markers"], "0-7", "more than one stim/marker channel"),
    (UNICORN, "Fz=EEG 9," + ",".join(f"{n}=EEG {i}" for i, n in enumerate(EEG_CHANNELS[1:], 2)),
     "channel Fz: stream label 'EEG 9' found 0 times"),
    (UNICORN + ["EEG 1"], ",".join(f"{n}=EEG {i}" for i, n in enumerate(EEG_CHANNELS, 1)),
     "stream label 'EEG 1' found 2 times"),
])
def test_mapped_layout_refuses(labels, spec, message):
    with pytest.raises(ValueError, match=message):
        channel_layout(_info(labels), parse_channels(spec))


def test_main_reports_a_bad_channels_option_before_looking_for_streams(capsys):
    assert lsl_live.main([str(GOLDEN_DIR / "model.json"), "--channels", "0-6", "--timeout", "0"]) == 1
    assert "needs 8 indices" in capsys.readouterr().err


def test_two_model_channels_on_one_stream_channel_are_refused():
    spec = "Fz=EEG 1,C3=EEG 1," + ",".join(f"{n}=EEG {i}" for i, n in enumerate(EEG_CHANNELS[2:], 3))
    with pytest.raises(ValueError, match="Fz and C3 both map to stream channel 0"):
        channel_layout(_info(UNICORN), parse_channels(spec))
