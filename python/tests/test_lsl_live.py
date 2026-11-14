import json
import os
import threading
import uuid

import numpy as np
import pytest

from mi_ref.golden import GOLDEN_DIR
from mi_ref.lsl_live import channel_layout, make_decoder, run
from mi_ref.lsl_replay import publish, stream_info
from mi_ref.model import Model
from mi_ref.recording import Recording, load_recording

try:
    import pylsl
except ImportError:
    if os.environ.get("REQUIRE_LSL") == "1":
        raise
    pytest.skip("pylsl is not installed", allow_module_level=True)
MODEL_TEXT = (GOLDEN_DIR / "model.json").read_text()
MODEL = Model.from_json(MODEL_TEXT)


def _pull_all(inlet, expected, timeout=10):
    samples = []
    while len(samples) < expected:
        chunk, timestamps = inlet.pull_chunk(timeout=timeout, max_samples=min(1024, expected - len(samples)))
        if not chunk:
            pytest.fail(f"timed out after receiving {len(samples)} of {expected} samples")
        samples.extend(zip(chunk, timestamps))
    return samples


def test_replay_outlet_metadata_and_payload():
    recording = load_recording(GOLDEN_DIR / "stream.csv")
    source_id = f"test-{uuid.uuid4()}"
    info = stream_info("mi-realtime-test", source_id, MODEL.fs, pylsl)
    outlet = pylsl.StreamOutlet(info)
    inlet = pylsl.StreamInlet(info)
    inlet.open_stream(timeout=2)
    started = threading.Event()
    failures = []

    def send():
        try:
            started.set()
            publish(recording, MODEL, source_id=source_id, paced=False, outlet=outlet)
        except Exception as error:
            failures.append(error)

    thread = threading.Thread(target=send)
    thread.start()
    assert started.wait(2)
    received = _pull_all(inlet, len(recording.stim))
    thread.join(timeout=10)
    inlet.close_stream()
    assert not thread.is_alive()
    assert not failures
    assert info.channel_count() == 9
    assert info.nominal_srate() == pytest.approx(MODEL.fs)
    assert [sample for sample, _ in received] == [
        [*signal.tolist(), float(stim)]
        for signal, stim in zip(recording.signal, recording.stim)
    ]
    timestamps = [timestamp for _, timestamp in received]
    assert all(right > left for left, right in zip(timestamps, timestamps[1:]))


def test_replay_outlet_preserves_large_stim_codes():
    code = 2**24 + 1
    recording = Recording(np.array([0.0]), np.zeros((1, 8)), np.array([code], dtype=np.int64))
    source_id = f"large-stim-{uuid.uuid4()}"
    info = stream_info("mi-realtime-large-stim", source_id, MODEL.fs, pylsl)
    outlet = pylsl.StreamOutlet(info)
    inlet = pylsl.StreamInlet(info)
    inlet.open_stream(timeout=2)
    publish(recording, MODEL, source_id=source_id, paced=False, outlet=outlet)
    samples = _pull_all(inlet, 1)
    inlet.close_stream()
    assert samples[0][0][-1] == float(code)


@pytest.mark.parametrize("core", ["python", "rust", "cpp"])
def test_fake_lsl_stream_decodes_and_publishes_marker_decisions(core):
    recording = load_recording(GOLDEN_DIR / "stream.csv")
    source_id = f"eeg-{uuid.uuid4()}"
    marker_id = f"decisions-{uuid.uuid4()}"
    eeg_info = stream_info("mi-realtime-test", source_id, MODEL.fs, pylsl)
    eeg_outlet = pylsl.StreamOutlet(eeg_info)
    eeg_inlet = pylsl.StreamInlet(eeg_info)
    marker_info = pylsl.StreamInfo("mi-realtime-test-decisions", "Markers", 1, 0,
                                   pylsl.cf_string, marker_id)
    marker_outlet = pylsl.StreamOutlet(marker_info)
    marker_inlet = pylsl.StreamInlet(marker_info)
    eeg_inlet.open_stream(timeout=2)
    marker_inlet.open_stream(timeout=2)
    decoder = make_decoder(core, MODEL, MODEL_TEXT)
    lines = []
    publisher_errors = []

    def send():
        try:
            publish(recording, MODEL, source_id=source_id, paced=False, outlet=eeg_outlet)
        except Exception as error:
            publisher_errors.append(error)

    publisher = threading.Thread(target=send)
    publisher.start()
    expected_samples = len(recording.stim)
    channel_indices, stim_index = channel_layout(eeg_info)
    accepted = run(eeg_inlet, decoder, marker_outlet, lines.append,
                   eeg_indices=channel_indices, stim_index=stim_index,
                   timeout=1, max_samples=expected_samples, pylsl_module=pylsl)
    publisher.join(timeout=10)
    assert not publisher.is_alive()
    assert not publisher_errors
    assert accepted == expected_samples

    marker_samples = _pull_all(marker_inlet, 8)
    marker_inlet.close_stream()
    actual = [json.loads(sample[0]) for sample, _ in marker_samples]
    expected = json.loads((GOLDEN_DIR / "expected.json").read_text())["stream"]["decisions"]
    keys = ("sample", "trial", "true_label", "label", "pinned")
    assert [tuple(d[key] for key in keys) for d in actual] == [
        tuple(d[key] for key in keys) for d in expected
    ]
    assert [d["score"] for d in actual] == pytest.approx([d["score"] for d in expected], abs=1e-7)
    assert len(lines) == len(actual) == 8
    assert all("Fz" not in line and "C3" not in line for line in lines)
    assert all(set(d) == {"sample", "trial", "true_label", "label", "score", "pinned"} for d in actual)
    assert all("signal" not in sample[0] for sample, _ in marker_samples)


def test_channel_layout_reorders_eeg_and_accepts_absent_stim():
    info = pylsl.StreamInfo("reordered", "EEG", 8, MODEL.fs, pylsl.cf_double64, "reordered")
    channels = info.desc().append_child("channels")
    for name in reversed(MODEL.channels):
        channels.append_child("channel").append_child_value("label", name)
    assert channel_layout(info) == (list(reversed(range(8))), None)


def test_channel_layout_rejects_missing_duplicate_and_unknown_labels():
    for labels in ([*MODEL.channels[:-1]], [*MODEL.channels, "Fz"], [*MODEL.channels, "extra"]):
        info = pylsl.StreamInfo("bad", "EEG", len(labels), MODEL.fs, pylsl.cf_double64, str(uuid.uuid4()))
        channels = info.desc().append_child("channels")
        for name in labels:
            channels.append_child("channel").append_child_value("label", name)
        with pytest.raises(ValueError, match="(invalid LSL channel layout|duplicate LSL channel label)"):
            channel_layout(info)


def test_run_rejects_bad_sample_before_decoder_push():
    class Inlet:
        def __init__(self):
            self.closed = False

        def pull_chunk(self, timeout, max_samples):
            return [[0.0] * 7], [1.0]

        def open_stream(self, timeout):
            pass

        def close_stream(self):
            self.closed = True

    class Decoder:
        def push(self, sample, stim):
            raise AssertionError("invalid input reached decoder")

    inlet = Inlet()
    with pytest.raises(ValueError, match="expected 8 EEG values"):
        run(inlet, Decoder(), eeg_indices=None, max_samples=1, pylsl_module=pylsl)
    assert inlet.closed


def test_marker_formatter_falls_back_to_local_clock():
    from mi_ref.lsl_live import publish_decision

    class Outlet:
        sample = None

        def push_sample(self, sample, timestamp):
            self.sample = (sample, timestamp)

    outlet = Outlet()
    decision = {"sample": 4, "trial": 1, "true_label": 1, "label": 2, "score": 0.5, "pinned": []}
    publish_decision(outlet, decision, timestamp=float("nan"), pylsl_module=pylsl)
    assert json.loads(outlet.sample[0][0]) == decision
    assert outlet.sample[1] > 0
