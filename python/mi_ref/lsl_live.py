"""Consume an LSL EEG stream (optionally with a separate cue-marker stream) and publish decoder decisions."""
import argparse
import importlib
import json
import math
import sys
import time

import numpy as np

from .cue_merge import CueMerger
from .model import Model
from .recording import EEG_CHANNELS
from .streaming import StreamingDecoder


def _pylsl():
    try:
        import pylsl
    except ImportError as exc:
        raise RuntimeError("LSL support is optional; install with .venv/bin/pip install -e 'python[live]'") from exc
    return pylsl


MARKER_LABELS = ("stim", "markers")


def _labels(info):
    labels = []
    channel = info.desc().child("channels").child("channel")
    while not channel.empty():
        labels.append((channel.child_value("label").strip(), channel.child_value("type").strip().lower()))
        channel = channel.next_sibling("channel")
    return labels


def parse_channels(spec: str):
    """The --channels option. Either 8 zero-based stream indices in model channel order
    ('0-7', '3,0,1,2,4,5,6,7') -> list of ints, or every model channel mapped to a stream label
    ('Fz=EEG 1,C3=EEG 2,...') -> {model channel: stream label}."""
    if "=" in spec:
        pairs = {}
        for part in spec.split(","):
            name, sep, label = part.partition("=")
            name, label = name.strip(), label.strip()
            if not sep or not name or not label:
                raise ValueError(f"bad channel mapping {part!r}; expected MODEL=LABEL")
            if name not in EEG_CHANNELS:
                raise ValueError(f"unknown model channel {name!r}; expected one of {', '.join(EEG_CHANNELS)}")
            if name in pairs:
                raise ValueError(f"model channel {name} is mapped twice")
            pairs[name] = label
        missing = [name for name in EEG_CHANNELS if name not in pairs]
        if missing:
            raise ValueError(f"channel mapping is missing {', '.join(missing)}")
        return pairs
    indices = []
    for part in spec.split(","):
        lo, sep, hi = part.strip().partition("-")
        if not lo.isdigit() or (sep and not hi.isdigit()) or (sep and int(hi) < int(lo)):
            raise ValueError(f"bad channel index or range {part.strip()!r}")
        indices.extend(range(int(lo), int(hi if sep else lo) + 1))
    if len(indices) != len(EEG_CHANNELS):
        raise ValueError(f"--channels needs {len(EEG_CHANNELS)} indices, one per model channel in the order "
                         f"{', '.join(EEG_CHANNELS)}; got {len(indices)}")
    if len(set(indices)) != len(indices):
        raise ValueError(f"--channels repeats an index: {indices}")
    return indices


def _mapped_layout(info, channels):
    """EEG indices picked by --channels; every other channel is ignored, except that a single
    channel labelled stim or Markers still supplies the stim code."""
    n = info.channel_count()
    names = [label for label, _ in _labels(info)]
    names += [""] * (n - len(names))
    markers = [i for i, name in enumerate(names) if name.lower() in MARKER_LABELS]
    if len(markers) > 1:
        raise ValueError(f"more than one stim/marker channel: {[names[i] for i in markers]}")
    if isinstance(channels, dict):
        eeg = []
        for name in EEG_CHANNELS:
            hits = [i for i, label in enumerate(names) if label == channels[name]]
            if len(hits) != 1:
                raise ValueError(f"channel {name}: stream label {channels[name]!r} found {len(hits)} times, expected once")
            eeg.append(hits[0])
        for i, index in enumerate(eeg):
            if index in eeg[:i]:
                first = EEG_CHANNELS[eeg.index(index)]
                raise ValueError(f"{first} and {EEG_CHANNELS[i]} both map to stream channel {index}")
    else:
        eeg = list(channels)
        if max(eeg) >= n:
            raise ValueError(f"--channels index {max(eeg)} is out of range for a {n}-channel stream")
    if markers and markers[0] in eeg:
        raise ValueError(f"channel {markers[0]} ({names[markers[0]]}) is a stim/marker channel, not EEG")
    return eeg, (markers[0] if markers else None)


def channel_layout(info, channels=None):
    if channels is not None:
        return _mapped_layout(info, channels)
    labels = _labels(info)
    if not labels:
        if info.channel_count() != 8:
            raise ValueError("LSL stream has no channel labels; expected exactly 8 EEG channels")
        labels = [(name, "eeg") for name in EEG_CHANNELS]
    by_name = {}
    for index, (label, kind) in enumerate(labels):
        if not label:
            raise ValueError(f"LSL channel {index} has no label")
        if label in by_name:
            raise ValueError(f"duplicate LSL channel label {label!r}")
        by_name[label] = (index, kind)
    missing = [label for label in EEG_CHANNELS if label not in by_name]
    unknown = [label for label, kind in labels if label not in EEG_CHANNELS and label.lower() not in MARKER_LABELS]
    markers = [label for label, _ in labels if label.lower() in MARKER_LABELS]
    if missing or unknown or len(markers) > 1:
        raise ValueError(f"invalid LSL channel layout: missing={missing}, unknown={unknown}, markers={markers}")
    eeg = [by_name[label][0] for label in EEG_CHANNELS]
    stim = by_name[markers[0]][0] if markers else None
    if any(by_name[label][1] not in ("eeg", "") for label in EEG_CHANNELS):
        raise ValueError("EEG channel labels must have type EEG")
    return eeg, stim


def make_decoder(core, model: Model, model_json: str):
    if core == "python":
        return StreamingDecoder(model)
    module_name = {"rust": "mi_core", "cpp": "mi_core_cpp"}.get(core)
    if module_name is None:
        raise ValueError(f"unknown core {core!r}; choose python, rust, or cpp")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(f"core {core!r} requires {module_name}; build its Python binding first") from exc
    return module.Decoder(model_json)


def format_decision(decision, channels=EEG_CHANNELS):
    pinned = [channels[index] for index in decision["pinned"]]
    return {
        "sample": int(decision["sample"]),
        "trial": int(decision["trial"]),
        "true_label": int(decision["true_label"]),
        "label": int(decision["label"]),
        "score": float(decision["score"]),
        "pinned": pinned,
    }


def terminal_line(decision):
    pinned = ",".join(decision["pinned"]) or "-"
    return (f"sample={decision['sample']} trial={decision['trial']} true_label={decision['true_label']} "
            f"label={decision['label']} score={decision['score']:.17g} pinned={pinned}")


def publish_decision(outlet, decision, timestamp=None, pylsl_module=None):
    payload = json.dumps(decision, separators=(",", ":"), allow_nan=False)
    if timestamp is None or not math.isfinite(timestamp) or timestamp <= 0:
        timestamp = (pylsl_module or _pylsl()).local_clock()
    outlet.push_sample([payload], timestamp=timestamp)


def _validated(sample, eeg_indices, stim_index, index):
    values = np.asarray(sample, dtype=np.float64)
    if eeg_indices is None:
        if values.shape != (8,):
            raise ValueError(f"expected 8 EEG values, got {values.shape}")
        eeg = values
    else:
        eeg = values[eeg_indices]
    stim = 0.0 if stim_index is None else float(values[stim_index])
    if eeg.shape != (8,) or not np.all(np.isfinite(eeg)) or not math.isfinite(stim):
        raise ValueError(f"invalid LSL sample at accepted index {index}")
    return eeg, stim


def _drain(cue_inlet, merger, timeout):
    """Hand every waiting cue to the merger; the first pull may wait up to timeout."""
    while True:
        chunk, stamps = cue_inlet.pull_chunk(timeout=timeout)
        if not chunk:
            return
        for sample, stamp in zip(chunk, stamps):
            merger.add_cue(sample[0], stamp)
        timeout = 0.0


def samples(inlet, *, eeg_indices=None, stim_index=None, cue_inlet=None, merger=None, timeout=0.25,
            max_samples=None, lost=(), report=None):
    """(eeg, stim, timestamp) for every accepted sample, in stream order. With a cue inlet the
    samples come out of the merger after its lag, cues are collected after every EEG chunk, and
    everything still held is released when the stream ends, is lost or reaches max_samples.
    Losing the cue stream (say the cue program finished) is reported and decoding goes on
    without new cues; losing the EEG stream ends the samples."""
    pulled = 0
    cues_alive = cue_inlet is not None

    def drain(timeout_s):
        nonlocal cues_alive
        if not cues_alive:
            return
        try:
            _drain(cue_inlet, merger, timeout_s)
        except lost as err:
            cues_alive = False
            if report is not None:
                report(f"LSL cue stream lost: {err}; decoding continues without new cues")

    try:
        while max_samples is None or pulled < max_samples:
            want = 1024 if max_samples is None else min(1024, max_samples - pulled)
            chunk, stamps = inlet.pull_chunk(timeout=timeout, max_samples=want)
            for sample, stamp in zip(chunk, stamps):
                eeg, stim = _validated(sample, eeg_indices, stim_index, pulled)
                pulled += 1
                if merger is None:
                    yield eeg, stim, stamp
                else:
                    merger.add_eeg(eeg, stamp)
            if merger is not None:
                drain(0.0)
                yield from merger.pop_ready()
    except lost as err:
        if report is not None:
            report(f"LSL EEG stream lost: {err}")
    if merger is not None:
        drain(timeout)  # markers sent together with the last samples
        yield from merger.flush()


def run(inlet, decoder, marker_outlet=None, printer=print, *, eeg_indices=None, stim_index=None,
        timeout=0.25, max_samples=None, pylsl_module=None, cue_inlet=None, cue_lag=0.5, fs=None, report=None):
    """Decode an LSL EEG stream. With cue_inlet, cue markers from that second stream become the
    stim values of the EEG samples they belong to (cue_merge.py). Returns the number of samples
    given to the decoder. Timestamps are expected on the local LSL clock (open_streams turns on
    clock synchronisation), so decision markers reuse them as they are."""
    pylsl_module = pylsl_module or _pylsl()
    report = report or (lambda message: print(message, file=sys.stderr, flush=True))
    merger = None
    if cue_inlet is not None:
        if stim_index is not None:
            raise ValueError("the EEG stream has its own stim channel; use it or a cue stream, not both")
        if fs is None:
            raise ValueError("fs is required to merge a cue stream")
        merger = CueMerger(fs, cue_lag, report)
    lost = getattr(getattr(pylsl_module, "util", None), "LostError", None)
    count = 0
    try:
        inlet.open_stream(timeout=timeout)
        if cue_inlet is not None:
            cue_inlet.open_stream(timeout=timeout)
        for eeg, stim, timestamp in samples(inlet, eeg_indices=eeg_indices, stim_index=stim_index,
                                            cue_inlet=cue_inlet, merger=merger, timeout=timeout,
                                            max_samples=max_samples, lost=(lost,) if lost else (), report=report):
            decision = decoder.push(eeg, stim)
            count += 1
            if decision is not None:
                normalized = format_decision(decision)
                printer(terminal_line(normalized))
                if marker_outlet is not None:
                    publish_decision(marker_outlet, normalized, timestamp, pylsl_module)
    finally:
        inlet.close_stream()
        if cue_inlet is not None:
            cue_inlet.close_stream()
    if merger is not None and merger.summary():
        report(merger.summary())
    return count


def _resolve(name, stream_type, timeout, pylsl):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        candidates = pylsl.resolve_streams(wait_time=min(0.25, max(0.0, deadline - time.monotonic())))
        matching = [info for info in candidates if info.name() == name and info.type() == stream_type]
        if len(matching) == 1:
            return matching[0]
        if len(matching) > 1:
            raise RuntimeError(f"multiple LSL streams match name={name!r}, type={stream_type!r}")
    found = [(info.name(), info.type(), info.channel_count()) for info in pylsl.resolve_streams(wait_time=0.1)]
    raise TimeoutError(f"no LSL stream name={name!r}, type={stream_type!r}; candidates={found}")


def open_streams(pylsl, fs, stream_name, stream_type="EEG", channels=None, cue_stream=None, timeout=10.0):
    """Resolve and open the EEG inlet and, when cue_stream is given, a Markers inlet for the cues.
    Both use LSL clock synchronisation; the EEG inlet also smooths timestamp jitter and keeps
    timestamps from going backwards, and neither reconnects silently. Checks the sample rate
    against fs and maps the channels. Returns (inlet, eeg_indices, stim_index, cue_inlet)."""
    resolved = _resolve(stream_name, stream_type, timeout, pylsl)
    inlet = pylsl.StreamInlet(resolved, recover=False,
                              processing_flags=pylsl.proc_clocksync | pylsl.proc_dejitter | pylsl.proc_monotonize)
    inlet.open_stream(timeout=timeout)
    info = inlet.info(timeout=timeout)
    if info.nominal_srate() and abs(info.nominal_srate() - fs) > 1e-6 * fs:
        raise ValueError(f"LSL sample rate {info.nominal_srate()} does not match model rate {fs}")
    eeg_indices, stim_index = channel_layout(info, channels)
    cue_inlet = None
    if cue_stream is not None:
        cue_inlet = pylsl.StreamInlet(_resolve(cue_stream, "Markers", timeout, pylsl), recover=False,
                                      processing_flags=pylsl.proc_clocksync)
        cue_inlet.open_stream(timeout=timeout)
    return inlet, eeg_indices, stim_index, cue_inlet


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--core", choices=("python", "rust", "cpp"), default="python")
    parser.add_argument("--stream-name", default="Unicorn")
    parser.add_argument("--stream-type", default="EEG")
    parser.add_argument("--channels", help="pick the 8 EEG channels: indices in model order (e.g. 0-7) "
                        "or MODEL=LABEL pairs (e.g. Fz=EEG 1,C3=EEG 2,...); other channels are ignored")
    parser.add_argument("--cue-stream", metavar="NAME",
                        help="merge cue markers from this LSL Markers stream (e.g. mi-realtime-cues from mi_ref.cues)")
    parser.add_argument("--cue-lag", type=float, default=0.5, metavar="SECONDS",
                        help="hold EEG this long so cues that arrive a little late still land (default 0.5)")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--marker-name", default="mi-realtime-decisions")
    parser.add_argument("--marker-source-id", default="mi-realtime-decisions")
    parser.add_argument("--no-marker-outlet", action="store_true")
    args = parser.parse_args(argv)
    try:
        pylsl = _pylsl()
        with open(args.model, encoding="utf-8") as file:
            model_json = file.read()
        model = Model.from_json(model_json)
        channels = parse_channels(args.channels) if args.channels else None
        inlet, eeg_indices, stim_index, cue_inlet = open_streams(
            pylsl, model.fs, args.stream_name, args.stream_type, channels, args.cue_stream, args.timeout)
        decoder = make_decoder(args.core, model, model_json)
        marker_outlet = None
        if not args.no_marker_outlet:
            marker_info = pylsl.StreamInfo(args.marker_name, "Markers", 1, 0, pylsl.cf_string, args.marker_source_id)
            marker_outlet = pylsl.StreamOutlet(marker_info)
        run(inlet, decoder, marker_outlet, eeg_indices=eeg_indices, stim_index=stim_index, pylsl_module=pylsl,
            cue_inlet=cue_inlet, cue_lag=args.cue_lag, fs=model.fs)
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
