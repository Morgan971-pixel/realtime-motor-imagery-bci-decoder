"""Publish a CSV recording as a paced LSL EEG stream."""
import argparse
import sys
import time
import uuid

import numpy as np

from .cues import marker_info
from .model import Model
from .recording import EEG_CHANNELS, load_recording


def stream_info(name: str, source_id: str, fs: float, pylsl):
    info = pylsl.StreamInfo(name, "EEG", 9, fs, pylsl.cf_double64, source_id)
    channels = info.desc().append_child("channels")
    for label in (*EEG_CHANNELS, "stim"):
        channel = channels.append_child("channel")
        channel.append_child_value("label", label)
        channel.append_child_value("type", "EEG" if label != "stim" else "Markers")
        channel.append_child_value("unit", "microvolts" if label != "stim" else "code")
    return info


AUX_LABELS = ("Accelerometer X", "Accelerometer Y", "Accelerometer Z", "Gyroscope X", "Gyroscope Y",
              "Gyroscope Z", "Battery Level", "Counter")


def split_stream_info(name: str, source_id: str, fs: float, pylsl, eeg_positions=tuple(range(8))):
    """A Unicorn-like EEG stream: 16 double64 channels and no stim. Model channel k sits at
    eeg_positions[k] with label 'EEG k+1'; the other positions carry motion, battery and counter
    labels (their values are 0, except the counter). Not the official g.tec labels."""
    if sorted(eeg_positions) != sorted(set(eeg_positions)) or len(eeg_positions) != 8 or max(eeg_positions) > 15:
        raise ValueError(f"eeg_positions must be 8 distinct indices below 16, got {eeg_positions}")
    labels = [None] * 16
    for k, position in enumerate(eeg_positions):
        labels[position] = f"EEG {k + 1}"
    aux = iter(AUX_LABELS)
    labels = [label or next(aux) for label in labels]
    info = pylsl.StreamInfo(name, "EEG", 16, fs, pylsl.cf_double64, source_id)
    channels = info.desc().append_child("channels")
    for label in labels:
        channel = channels.append_child("channel")
        channel.append_child_value("label", label)
        channel.append_child_value("type", "EEG" if label.startswith("EEG") else "AUX")
    return info


def cue_onsets(stim) -> list[tuple[int, int]]:
    """(sample, code) at the first sample of every stim pulse, as events.markers_from_stim reads them."""
    return [(i, int(code)) for i, code in enumerate(stim) if code and (i == 0 or stim[i - 1] != code)]


def publish_split(recording, model: Model, *, eeg_outlet, cue_outlet, pylsl_module, eeg_positions=tuple(range(8)),
                  paced=True, cues_first=False, start=None, sleep=time.sleep, clock=time.perf_counter):
    """Publish the EEG as a Unicorn-like stream and its stim pulses as separate cue markers.
    Sample i is stamped start + i / fs and its cue (if any) gets the same timestamp. Paced, each
    cue is pushed right after its sample, like a live cue program; cues_first pushes every cue
    before any EEG (for unpaced tests, so no cue can arrive after its sample was decoded)."""
    if recording.signal.shape[1] != len(EEG_CHANNELS):
        raise ValueError(f"expected {len(EEG_CHANNELS)} EEG channels")
    start = pylsl_module.local_clock() if start is None else start
    cues = dict(cue_onsets(recording.stim))
    if cues_first:
        for index, code in sorted(cues.items()):
            cue_outlet.push_sample([code], timestamp=start + index / model.fs)
    positions = list(eeg_positions)
    began = clock()
    row = np.zeros(16)
    counter = positions and [i for i in range(16) if i not in positions][-1]
    for index, signal in enumerate(recording.signal):
        if paced:
            delay = began + index / model.fs - clock()
            if delay > 0:
                sleep(delay)
        row[:] = 0.0
        row[positions] = signal
        row[counter] = index
        eeg_outlet.push_sample(row.tolist(), timestamp=start + index / model.fs)
        if not cues_first and index in cues:
            cue_outlet.push_sample([cues[index]], timestamp=start + index / model.fs)
    return len(recording.stim)


def publish(recording, model: Model, *, name="mi-realtime-golden", source_id=None,
            paced=True, pylsl_module=None, sleep=time.sleep, clock=time.perf_counter,
            on_published=None, outlet=None):
    if pylsl_module is None:
        try:
            import pylsl as pylsl_module
        except ImportError as exc:
            raise RuntimeError("LSL support is optional; install with .venv/bin/pip install -e 'python[live]'") from exc
    if recording.signal.shape[1] != len(EEG_CHANNELS):
        raise ValueError(f"expected {len(EEG_CHANNELS)} EEG channels")
    if not np.all(np.isfinite(recording.signal)):
        raise ValueError("recording contains non-finite EEG")
    source_id = source_id or f"mi-realtime-{uuid.uuid4()}"
    owns_outlet = outlet is None
    if owns_outlet:
        outlet = pylsl_module.StreamOutlet(stream_info(name, source_id, model.fs, pylsl_module))
    started = clock()
    try:
        for index, (signal, stim) in enumerate(zip(recording.signal, recording.stim)):
            if paced:
                due = started + index / model.fs
                delay = due - clock()
                if delay > 0:
                    sleep(delay)
            sample = [*np.asarray(signal, dtype=np.float64).tolist(), float(stim)]
            outlet.push_sample(sample, timestamp=pylsl_module.local_clock())
            if on_published is not None:
                on_published(index)
    finally:
        if owns_outlet:
            del outlet
    return len(recording.stim)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("csv")
    parser.add_argument("--name", default="mi-realtime-golden")
    parser.add_argument("--source-id")
    parser.add_argument("--unpaced", action="store_true")
    parser.add_argument("--split", metavar="CUE_STREAM",
                        help="publish a Unicorn-like EEG stream (16 channels, EEG at 0-7, no stim) and the stim "
                             "codes as markers on a separate stream CUE_STREAM, like the headset plus mi_ref.cues")
    args = parser.parse_args(argv)
    try:
        with open(args.model, encoding="utf-8") as file:
            model = Model.from_json(file.read())
        recording = load_recording(args.csv)
        if args.split:
            import pylsl
            source_id = args.source_id or f"mi-realtime-{uuid.uuid4()}"
            count = publish_split(recording, model, pylsl_module=pylsl, paced=not args.unpaced,
                                  eeg_outlet=pylsl.StreamOutlet(split_stream_info(args.name, source_id, model.fs, pylsl)),
                                  cue_outlet=pylsl.StreamOutlet(marker_info(args.split, f"{source_id}-cues", pylsl)))
        else:
            count = publish(recording, model, name=args.name, source_id=args.source_id, paced=not args.unpaced)
    except (OSError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"published {count} samples at {model.fs:g} Hz", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
