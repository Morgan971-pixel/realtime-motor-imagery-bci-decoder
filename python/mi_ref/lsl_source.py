"""Samples from a live LSL EEG stream, with cues from a separate marker stream merged in.

The live decoder (lsl_live.run) and the recorder (lsl_record.record) read EEG the same way;
LiveSource is that way, so both get the same checks, cue merging and stream-loss handling.
"""
import math
import sys

import numpy as np

from .cue_merge import CueMerger


def _pylsl():
    try:
        import pylsl
    except ImportError as exc:
        raise RuntimeError("LSL support is optional; install with .venv/bin/pip install -e 'python[live]'") from exc
    return pylsl


def _to_stderr(message):
    print(message, file=sys.stderr, flush=True)


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


class LiveSource:
    """(eeg, stim, timestamp) for every accepted EEG sample, in stream order.

    Use it as a context manager: entering opens the inlets, leaving closes them and, after a
    clean run, reports what the cue merger dropped. Iterating pulls samples until max_samples,
    the end of the EEG stream, or its loss (reported, not raised).

    The stim value comes from the EEG stream's own stim channel (stim_index), from a cue
    inlet (merged by cue_merge.CueMerger, which holds samples for cue_lag seconds so that a
    slightly late cue still lands), or is 0. Losing the cue stream (say the cue program
    finished) is reported and the EEG goes on without new cues. A malformed sample raises
    ValueError before anyone sees it.
    """

    def __init__(self, inlet, *, eeg_indices=None, stim_index=None, cue_inlet=None, cue_lag=0.5, fs=None,
                 timeout=0.25, max_samples=None, pylsl_module=None, report=None):
        if cue_inlet is not None:
            if stim_index is not None:
                raise ValueError("the EEG stream has its own stim channel; use it or a cue stream, not both")
            if fs is None:
                raise ValueError("fs is required to merge a cue stream")
        self.pylsl = pylsl_module or _pylsl()
        self.report = report or _to_stderr
        self.merger = CueMerger(fs, cue_lag, self.report) if cue_inlet is not None else None
        lost = getattr(getattr(self.pylsl, "util", None), "LostError", None)
        self._lost = (lost,) if lost else ()
        self._inlet, self._cue_inlet = inlet, cue_inlet
        self._eeg_indices, self._stim_index = eeg_indices, stim_index
        self._timeout, self._max_samples = timeout, max_samples
        self._cues_alive = cue_inlet is not None

    def __enter__(self):
        try:
            self._inlet.open_stream(timeout=self._timeout)
            if self._cue_inlet is not None:
                self._cue_inlet.open_stream(timeout=self._timeout)
        except BaseException:
            self._close()
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        self._close()
        if exc_type is None and self.merger is not None and self.merger.summary():
            self.report(self.merger.summary())
        return False

    def _close(self):
        self._inlet.close_stream()
        if self._cue_inlet is not None:
            self._cue_inlet.close_stream()

    def _drain_cues(self, timeout):
        if not self._cues_alive:
            return
        try:
            _drain(self._cue_inlet, self.merger, timeout)
        except self._lost as err:
            self._cues_alive = False
            self.report(f"LSL cue stream lost: {err}; decoding continues without new cues")

    def __iter__(self):
        merger, pulled = self.merger, 0
        try:
            while self._max_samples is None or pulled < self._max_samples:
                want = 1024 if self._max_samples is None else min(1024, self._max_samples - pulled)
                chunk, stamps = self._inlet.pull_chunk(timeout=self._timeout, max_samples=want)
                for sample, stamp in zip(chunk, stamps):
                    eeg, stim = _validated(sample, self._eeg_indices, self._stim_index, pulled)
                    pulled += 1
                    if merger is None:
                        yield eeg, stim, stamp
                    else:
                        merger.add_eeg(eeg, stamp)
                if merger is not None:
                    self._drain_cues(0.0)
                    yield from merger.pop_ready()
        except self._lost as err:
            self.report(f"LSL EEG stream lost: {err}")
        if merger is not None:
            self._drain_cues(self._timeout)  # markers sent together with the last samples
            yield from merger.flush()

    def release_held(self):
        """After an interrupted iteration (Ctrl+C): the samples the cue merger still holds, with
        every cue that has already arrived merged in."""
        if self.merger is None:
            return []
        self._drain_cues(0.0)
        return list(self.merger.flush())
