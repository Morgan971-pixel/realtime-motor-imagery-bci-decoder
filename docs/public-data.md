# Public data: EEGMMIDB as a positive control

**Bottom line:** on public EEG recorded at the club headset's 8 electrode positions, the unchanged pipeline (same bands, same 0.5 to 2.5 s window, same CSP + LDA, same contiguous-block cross-validation) decodes imagined left vs right fist above chance for 56 of 106 people, against about 5 expected by luck. The club recordings' near-chance results (1 of 5 above chance) therefore point at the recordings, not at the decoder. On the same public data, the Rust and C++ cores produce bit-identical decisions and agree with Python to about 1e-15.

## The dataset

The PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB, version 1.0.0): 109 people, 64 channels at 160 Hz, recorded with BCI2000. In runs 4, 8 and 12 the person imagines opening and closing the left fist (annotation `T1`) or the right fist (`T2`) while a target is on screen for about 4 s; `T0` is rest. That gives 45 imagery trials per person (15 per run).

Cite it as: Schalk G, McFarland DJ, Hinterberger T, Birbaumer N, Wolpaw JR. BCI2000: a general-purpose brain-computer interface (BCI) system. IEEE Trans Biomed Eng 51(6):1034-1043, 2004; and Goldberger AL et al. PhysioBank, PhysioToolkit, and PhysioNet. Circulation 101(23):e215-e220, 2000. Licence: Open Data Commons Attribution v1.0 (https://physionet.org/content/eegmmidb/1.0.0/).

The data is public, but it is still never committed: `public-data/` is in `.gitignore` and the pre-commit hook. Results computed from it may be committed.

## Fetch and convert

    cd python
    python -m mi_ref.eegmmidb --out ../public-data/eegmmidb            # all 109 people, runs 4, 8, 12

For each run the command downloads the EDF file, checks it against the SHA-256 pinned in `python/mi_ref/eegmmidb_sha256.txt` (copied from PhysioNet's `SHA256SUMS.txt`; a mismatch stops the fetch), reads it with the strict reader in `mi_ref/edf.py`, writes a club-layout CSV, and deletes the EDF (`--keep-edf` keeps it). The 318 CSVs take about 570 MB; the download is about 800 MB in total, but each EDF is deleted once converted. Rerunning skips finished runs, so an interrupted fetch just continues. A refused run is skipped too: delete its `.refused` file to retry it (for example after changing the converter). `--subjects 1,5,7-9` fetches a subset.

Conversion keeps `Fz, C3, Cz, C4, Pz, PO7, Oz, PO8` in the club's column order, in microvolts (the files store whole microvolts). Timestamps are sample index / 160. Each `T1`/`T2` onset becomes a one-sample stim pulse with the club code `trial * 100 + 30 + movement` (phase 3 = imagery, movement 1 = left, 2 = right, trial counted from 1 within the run), so `mi_ref.evaluate`, `train` and every core read the files unchanged.

**Refused runs.** People 88, 92 and 100 were recorded at 128 Hz; the converter refuses those 9 runs (`channel Fz: sample rate 128 Hz, expected 160`) and writes the reason to `SNNNRrr.refused` instead of resampling. That leaves 106 people. Person 104's run 8 is shorter than the others (13 cues instead of 15) and is used as is.

**Reader check.** While this was built, `mi_ref.edf.read_edf` was compared with MNE 1.13.2's EDF reader on 15 files (people 1, 42, 88, 104, 109): same labels, rates and annotations, and samples within 1.2e-13 µV.

## Results

    python -m mi_ref.evaluate --data-dir ../public-data/eegmmidb --fs 160 --min-trials 10 | tee ../docs/eegmmidb-evaluate.txt

`--fs 160` sets the sample rate (the default is the club's 250 Hz). `--min-trials 10` is needed because one EEGMMIDB run holds 15 imagery trials and the default minimum is the club's 30 per recording. The full per-person table is in [eegmmidb-evaluate.txt](eegmmidb-evaluate.txt).

| | EEGMMIDB (8 club channels) | Club recordings |
|---|---|---|
| People | 106 | 5 |
| Trials per person | 45 (43 for person 104) | 32 or 64 |
| Chance threshold (95%) | 64.4% | 62.5% (64 trials), 68.8% (32) |
| Mean accuracy | 64.3% | 49.7% |
| Median accuracy | 64.4% | 53.1% |
| Above chance | 56 of 106 | 1 of 5 |
| Expected above chance by luck | about 5.3 | about 0.25 |

Accuracies range from 37.8% to 95.6%; 17 people reach 80% or more. The chance threshold treats trials as independent, as everywhere in this repo, so treat narrow passes with suspicion; 56 against an expected 5.3 is not a narrow result.

This is a positive control, not a benchmark. Published EEGMMIDB numbers usually use all 64 channels and tuned windows; this run deliberately keeps every club setting so that the only thing that changes is the data.

## Real-data parity test

`python/tests/test_public_parity.py` trains on person 1's runs 4 and 8, writes the model, and decodes run 12 with the Python reference (batch and streaming) and with each native core (`decode_file` and the streaming `Decoder`). All must make the same 15 decisions with features within 1e-9 relative, as for the goldens. It is the first test of the cores on real EEG and the first at a rate other than 250 Hz. Measured while this was built: Rust and C++ identical bit for bit, both within 1.1e-15 relative of Python on features and 7.1e-14 on scores.

The test skips when the data is missing; `REQUIRE_PUBLIC_DATA=1` (set in CI) makes that a failure. `MI_PUBLIC_DATA` points it at another folder. CI fetches person 1 (about 8 MB) with `--keep-edf` and caches the EDF files, keyed on the pinned checksums.
