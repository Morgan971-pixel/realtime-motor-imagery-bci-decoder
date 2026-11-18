# Public data: two positive controls

**Bottom line:** on public EEG recorded at the club headset's 8 electrode positions, the unchanged pipeline (same bands, same 0.5 to 2.5 s window, same CSP + LDA, same contiguous-block cross-validation) decodes imagined left vs right hand above chance on two independent datasets. EEGMMIDB: 56 of 106 people, against about 5 expected by luck. Cho et al. 2017: 40 of 52, against about 2.6 ([second dataset](#second-dataset-cho-et-al-2017)). The club recordings' near-chance results (1 of 5 above chance) therefore point at the recordings, not at the decoder. On both datasets the Rust and C++ cores produce bit-identical decisions and agree with Python to within 1e-13 relative on features.

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

## Second dataset: Cho et al. 2017

A second, independent check: another lab, another headset (BioSemi ActiveTwo), another sample rate (512 Hz), and about four times as many trials per person.

**The dataset.** 52 people. After a 2 s fixation cross an instruction ("left hand" or "right hand") stays on screen for 3 s while the person imagines moving that hand. 100 trials per hand (120 for people 7, 9 and 46), recorded with 64 EEG channels. Cite it as: Cho H, Ahn M, Ahn S, Kwon M, Jun SC. EEG datasets for motor imagery brain-computer interface. GigaScience 6(7):1-8, 2017 (doi 10.1093/gigascience/gix034); data doi 10.5524/100295. Licence: CC0 1.0.

**Fetch and convert.**

    cd python
    python -m mi_ref.cho2017 --out ../public-data/cho2017       # all 52 people, 10.7 GB downloaded, 4.9 GB of CSV kept

The command downloads each person's file from GigaDB, checks its SHA-256, converts it, writes `subjectNNN/session001/sNN.csv` and deletes the download (`--keep-mat` keeps it). It is resumable like the EEGMMIDB fetch. A person whose file cannot be converted gets `sNN.refused` with the reason, and their download stays in `raw/` for inspection; delete the `.refused` file to try again.

**What the conversion does.**
- *Rebuilds each session.* The files store every trial as a 7 s piece (2 s before to 5 s after the instruction), with left-hand and right-hand pieces in separate arrays. The dataset's `trial_sequence.zip` gives each trial's onset in the session. The converter puts the pieces back in that order, end to end, with timestamps from the true onsets, and numbers the trials 1 to 200 in session order. Contiguous-block cross-validation needs that order. The joins between pieces fall at least 2.5 s before the next decision window. On person 1, filtering the stitched recording or each piece on its own changed a window's log variance by at most 1e-4.
- *Reference.* BioSemi records without a reference electrode (the values carry DC offsets near 1e5 µV). The converter subtracts the average of the 64 EEG channels before keeping the 8 club positions. This was decided before any result was computed, so it is not a tuned choice.
- *Channels.* The files carry no channel names, only each person's digitised electrode positions. The 8 rows come from the standard BioSemi 64 order, and every file must agree with it: C3 and PO7 on the left, C4 and PO8 on the right, C3 in front of PO7 and C4 in front of PO8, the midline channels between C3 and C4, and Fz, Cz, Pz, Oz from front to back. A first version also required the midline within 0.15 of the centre and Cz near the top. It refused 5 people whose caps sat about 0.2 off-centre or tilted back, although their channel order was right, so only relative checks remain.
- *Codes.* One-sample stim pulse at each instruction onset, code `trial * 100 + 30 + movement`, as for EEGMMIDB. All trials are kept: the dataset's own bad-trial lists are not applied.

**Checksums.** GigaDB publishes none. `python/mi_ref/cho2017_sha256.txt` holds the SHA-256 of the 53 files as first downloaded on 2026-11-17 (sizes matched GigaDB's listing to the byte). Every later download must match them. A second download of all 52 data files the same evening matched every pinned value, and every file passed the final position check. This proves a file is the one this repository was built and evaluated on. It cannot prove that file is the one the authors uploaded.

### Results

    python -m mi_ref.evaluate --data-dir ../public-data/cho2017 --fs 512 | tee ../docs/cho2017-evaluate.txt

The default `--min-trials 30` works here (200 or 240 trials per person). The full per-person table is in [cho2017-evaluate.txt](cho2017-evaluate.txt).

| | Cho 2017 (8 club channels) | EEGMMIDB (8 club channels) |
|---|---|---|
| People | 52 | 106 |
| Trials per person | 200 (240 for three people) | 45 |
| Chance threshold (95%) | 56.5% (55.8% at 240) | 64.4% |
| Mean accuracy | 63.8% | 64.3% |
| Median accuracy | 60.8% | 64.4% |
| Above chance | 40 of 52 | 56 of 106 |
| Expected above chance by luck | about 2.6 | about 5.3 |

- **Spread.** Accuracies range from 46.0% to 94.5%; 6 people reach 80% or more.
- **Narrow passes.** People 13 and 44 sit exactly on the threshold (56.5%), so treat them as borderline.
- **The paper's flagged people.** The dataset's authors set aside people 29 and 34, because over 90% of their trials showed real hand movement in the EMG. Here they score 55.5% and 52.0%, both below chance. Without them: 40 of 50 above chance, mean 64.2%.
- **For comparison.** The authors report 38 of 52 people with "reasonably discriminative" data, using all 64 channels.

Mean accuracy is almost the same as on EEGMMIDB. The larger share above chance comes mostly from the lower threshold: 200 trials per person instead of 45.

### Real-data parity at 512 Hz

`python/tests/test_cho2017_parity.py` trains on person 1's first 150 trials, writes the last 50 to their own CSV, and decodes them with the Python reference and with each native core (`decode_file` and the streaming `Decoder`). It is the first test of the cores above 250 Hz: a 1024-sample window and offset-laden BioSemi values. Measured while this was built:
- Rust and C++ identical bit for bit on all 50 decisions;
- both within 9.1e-14 relative of Python on features and 2.1e-12 on scores.

The test skips when person 1 is not fetched; CI does not fetch it (a 203 MB download). `REQUIRE_CHO2017_DATA=1` makes missing data a failure, and `MI_CHO2017_DATA` points it at another folder.

