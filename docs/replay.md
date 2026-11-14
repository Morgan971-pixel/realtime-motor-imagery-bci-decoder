# Replay and benchmarks

Each implementation has a replay command that plays a recording through its decoder as if a headset were streaming it, and times every decision. The three commands share one interface so a single script can benchmark them side by side.

| Implementation | Command |
|---|---|
| Python reference | `python -m mi_ref.replay` |
| Rust | `rust/mi_core/target/release/mi-replay` |
| C++ | `cpp/build/release/mi_replay` |

## Command line

    <command> MODEL CSV [--chunk N] [--repeat N] [--paced]

- `--chunk N` (default 25): samples handed to the decoder together. A headset delivers small blocks, not single samples.
- `--repeat N` (default 1): play the recording N times back to back, through one decoder. The stream simply continues, so an imagery onset too close to the end of one pass completes its window in the next.
- `--paced`: deliver chunks at the model's sampling rate instead of as fast as possible.
- `-h`, `--help`: print the usage line (Python prints its full help) on standard output and exit 0.

The grammar is the same in all three commands. A count `N` is ASCII decimal digits only, from 1 to 2^64 - 1 (no sign, no spaces, no underscores). An option's value comes either as the next argument or after `=` (`--chunk 7` or `--chunk=7`). Options must be spelled out in full (`--rep` is unknown). Any other argument starting with `-` is an unknown option, and there must be exactly two positional arguments.

Exit status:

- 0 on success (or after `-h`).
- 2 with a line starting `usage:` on standard error for bad arguments. Rust and C++ also exit 2 when the recording's length times `--repeat` does not fit in 64 bits; Python has no such limit.
- 1 for a model or recording the core rejects (unreadable, malformed, invalid values, no samples), with standard error starting `PATH: ` where `PATH` is the file at fault.

## The loop

    start = now
    for each chunk (the recording played --repeat times):
        paced only: sleep until start + (samples up to the end of this chunk) / fs,
                    rounded down to whole ns
        handover = now
        push each sample of the chunk; for each decision: latency = now - handover

Fast mode keeps the decoder busy. Paced mode lets it idle between chunks, as a live decoder does, so its caches go cold and the CPU may clock down; that is what paced latency adds. Latency starts at the handover, when the loop is awake, not at the due time: on macOS (Apple M4) a sleeping thread woke 0.7 to 4 ms late in our tests (the kernel batches timer wake-ups to save power), while a decoder woken by real headset data does not pay that. The recording is read and the model loaded before `start`, so neither counts.

A decision's latency covers everything since its chunk's handover, including the samples (and any earlier decisions) ahead of it in the same chunk. With `--chunk 1` that is the decision sample alone; with a chunk as long as the recording it would grow with the sample index, so keep chunks small.

## Output

One line of JSON on standard output:

| Key | Meaning |
|---|---|
| `impl` | `python`, `rust` or `cpp` |
| `chunk`, `paced`, `repeat` | the options used |
| `samples` | samples pushed (recording length × repeat) |
| `elapsed_ns` | wall time of the whole loop |
| `decisions` | `[sample, label]` per decision, in order; label 0 is a decision the model's pinned-channel check refused (`docs/model-format.md`, step 7) |
| `latencies_ns` | one integer per decision |

## The benchmark script

`python -m mi_ref.bench` refuses to run a release binary older than any of its source files (rebuild it as the README says); the replay command tests refuse the same way. It runs every command on `golden/stream.csv` in two modes: **fast** (1-sample chunks, 20 passes) and **paced** (25-sample chunks at 250 Hz, one pass, about 92 seconds per implementation). Each command must echo back the options it was given and make at least one decision, and all of them must agree on the decisions and the sample count, or the script refuses to report. It checks the README's markers before the first run. Then it prints a Markdown table:

- p50, p99, max latency: nearest-rank percentiles (p99 is the smallest latency with at least 99% of decisions at or below it). Below 100 decisions p99 would only repeat the maximum, so it reads n/a; the paced run makes 8.
- Samples/s: `samples / elapsed` in fast mode. The loop reads the clock once per sample, so this is a little below the decoder's own rate (the micro-benchmarks measure that).
- The caption above the table describes only the modes that ran, gives the model's `fs`, and counts how many paced decisions fall on the last sample of their chunk. In `golden/stream.csv` all 8 do, so paced latency there includes pushing the whole 25-sample chunk.
- Peak RSS: the replay process's maximum resident memory, from `os.wait4`. For Python this includes the interpreter and NumPy. On Linux the kernel can also count the memory the process had just before it started the command (a copy of the benchmark script's own), so Peak RSS there may read higher than on macOS, which reports the command alone.

`--readme ../README.md` also rewrites the table between the `<!-- bench:start -->` and `<!-- bench:end -->` lines. Timings depend on the machine and on what else it is doing; the script records which machine produced them.

The micro-benchmarks (`cargo bench` in `rust/mi_core`, `build/release/mi_core_bench` in `cpp`) time one sample through `push` and a whole golden stream through `decode`, with each tool's own statistics.
