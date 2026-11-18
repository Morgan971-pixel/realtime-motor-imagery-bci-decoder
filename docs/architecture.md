# Architecture and module map

Drawn from the code's imports. Arrows point from a module to the module it uses.

## 1. Overall architecture

Four parts, all meeting at one written contract:
- **Data in:** the club headset live, recordings, and two public datasets.
- **Training and evaluation:** turns recordings into a `model.json`.
- **Three decoders:** each makes the same decisions from that model.
- **Checks:** prove the three decoders agree.

```mermaid
flowchart LR
    subgraph IN["Data in"]
        HS["Unicorn headset<br/>(LSL stream)"]
        CUE["cue program<br/>mi_ref.cues"]
        REC["recorder<br/>lsl_record"]
        CSV[("club-layout CSV<br/>timestamps, 8 channels, stim")]
        PUB1["EEGMMIDB converter<br/>eegmmidb"]
        PUB2["Cho 2017 converter<br/>cho2017"]
    end

    subgraph TRAIN["Train and evaluate (Python)"]
        EVAL["evaluate<br/>block cross-validation vs chance"]
        TR["train"]
        MODEL[("model.json")]
    end

    CONTRACT{{"docs/model-format.md<br/>the written contract<br/>+ golden/ fixtures"}}

    subgraph CORES["Decoders: push(sample, stim) -> decision or nothing"]
        PY["Python reference<br/>StreamingDecoder"]
        RS["Rust core<br/>mi_core"]
        CPP["C++ core<br/>mi_core_cpp"]
    end

    subgraph LIVE["Live path"]
        SRC["LiveSource<br/>reads EEG, merges cues"]
        RUN["lsl_live.run"]
    end

    OUT["terminal line<br/>+ LSL decision markers"]

    HS --> SRC
    CUE -- "cue markers" --> SRC
    HS --> REC
    CUE --> REC
    REC --> CSV
    PUB1 --> CSV
    PUB2 --> CSV
    CSV --> EVAL
    CSV --> TR --> MODEL
    MODEL --> PY & RS & CPP
    CONTRACT -. "specifies" .-> PY & RS & CPP
    SRC --> RUN
    RUN -- "one sample at a time" --> PY & RS & CPP
    PY & RS & CPP --> OUT
```

## 2. The live path, step by step

```mermaid
flowchart LR
    H["headset EEG stream<br/>(16 channels, no cues)"] --> P["pick the 8 channels<br/>--channels"]
    C["cue program<br/>LEFT / RIGHT markers"] --> M["CueMerger<br/>cue to the nearest sample,<br/>held cue_lag seconds"]
    P --> M
    M --> L["LiveSource<br/>checks, stream loss,<br/>open and close"]
    L --> D["decoder.push(sample, stim)"]
    D --> F["filters: 4 bands, causal"] --> W["ring buffer:<br/>last 2 s per band"]
    W -- "0.5 s after a cue,<br/>window full" --> X["CSP + log variance<br/>16 features"] --> Y["LDA score<br/>> 0 means RIGHT"]
    Y --> T["terminal + LSL marker"]
```

## 3. Python module map (`python/mi_ref`)

`events` (stim-code rules) and `recording` (the CSV format) are the foundation; almost every module uses them, so those arrows are left out.

```mermaid
flowchart TB
    subgraph FOUND["Foundation (used almost everywhere)"]
        events["events<br/>stim codes"]
        recording["recording<br/>CSV format"]
    end

    subgraph DATA["Data sources"]
        edf["edf<br/>strict EDF reader"]
        eegmmidb["eegmmidb"]
        cho2017["cho2017"]
        synth["synth<br/>fake EEG"]
        golden["golden<br/>fixtures"]
        invalid["invalid<br/>bad-input fixtures"]
    end

    subgraph DSP["Signal processing"]
        filters["filters"]
        epochs["epochs<br/>window maths"]
        csp["csp"]
        lda["lda"]
        model["model<br/>train, decode, model.json"]
    end

    subgraph DEC["Decoding"]
        streaming["streaming<br/>Python reference decoder"]
        cores["cores<br/>loads Python / Rust / C++"]
        replay["replay"]
        bench["bench"]
    end

    subgraph EV["Evaluate and train"]
        evaluate["evaluate"]
        train["train"]
        paths["paths<br/>outside-repo check"]
    end

    subgraph LV["Live"]
        cue_merge["cue_merge"]
        lsl_source["lsl_source<br/>LiveSource"]
        lsl_live["lsl_live"]
        lsl_record["lsl_record"]
        lsl_replay["lsl_replay"]
        cues["cues<br/>cue program"]
    end

    eegmmidb --> edf & epochs
    cho2017 --> eegmmidb
    synth --> filters
    golden --> epochs & filters & model & synth
    invalid --> golden & model & streaming
    lda --> csp
    model --> csp & epochs & filters & lda
    streaming --> csp & epochs & model
    cores --> model & streaming
    replay --> model & streaming
    bench --> golden & model
    evaluate --> filters & lda & model
    train --> evaluate & model & paths
    lsl_source --> cue_merge
    lsl_live --> cores & lsl_source & model
    lsl_record --> lsl_live & lsl_source & paths
    lsl_replay --> cues & model
    cues --> epochs

```

`cho2017` uses `eegmmidb` only for its download, checksum-file and subject-range helpers.

## 4. Native cores (Rust and C++)

The two cores split the work the same way, file for file, because both follow the same contract. Each has a small public interface:
- load and check a model;
- `push` one sample;
- `decode_file`.

Each also has Python bindings and a `replay` command.

```mermaid
flowchart TB
    subgraph RUST["Rust: rust/mi_core/src"]
        r_py["python.rs<br/>PyO3 bindings"]
        r_bin["bin/mi-replay.rs"]
        r_replay["replay.rs"]
        r_dec["decoder.rs<br/>push, decode_file"]
        r_csv["csv.rs"]
        r_feat["features.rs<br/>CSP + log variance"]
        r_filt["filter.rs<br/>biquads"]
        r_ring["ring.rs"]
        r_stim["stim.rs"]
        r_model["model.rs<br/>load + validate"]
        r_time["timing.rs"]
        r_err["error.rs"]
        r_py --> r_dec & r_model & r_csv
        r_bin --> r_replay & r_csv
        r_replay --> r_dec & r_csv & r_model
        r_dec --> r_csv & r_feat & r_filt & r_ring & r_stim & r_model
        r_feat --> r_ring & r_model
        r_filt --> r_model
        r_ring --> r_model
        r_csv --> r_model
        r_model --> r_time & r_err
    end

    subgraph CPP["C++20: cpp/"]
        c_py["python/bindings.cpp<br/>pybind11"]
        c_bin["tools/mi_replay.cpp"]
        c_replay["replay"]
        c_dec["decoder<br/>push, decode_file"]
        c_csv["csv"]
        c_feat["features"]
        c_filt["filter"]
        c_ring["ring"]
        c_stim["stim"]
        c_model["model"]
        c_json["json_strict<br/>refuses repeated keys"]
        c_time["timing"]
        c_err["error"]
        c_py --> c_dec & c_model & c_csv
        c_bin --> c_replay & c_csv & c_model
        c_replay --> c_dec & c_csv & c_model
        c_dec --> c_csv & c_feat & c_filt & c_ring & c_stim & c_model
        c_feat --> c_ring & c_model
        c_filt --> c_model
        c_ring --> c_model
        c_csv --> c_model
        c_model --> c_json & c_time & c_err
    end
```

The one structural difference is `json_strict` in C++. The C++ JSON library quietly keeps the last of two repeated keys, while Rust's refuses them, so C++ needs its own check to keep the contract.

## 5. How the checks fit around it

```mermaid
flowchart LR
    G[("golden/<br/>synthetic recordings,<br/>expected decisions,<br/>72 bad models,<br/>13 bad streams")]
    PUBD[("public EEG<br/>EEGMMIDB person 1 (CI)<br/>Cho 2017 person 1 (local)")]
    G --> PT["Python suite<br/>every core vs the reference"]
    PUBD --> PT
    G --> RT["Rust tests<br/>+ Miri + 3 fuzzers"]
    G --> CT["C++ tests<br/>+ ASan + UBSan + 2 fuzzers"]
    PT & RT & CT --> CI["GitHub Actions<br/>Ubuntu every push,<br/>macOS (public repo)"]
```
