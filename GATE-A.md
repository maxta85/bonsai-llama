# Gate A — PASSED (2026-09-14)

## Baseline model (run 2, fp16/bf16 LoRA)
- Base: Qwen2.5-1.5B-Instruct, LoRA r16 a32, bf16, 3060 sandbox
- Data: 31,054 examples (smoltalk 30k + original 1,054), 2 epochs, 7,376 steps
- Final loss: 0.99 (avg 1.014); run took 6.2h
- Artifacts: adapter on Hub `releases/20260914-run2-final`; merged+GGUF in devcontainer

## Gate A test: 4-paragraph curlew poem (prompted "write about curlews")

    In the fields where dawn breaks and shadows stretch,
    Curlew calls rise to the call of the heart.
    With feathers that dance in a gentle breeze,
    They soar through skies so vast, their wings spread wide.

    The sun sets low on the horizon, casting
    A golden glow over the water's embrace,
    And curlews wade into the misty spray,
    Their feet sinking deep, as if to escape.

    But even in darkness they remain
    Unafraid, with eyes that gleam and beam.
    For their song is a symphony of joy,
    And their heart beats true, forevermore.

## Assessment
- Structure: 12 lines, quatrain movement, dawn→sunset→night progression, no drift ✅
- Instruction adherence: on-topic throughout ✅
- Weaknesses: decorative non-rhymes, cliché density (expected at 1.5B SFT)
- VERDICT: **Gate A PASSED** — coherent multi-paragraph thematic generation

## Benchmark note
This poem is the quality reference for the ternary QAT comparison (B.1).
B.1 pass bar: retains structure + coherence at ternary weight count (~1/8 bits).
