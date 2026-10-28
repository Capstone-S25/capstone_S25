import pickle
import sys

sys.path.insert(0, "/Users/ianvarella/capstone_S25/notebooks")
from stage1_bench import avenue_sequences, littering_sequences, run_perception  # noqa: E402
from stage1_gate import GateConfig, Stage1Gate  # noqa: E402

D = "/Users/ianvarella/capstone_S25/data"
OUT = sys.argv[1]
which = sys.argv[2]
if which == "litter":
    for split in ("train", "valid", "test"):
        seqs = littering_sequences(f"{D}/littering_clips", splits=(split,))
        res = run_perception(Stage1Gate(GateConfig()), seqs, verbose=False)
        assert sum(len(r.observations) for _, r in res) == sum(len(g.seq) for g in seqs)
        pickle.dump(res, open(f"{OUT}/litter_{split}.pkl", "wb"))
        print(f"litter_{split} done {len(seqs)}", flush=True)
if which == "avenue":
    seqs = avenue_sequences(f"{D}/loitering", f"{D}/loitering/avenue_gt/ground_truth_demo/testing_label_mask", stride=2)
    res = run_perception(Stage1Gate(GateConfig()), seqs, verbose=False)
    pickle.dump(res, open(f"{OUT}/avenue.pkl", "wb"))
    print("avenue done", len(seqs), flush=True)
