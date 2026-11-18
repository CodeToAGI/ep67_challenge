"""EP67 - State space models and Mamba: a recurrence that is also a convolution, a selective state space block built from scratch, its cost against attention as the sequence grows, and a memory task where both are trained and then tested on longer sequences than they saw.
    python ep67_mamba.py     # 8-12 minutes on a laptop CPU; writes the mamba/ folder (dashboard.png) and ep67_results.json
pip install torch numpy matplotlib
# the Mamba block is written here in plain PyTorch (a simple loop, no fused GPU kernel), so the TIMES are this simple version's, not the official mamba-ssm kernel's. Toy scale on purpose.
"""
import json, math, os, platform, time, warnings
from importlib.metadata import version
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
warnings.filterwarnings("ignore")
# ── 1. CONFIG ─────────────────────────────────────────────────────────────
SEEDS = 2                            # every accuracy is a mean over this many independent training runs
D_MODEL, N_STATE, LAYERS = 32, 8, 2  # model width, state size per channel (the SSM's memory), blocks per model
KINDS = ["attention", "ssm_fixed", "mamba"]   # the transformer, a state space model that is the same at every step, and the selective one
V, M = 8, 4                          # the memory task: V kinds of token, M of them hidden among noise; repeat them in order at the end
TRAIN_L, STEPS, BATCH, LR = 32, 500, 64, 3e-3     # trained only on sequences of this length
TEST_LENS = [32, 64, 128, 256, 512]  # then tested on these, up to 16 times longer
BENCH_LENS, BENCH_D = [128, 256, 512, 1024, 2048, 4096], 64   # speed test: one layer, this width, these sequence lengths
OUT = Path(__file__).with_name("mamba")          # dashboard.png ends up here

# ── 2. MODELS + DATA: attention, the SSM block (fixed or selective), the memory task ──
def batch(B, L, g):                  # B sequences of L noise tokens (0); M data tokens (1..V) hidden at random places; M query tokens at the end; the answer is the M data tokens in order
    pos = torch.rand(B, L, generator=g).argsort(1)[:, :M].sort(1).values; tok = torch.randint(1, V + 1, (B, M), generator=g)
    x = torch.zeros(B, L + M, dtype=torch.long); x.scatter_(1, pos, tok); x[:, L:] = V + 1
    return x, tok
class SSM(nn.Module):                # the heart of Mamba. state h = A*h + B*u ; output y = C*h. selective=True: step size, B and C are computed FROM the input (it can choose what to keep)
    def __init__(s, d, N=N_STATE, selective=True, k=4):
        super().__init__(); s.di, s.N, s.sel = 2 * d, N, selective
        s.inp = nn.Linear(d, 2 * s.di); s.conv = nn.Conv1d(s.di, s.di, k, groups=s.di, padding=k - 1); s.out = nn.Linear(s.di, d)
        s.A_log = nn.Parameter(torch.log(torch.arange(1, N + 1).float()).repeat(s.di, 1)); s.D = nn.Parameter(torch.ones(s.di))
        if selective: s.x_proj = nn.Linear(s.di, 1 + 2 * N); s.dt_proj = nn.Linear(1, s.di)
        else: s.dt = nn.Parameter(torch.full((s.di,), -2.0)); s.Bp = nn.Parameter(torch.randn(N) * .5); s.Cp = nn.Parameter(torch.randn(N) * .5)
    def forward(s, x):
        B, L, _ = x.shape; u, z = s.inp(x).chunk(2, -1); u = F.silu(s.conv(u.transpose(1, 2))[..., :L].transpose(1, 2)); A = -torch.exp(s.A_log)
        if s.sel: dt, Bm, Cm = s.x_proj(u).split([1, s.N, s.N], -1); dt = F.softplus(s.dt_proj(dt))
        else: dt = F.softplus(s.dt).expand(B, L, -1); Bm = s.Bp.expand(B, L, -1); Cm = s.Cp.expand(B, L, -1)
        dA = torch.exp(dt.unsqueeze(-1) * A); dBu = dt.unsqueeze(-1) * Bm.unsqueeze(2) * u.unsqueeze(-1); h = torch.zeros(B, s.di, s.N); ys = []
        for t in range(L): h = dA[:, t] * h + dBu[:, t]; ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))      # one step per token: the cost grows LINEARLY with L
        return s.out((torch.stack(ys, 1) + u * s.D) * F.silu(z))
class Attn(nn.Module):               # causal self-attention: every token looks at every earlier token, so the score matrix is L x L
    def __init__(s, d, heads=2): super().__init__(); s.a = nn.MultiheadAttention(d, heads, batch_first=True)
    def forward(s, x): L = x.shape[1]; return s.a(x, x, x, attn_mask=torch.triu(torch.ones(L, L, dtype=torch.bool), 1), need_weights=False)[0]
class Res(nn.Module):
    def __init__(s, d, mix): super().__init__(); s.n, s.m = nn.LayerNorm(d), mix
    def forward(s, x): return x + s.m(s.n(x))
def pos_enc(L, d):                   # sinusoidal positions: the transformer needs them to know the ORDER of the answers
    p = torch.arange(L).unsqueeze(1); i = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.) / d)); o = torch.zeros(L, d); o[:, 0::2] = torch.sin(p * i); o[:, 1::2] = torch.cos(p * i); return o
class Net(nn.Module):
    def __init__(s, kind, d=D_MODEL):
        super().__init__(); s.kind, s.d = kind, d; s.emb = nn.Embedding(V + 2, d); s.head = nn.Linear(d, V + 1)
        mix = {"attention": lambda: Attn(d), "ssm_fixed": lambda: SSM(d, selective=False), "mamba": lambda: SSM(d, selective=True)}[kind]
        s.blocks = nn.Sequential(*[nn.Sequential(Res(d, mix()), Res(d, nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)))) for _ in range(LAYERS)])
    def forward(s, x): h = s.emb(x); h = h + pos_enc(x.shape[1], s.d) if s.kind == "attention" else h; return s.head(s.blocks(h))[:, -M:]

# ── 3. MEASURE: recurrence = convolution, the cost of one layer as the sequence grows ──
def equivalence(L=64, N=4, seed=0):  # a fixed (non-selective) SSM can be run as a recurrence OR as one convolution with the kernel K_k = C A^k B. do they agree?
    g = torch.Generator().manual_seed(seed); A = torch.rand(N, generator=g) * 0.5 + 0.4; Bv = torch.randn(N, generator=g); C = torch.randn(N, generator=g); x = torch.randn(L, generator=g)
    h = torch.zeros(N); rec = []
    for t in range(L): h = A * h + Bv * x[t]; rec.append(float((C * h).sum()))
    K = torch.stack([(C * A ** k * Bv).sum() for k in range(L)]); conv = [float((K[:t + 1].flip(0) * x[:t + 1]).sum()) for t in range(L)]
    return dict(L=L, N=N, input=[float(v) for v in x], recurrent=rec, convolution=conv, kernel=[float(v) for v in K], max_diff=float(np.abs(np.array(rec) - np.array(conv)).max()), scale=float(np.abs(rec).max()))
def timeit(f, reps=3):
    f(); ts = []
    for _ in range(reps): t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts))
def slope(xs, ys): return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])
def bench():                         # one layer, forward only, batch 1: seconds per pass, and the memory each one needs
    att, ssm = Attn(BENCH_D).eval(), SSM(BENCH_D).eval(); rows = []
    with torch.no_grad():
        for L in BENCH_LENS: x = torch.randn(1, L, BENCH_D); rows.append(dict(L=L, attention=timeit(lambda: att(x)), ssm=timeit(lambda: ssm(x)), score_floats=2 * L * L, cache_floats=2 * L * BENCH_D, state_floats=ssm.di * ssm.N))
    cross = next((r["L"] for r in rows if r["ssm"] < r["attention"]), None)
    return dict(d=BENCH_D, rows=rows, slope_attention=slope(BENCH_LENS, [r["attention"] for r in rows]), slope_ssm=slope(BENCH_LENS, [r["ssm"] for r in rows]), crossover=cross,
                slope_attention_tail=slope(BENCH_LENS[-3:], [r["attention"] for r in rows[-3:]]), slope_ssm_tail=slope(BENCH_LENS[-3:], [r["ssm"] for r in rows[-3:]]))

# ── 4. TRAIN + TEST: the memory task, trained on short sequences, tested on long ones ──
def accuracy(net, L, g, n=300):
    with torch.no_grad(): x, y = batch(n, L, g); return float((net(x).argmax(-1) == y).float().mean())
def train(kind, seed):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed); net = Net(kind); opt = torch.optim.AdamW(net.parameters(), LR); curve = []; t0 = time.time()
    for st in range(STEPS):
        x, y = batch(BATCH, TRAIN_L, g); loss = F.cross_entropy(net(x).reshape(-1, V + 1), y.reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
        if (st + 1) % 50 == 0: curve.append([st + 1, accuracy(net, TRAIN_L, g, 200)])
    ge = torch.Generator().manual_seed(1000 + seed)
    return dict(curve=curve, test={L: accuracy(net, L, ge) for L in TEST_LENS}, params=sum(p.numel() for p in net.parameters()), seconds=time.time() - t0)
def ms(v): return [float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0]

# ── 5. DASHBOARD ──────────────────────────────────────────────────────────
def dashboard(R, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": "#050815", "axes.facecolor": "#0d1117", "text.color": "#F8F8F2", "axes.labelcolor": "#F8F8F2", "xtick.color": "#C8E8F8", "ytick.color": "#C8E8F8", "axes.edgecolor": "#4A7A90"})
    col = {"attention": "#ff79c6", "ssm_fixed": "#f1fa8c", "mamba": "#50fa7b"}; fig, ax = plt.subplots(2, 2, figsize=(13, 7.2)); fig.suptitle("State space models: what a loop can and cannot do", color="#8be9fd")
    e = R["equiv"]; a = ax[0, 0]; a.plot(e["recurrent"], color="#8be9fd", lw=3, label="recurrence (one step at a time)"); a.plot(e["convolution"], "--", color="#ff5555", lw=1.5, label="one convolution"); a.legend(fontsize=7)
    a.set_title(f"Same output, two ways (largest difference {e['max_diff']:.1e})", fontsize=10); a.set_xlabel("position")
    b = R["bench"]; a = ax[0, 1]; Ls = [r["L"] for r in b["rows"]]; a.loglog(Ls, [r["attention"] for r in b["rows"]], "-o", color=col["attention"], label=f"attention (slope {b['slope_attention']:.2f})"); a.loglog(Ls, [r["ssm"] for r in b["rows"]], "-o", color=col["mamba"], label=f"SSM loop (slope {b['slope_ssm']:.2f})")
    a.legend(fontsize=7); a.set_title("Seconds for one layer against sequence length", fontsize=10); a.set_xlabel("sequence length")
    a = ax[1, 0]
    for k in KINDS: a.plot([c[0] for c in R["curves"][k]], [c[1] for c in R["curves"][k]], color=col[k], label=k)
    a.axhline(1 / V, color="#6272a4", ls=":"); a.legend(fontsize=7); a.set_title(f"Learning the memory task (length {TRAIN_L})", fontsize=10); a.set_xlabel("training step"); a.set_ylabel("accuracy")
    a = ax[1, 1]
    for k in KINDS: a.errorbar(TEST_LENS, [R["test"][k][str(L)][0] for L in TEST_LENS], [R["test"][k][str(L)][1] for L in TEST_LENS], fmt="-o", color=col[k], label=k)
    a.set_xscale("log", base=2); a.axvline(TRAIN_L, color="#6272a4", ls="--"); a.axhline(1 / V, color="#6272a4", ls=":"); a.legend(fontsize=7); a.set_title("Tested on longer sequences than it trained on", fontsize=10); a.set_xlabel("test sequence length")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)

# ── 6. RUN: the equivalence, the cost test, the three models, one results file ──
def run():
    OUT.mkdir(exist_ok=True); torch.set_num_threads(max(1, min(4, os.cpu_count() or 1))); t0 = time.time(); eq = equivalence(); bn = bench(); runs = {k: [train(k, s) for s in range(SEEDS)] for k in KINDS}
    R = dict(ver={p: version(p) for p in ("torch", "numpy", "matplotlib")}, cores=os.cpu_count(), cpu=platform.processor() or platform.machine(), seeds=SEEDS, d_model=D_MODEL, n_state=N_STATE, layers=LAYERS, kinds=KINDS, v=V, m=M, train_l=TRAIN_L, steps=STEPS,
             test_lens=TEST_LENS, equiv=eq, bench=bn, curves={k: [[c[0], float(np.mean([r["curve"][i][1] for r in runs[k]]))] for i, c in enumerate(runs[k][0]["curve"])] for k in KINDS},
             test={k: {str(L): ms([r["test"][L] for r in runs[k]]) for L in TEST_LENS} for k in KINDS}, params={k: runs[k][0]["params"] for k in KINDS}, train_seconds={k: float(np.mean([r["seconds"] for r in runs[k]])) for k in KINDS}, seconds=time.time() - t0)
    dashboard(R, OUT / "dashboard.png"); Path(__file__).with_name("ep67_results.json").write_text(json.dumps(R)); return R

if __name__ == "__main__":
    R = run(); e, b = R["equiv"], R["bench"]; f = lambda m: f"{m[0]:.2f} ± {m[1]:.2f}"
    print(f"{R['cores']} CPU core(s); torch {R['ver']['torch']}; {R['seeds']} runs per model; {R['seconds']:.0f} s")
    print(f"  recurrence vs convolution: largest difference {e['max_diff']:.2e} (outputs are up to {e['scale']:.2f})")
    print("  seconds per layer:", " | ".join(f"L={r['L']}: attention {r['attention']:.4f}, SSM {r['ssm']:.4f}" for r in b["rows"]), f"| slopes {b['slope_attention']:.2f} and {b['slope_ssm']:.2f} (last three points: {b['slope_attention_tail']:.2f} and {b['slope_ssm_tail']:.2f}); SSM faster from length {b['crossover']}")
    print(f"  memory per layer at length {b['rows'][-1]['L']}: attention scores {b['rows'][-1]['score_floats']:,} numbers, attention cache {b['rows'][-1]['cache_floats']:,}, SSM state {b['rows'][-1]['state_floats']:,}")
    for k in KINDS: print(f"  {k} ({R['params'][k]:,} parameters, {R['train_seconds'][k]:.0f} s to train):", ", ".join(f"length {L}: {f(R['test'][k][str(L)])}" for L in TEST_LENS))
