"""CPAF-Net family (LPCF + fusion ablations) and neural baselines, all with identical heads/losses/training budget."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
from data import CFG

D = CFG['hidden']; KCAP = CFG['K_cap']; S_DEF = 4


class TempAttn(nn.Module):
    def __init__(self, d):
        super().__init__(); self.W = nn.Linear(d, d); self.v = nn.Linear(d, 1, bias=False)
    def forward(self, H):                                   # H [B,L,d]
        a = torch.softmax(self.v(torch.tanh(self.W(H))).squeeze(-1), dim=1)
        return (a.unsqueeze(-1) * H).sum(1), a


class SigEnc(nn.Module):
    """BiLSTM + temporal attention for one input stream (1 channel, or S channels for early concatenation)."""
    def __init__(self, c_in, d=D):
        super().__init__(); self.rnn = nn.LSTM(c_in, d // 2, batch_first=True, bidirectional=True); self.att = TempAttn(d)
    def forward(self, x):
        H, _ = self.rnn(x); return self.att(H)


class GRN(nn.Module):
    def __init__(self, i, h, o):
        super().__init__(); self.a = nn.Linear(i, h); self.b = nn.Linear(h, o * 2); self.o = o
    def forward(self, x):
        y = self.b(Fn.elu(self.a(x))); return y[..., :self.o] * torch.sigmoid(y[..., self.o:])


class CPAF(nn.Module):
    """fusion in {'lpcf','k1','fixed','concat','tft'}.  lpcf with K phases; K=1 -> learned fixed weights.
    phi_to_end: pass the phase distribution to the endurance head (LPCF design)."""
    def __init__(self, fusion='lpcf', S=S_DEF, K=3, L=14, phi_to_end=True, d=D):
        super().__init__()
        self.fusion, self.S, self.K, self.phi_to_end = fusion, S, K, (phi_to_end and fusion == 'lpcf')
        if fusion == 'concat': self.enc = SigEnc(S, d)
        else: self.enc = nn.ModuleList([SigEnc(1, d) for _ in range(S)])
        if fusion == 'lpcf':
            self.penc = nn.GRU(2, d, batch_first=True); self.pproj = nn.Linear(d, K)
            self.u = nn.Parameter(0.01 * torch.randn(K, S))         # phase-specific signal logits u_k
        elif fusion == 'k1': self.u = nn.Parameter(torch.zeros(1, S))
        elif fusion == 'tft': self.sel = GRN(L * S, d, S)           # input-dependent variable selection
        self.trunk = nn.Sequential(nn.Linear(d, d), nn.ReLU())
        self.hp = nn.Linear(d, 1)
        self.he = nn.Linear(d + (K if self.phi_to_end else 0), 1)

    def forward(self, x):                                     # x [B,L,S]
        B = x.shape[0]; phi = None; w = None; att = None
        if self.fusion == 'concat':
            z, att = self.enc(x)
        else:
            hs = []
            for s in range(self.S):
                h, a = self.enc[s](x[:, :, s:s + 1]); hs.append(h)
            Hs = torch.stack(hs, 1)                              # [B,S,d]
            if self.fusion == 'fixed': w = torch.full((B, self.S), 1.0 / self.S)
            elif self.fusion == 'k1': w = torch.softmax(self.u, -1).expand(B, -1)
            elif self.fusion == 'tft': w = torch.softmax(self.sel(x.reshape(B, -1)), -1)
            else:
                p = x[:, :, 0]; g = torch.diff(p, dim=1, prepend=p[:, :1])
                _, hn = self.penc(torch.stack([p, g], -1)); phi = torch.softmax(self.pproj(hn[-1]), -1)   # [B,K]
                A = torch.softmax(self.u, -1)                    # a_{k,s}
                w = phi @ A                                      # w_s = sum_k phi_k a_{k,s}  (convex combination)
            z = (w.unsqueeze(-1) * Hs).sum(1)
        t = self.trunk(z)
        pop = self.hp(t).squeeze(-1)
        end = self.he(torch.cat([t, phi], -1) if self.phi_to_end else t).squeeze(-1)
        return dict(pop=pop, end=end, phi=phi, w=w)


class LSTMBase(nn.Module):
    def __init__(self, S=S_DEF, d=24):
        super().__init__(); self.rnn = nn.LSTM(S, d, batch_first=True); self.hp = nn.Linear(d, 1); self.he = nn.Linear(d, 1)
    def forward(self, x):
        H, _ = self.rnn(x); h = H[:, -1]; return dict(pop=self.hp(h).squeeze(-1), end=self.he(h).squeeze(-1), phi=None, w=None)


class DLinear(nn.Module):
    """Decomposition-linear (Zeng et al. 2023) on the popularity channel (channel-independent as in the original); two outputs."""
    def __init__(self, L=14, k=5):
        super().__init__(); self.k = k; self.ls = nn.Linear(L, 2); self.lt = nn.Linear(L, 2)
    def forward(self, x):
        p = x[:, :, 0]; pad = Fn.pad(p.unsqueeze(1), (self.k // 2, self.k // 2), mode='replicate')
        trend = Fn.avg_pool1d(pad, self.k, 1).squeeze(1); seas = p - trend
        y = self.ls(seas) + self.lt(trend); return dict(pop=y[:, 0], end=y[:, 1], phi=None, w=None)


class TFTLite(nn.Module):
    """Compact TFT-style model: per-step GRN variable selection -> LSTM -> gated skip -> attention -> GRN heads (own re-implementation)."""
    def __init__(self, S=S_DEF, d=D):
        super().__init__(); self.emb = nn.ModuleList([nn.Linear(1, d) for _ in range(S)]); self.sel = GRN(S, d, S)
        self.rnn = nn.LSTM(d, d, batch_first=True); self.gate = GRN(d, d, d)
        self.mha = nn.MultiheadAttention(d, 2, batch_first=True); self.post = GRN(d, d, d)
        self.hp = nn.Linear(d, 1); self.he = nn.Linear(d, 1); self.S = S
    def forward(self, x):
        w = torch.softmax(self.sel(x), -1)                               # [B,L,S]
        e = torch.stack([self.emb[s](x[:, :, s:s + 1]) for s in range(self.S)], 2)   # [B,L,S,d]
        v = (w.unsqueeze(-1) * e).sum(2); H, _ = self.rnn(v); H = H + self.gate(H)
        a, _ = self.mha(H[:, -1:], H, H); h = self.post(a[:, 0] + H[:, -1])
        return dict(pop=self.hp(h).squeeze(-1), end=self.he(h).squeeze(-1), phi=None, w=w.mean(1))


NEURAL = {
    'LPCF (K=3)':            lambda S: CPAF('lpcf', S, K=3),
    'LPCF (K=3, no phi->E)': lambda S: CPAF('lpcf', S, K=3, phi_to_end=False),
    'LPCF (K=2)':            lambda S: CPAF('lpcf', S, K=2),
    'K=1 (learned fixed w)': lambda S: CPAF('k1', S),
    'Uniform fixed w':       lambda S: CPAF('fixed', S),
    'Early concat':          lambda S: CPAF('concat', S),
    'TFT-style selection':   lambda S: CPAF('tft', S),
    'LSTM':                  lambda S: LSTMBase(S),
    'DLinear':               lambda S: DLinear(),
    'TFT-lite':              lambda S: TFTLite(S),
}
SUP_MODELS = {'LPCF (K=3) + phase sup.'}     # same architecture as 'LPCF (K=3)', lambda_phi>0


def losses(out, y, lam_e=0.5, lam_phi=0.0):
    pop_l = Fn.smooth_l1_loss(out['pop'], y['dpop'])
    pe, e, c = out['end'], y['end'], y['cen']
    unc = (c < 0.5).float()
    # right-censoring aware: exact error on observed endurance, one-sided hinge when only a lower bound is known
    l_end = (unc * (pe - e) ** 2 + (1 - unc) * Fn.relu(e - pe) ** 2).mean()
    l = pop_l + lam_e * l_end
    if lam_phi > 0 and out['phi'] is not None:
        l = l + lam_phi * Fn.nll_loss(torch.log(out['phi'] + 1e-8), y['phase'])
    return l


def to_t(d):
    return dict(X=torch.tensor(d['X']), dpop=torch.tensor(d['dpop'], dtype=torch.float32),
                end=torch.tensor(d['end'] / KCAP, dtype=torch.float32), cen=torch.tensor(d['cen'].astype(np.float32)),
                phase=torch.tensor(d['phase'], dtype=torch.long))


def fit(model, tr, seed, epochs=None, lr=None, lam_phi=0.0, bs=None, val=None, patience=12, gate_mult=1.0):
    """Adam training. If `val` (chronologically later than `tr`, labels known at cut-off) is given: early stopping on the validation loss."""
    torch.manual_seed(seed); np.random.seed(seed)
    epochs = epochs or CFG['epochs']; lr = lr or CFG['lr']; bs = bs or CFG['batch']
    y = to_t(tr); n = len(y['X'])
    gate = [p for nme, p in model.named_parameters() if nme == 'u']; rest = [p for nme, p in model.named_parameters() if nme != 'u']
    opt = torch.optim.Adam([dict(params=rest, lr=lr), dict(params=gate, lr=lr * gate_mult, weight_decay=0.0)] if gate else [dict(params=rest, lr=lr)], weight_decay=CFG['weight_decay'])   # gate_mult: sensitivity check on under-training of the signal logits u
    yv = to_t(val) if val is not None else None
    best, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            ix = perm[i:i + bs]
            if len(ix) < 2: continue
            out = model(y['X'][ix]); l = losses(out, {k: v[ix] for k, v in y.items()}, lam_phi=lam_phi)
            opt.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if yv is not None:
            model.eval()
            with torch.no_grad(): lv = losses(model(yv['X']), yv, lam_phi=0.0).item()
            if lv < best - 1e-5: best, bad = lv, 0; best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience: break
    if best_state is not None: model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict(model, d):
    model.eval(); out = model(torch.tensor(d['X']))
    res = dict(dpop=out['pop'].numpy(), end=np.clip(out['end'].numpy() * KCAP, 0, KCAP))
    if out['phi'] is not None: res['phi'] = out['phi'].numpy()
    if out['w'] is not None: res['w'] = out['w'].numpy()
    return res
