/* ==========================================================================
   ANN Course — nn.js
   A real reverse-mode autograd engine, a small matrix layer, an MLP, and the
   helpers the course's demos need. Vanilla ES2020, no imports, globals at the
   bottom. Include with <script src="assets/nn.js"></script> AFTER course.js.

   Design notes for the reader (the course may put this file on screen):

   - The scalar `Value` engine is micrograd-shaped on purpose. It is slow, and
     that is fine: the course's spine is a 2-2-1 network with NINE parameters,
     and a scalar graph lets a demo point at one edge and say "that number,
     right there, is dL/dw". Traceability beats throughput at nine parameters.
   - `Mat` exists for the demos scalars cannot carry: attention maps, embedding
     geometry, matmul shape drills. Flat Float32Array + explicit shape, because
     the shape bookkeeping IS the lesson.
   - Everything that samples takes an explicit RNG. A learner who reloads the
     page must see the identical run, or the demo is not evidence of anything.
   ========================================================================== */
"use strict";

/* Everything lives inside this IIFE and escapes only through window.NN at the
   bottom. This is not ceremony: a top-level `class Value` in a classic script
   creates a binding in the GLOBAL LEXICAL scope, and then any page that writes
   `const {Value} = NN;` dies with "Identifier 'Value' has already been
   declared" — a SyntaxError, so the whole page script never runs and the demo
   silently shows nothing. Found by exactly that failure in nn.test.html. */
(function (window) {

/* ==========================================================================
   1. SEEDED RNG
   Math.random() cannot be seeded, so a reload would reshuffle every demo and
   the learner could never compare two runs. mulberry32: 32-bit state, passes
   gjrand, ~2^32 period — far more than enough for a page of demos, and short
   enough to read.
   ========================================================================== */

/* mulberry32(seed) -> function(): float in [0,1) */
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

class RNG {
  constructor(seed) {
    this.seed = (seed === undefined ? 1337 : seed) >>> 0;
    this._next = mulberry32(this.seed);
    this._spare = null; // Box-Muller produces two normals per call; keep the second
  }
  /* Restore the exact starting state — this is what a demo's "Reset" button wants. */
  reset(seed) {
    this.seed = (seed === undefined ? this.seed : seed) >>> 0;
    this._next = mulberry32(this.seed);
    this._spare = null;
    return this;
  }
  random() { return this._next(); }
  uniform(a, b) { return a + (b - a) * this._next(); }
  int(n) { return Math.floor(this._next() * n); }
  /* Box-Muller. The diffusion demos need Gaussians and they need them seeded.
     u must be strictly > 0 or log(0) = -Infinity, hence the reject-zero loop. */
  normal(mu, sigma) {
    mu = mu === undefined ? 0 : mu;
    sigma = sigma === undefined ? 1 : sigma;
    if (this._spare !== null) {
      const s = this._spare; this._spare = null;
      return mu + sigma * s;
    }
    let u = 0, v = 0;
    while (u === 0) u = this._next();
    v = this._next();
    const r = Math.sqrt(-2 * Math.log(u));
    const th = 2 * Math.PI * v;
    this._spare = r * Math.sin(th);
    return mu + sigma * (r * Math.cos(th));
  }
  /* Fisher-Yates, in place. Deterministic given the seed. */
  shuffle(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = this.int(i + 1);
      const t = arr[i]; arr[i] = arr[j]; arr[j] = t;
    }
    return arr;
  }
  pick(arr) { return arr[this.int(arr.length)]; }
}

/* ==========================================================================
   2. SCALAR REVERSE-MODE AUTOGRAD — `Value`
   ========================================================================== */

let _valueUid = 0;

class Value {
  /* new Value(data, children, op, label) — children/op/label are internal. */
  constructor(data, _children, _op, label) {
    if (typeof data !== "number" || !isFinite(data)) {
      // A NaN slipping into a graph silently is the single most confusing bug a
      // learner can hit: the loss goes NaN twenty steps later and nothing points
      // back here. Fail loudly, at the source. (Infinity is allowed through
      // arithmetic below — this guard is only on explicit construction.)
      if (typeof data !== "number") {
        throw new TypeError("Value(data): expected a number, got " + typeof data + " (" + data + ")");
      }
    }
    this.data = data;
    this.grad = 0;
    this._prev = _children || [];
    this._op = _op || "";
    this.label = label || "";
    this.uid = _valueUid++;
    this._backward = null; // set by each op; null on leaves
  }

  get isLeaf() { return this._prev.length === 0; }

  /* Coerce a number to a constant Value; pass Values through untouched. */
  static wrap(x) { return x instanceof Value ? x : new Value(x); }
  static get(x) { return x instanceof Value ? x.data : x; }

  /* ---- arithmetic ------------------------------------------------------ */

  add(other) {
    const o = Value.wrap(other);
    const out = new Value(this.data + o.data, [this, o], "+");
    out._backward = () => {
      // d(a+b)/da = 1, d/db = 1 — the gradient just splits and flows to both.
      this.grad += out.grad;
      o.grad += out.grad;
    };
    return out;
  }

  mul(other) {
    const o = Value.wrap(other);
    const out = new Value(this.data * o.data, [this, o], "*");
    out._backward = () => {
      // Each factor's gradient is the OTHER factor. This one line is the whole
      // reason "error x activation" shows up everywhere in backprop.
      this.grad += o.data * out.grad;
      o.grad += this.data * out.grad;
    };
    return out;
  }

  neg() { return this.mul(-1); }

  sub(other) { return this.add(Value.wrap(other).neg()); }

  /* pow(k): k is a CONSTANT exponent (number). x**k, dx = k*x**(k-1). */
  pow(k) {
    if (typeof k !== "number") {
      throw new TypeError("Value.pow(k): exponent must be a plain number constant, got " +
        (k instanceof Value ? "a Value (use a.log().mul(b).exp() for a variable exponent)" : typeof k));
    }
    const out = new Value(Math.pow(this.data, k), [this], "**" + k);
    out._backward = () => {
      this.grad += k * Math.pow(this.data, k - 1) * out.grad;
    };
    return out;
  }

  div(other) {
    const o = Value.wrap(other);
    // a/b implemented as a * b**-1 would give the right gradient, but building
    // it directly keeps the graph one node shallower and the readout legible.
    const out = new Value(this.data / o.data, [this, o], "/");
    out._backward = () => {
      this.grad += out.grad / o.data;
      o.grad += -this.data / (o.data * o.data) * out.grad;
    };
    return out;
  }

  /* ---- activations ----------------------------------------------------- */

  tanh() {
    const t = Math.tanh(this.data);
    const out = new Value(t, [this], "tanh");
    out._backward = () => {
      // tanh'(z) = 1 - tanh(z)^2 = 1 - a^2. The course derives this by hand and
      // then watches (1-a^2) act as the gate that destroys gradient at
      // saturation. Same expression, same place.
      this.grad += (1 - t * t) * out.grad;
    };
    return out;
  }

  sigmoid() {
    const s = _sigmoid(this.data);
    const out = new Value(s, [this], "sigmoid");
    out._backward = () => {
      this.grad += s * (1 - s) * out.grad;
    };
    return out;
  }

  relu() {
    const out = new Value(this.data > 0 ? this.data : 0, [this], "relu");
    out._backward = () => {
      // Subgradient at exactly 0 is chosen to be 0, matching PyTorch. Finite
      // differences DISAGREE at x=0 (they give 0.5) — that is the kink, not a
      // bug, and the test suite checks the convention rather than the FD value.
      this.grad += (this.data > 0 ? 1 : 0) * out.grad;
    };
    return out;
  }

  exp() {
    const e = Math.exp(this.data);
    const out = new Value(e, [this], "exp");
    out._backward = () => {
      this.grad += e * out.grad; // d/dx e^x = e^x — it is its own derivative
    };
    return out;
  }

  log() {
    if (this.data <= 0) {
      throw new RangeError("Value.log(): log of " + this.data + " is undefined (need > 0). " +
        "If this came from a loss, your probability underflowed to 0 — use a logits-based loss " +
        "(bceWithLogits / crossEntropy), which never forms the probability explicitly.");
    }
    const out = new Value(Math.log(this.data), [this], "log");
    out._backward = () => {
      this.grad += out.grad / this.data;
    };
    return out;
  }

  /* ---- graph ----------------------------------------------------------- */

  /* Topological order, children before parents. Iterative, not recursive: a
     100-step unrolled graph is thousands of nodes deep and a recursive DFS
     blows the JS stack on exactly the demos that matter most. */
  topo() {
    const order = [];
    const visited = new Set();
    // stack of [node, childIndex] — emulate the post-order recursion explicitly
    const stack = [[this, 0]];
    visited.add(this.uid);
    while (stack.length) {
      const frame = stack[stack.length - 1];
      const node = frame[0];
      if (frame[1] < node._prev.length) {
        const child = node._prev[frame[1]++];
        if (!visited.has(child.uid)) {
          visited.add(child.uid);
          stack.push([child, 0]);
        }
      } else {
        order.push(node);
        stack.pop();
      }
    }
    return order; // children strictly before parents
  }

  /* backward(seed=1) — populate .grad throughout the graph.

     GRADIENT ACCUMULATION SEMANTICS (matching PyTorch exactly — the course
     makes a teaching point of this):
       - LEAF grads ACCUMULATE. Calling backward() twice without zeroing gives
         leaves 2x the gradient. That is not a bug; it is what makes
         gradient accumulation over micro-batches work, and it is why forgetting
         zero_grad() blows a model up in 20-100 steps. A demo needs to show that,
         so the engine must reproduce it faithfully.
       - NON-LEAF grads are RESET each call. PyTorch does not retain grads on
         intermediate nodes; they are transient buffers of THIS backward pass.
         If we let them accumulate too, a second backward() would feed 2x through
         the interior and leaves would end up at 3x, not 2x — wrong, and it would
         make the zero_grad demo teach a number that PyTorch never prints. */
  backward(seed) {
    const order = this.topo();
    for (let i = 0; i < order.length; i++) {
      if (!order[i].isLeaf) order[i].grad = 0;
    }
    // If `this` is a leaf (a lone Value), += is the accumulate PyTorch does.
    // If it is not, it was just zeroed, so += is identical to =. One line, both.
    this.grad += (seed === undefined ? 1 : seed);
    for (let i = order.length - 1; i >= 0; i--) {
      if (order[i]._backward) order[i]._backward();
    }
    return this;
  }

  /* Zero this node and everything upstream of it. Rarely what you want —
     optimizers zero their own param list, which is the PyTorch pattern. */
  zeroGradGraph() {
    const order = this.topo();
    for (let i = 0; i < order.length; i++) order[i].grad = 0;
    return this;
  }

  toString() {
    return "Value(data=" + this.data.toFixed(6) + ", grad=" + this.grad.toFixed(6) +
      (this.label ? ", label=" + this.label : "") + ")";
  }
}

/* free-function conveniences ------------------------------------------------ */
function V(x, label) { const v = new Value(x); if (label) v.label = label; return v; }

/* Sum an array of Values (or numbers). Left fold; graph depth is O(n), which is
   fine at the scales this course trains at. */
function vsum(arr) {
  let acc = Value.wrap(arr.length ? arr[0] : 0);
  for (let i = 1; i < arr.length; i++) acc = acc.add(arr[i]);
  return acc;
}

function _sigmoid(z) {
  // Two branches. exp(-z) overflows for z very negative and exp(z) overflows for
  // z very positive; each branch is evaluated only where its exp is <= 1.
  if (z >= 0) { const e = Math.exp(-z); return 1 / (1 + e); }
  const e = Math.exp(z); return e / (1 + e);
}

/* ==========================================================================
   3. LOSSES (on Values)
   ========================================================================== */

/* mseLoss(pred, target) — pred: Value|Value[], target: number|number[].
   Mean over elements, matching torch.nn.MSELoss(reduction='mean'). */
function mseLoss(pred, target) {
  const p = Array.isArray(pred) ? pred : [pred];
  const t = Array.isArray(target) ? target : [target];
  if (p.length !== t.length) {
    throw new Error("mseLoss: pred has " + p.length + " elements but target has " + t.length);
  }
  const terms = p.map((pi, i) => Value.wrap(pi).sub(t[i]).pow(2));
  return vsum(terms).div(p.length);
}

/* bceLoss(prob, y) — prob is already a probability (post-sigmoid), y in {0,1}.
   Numerically fragile by construction: if prob saturates to exactly 0 or 1,
   log() throws. That is deliberate — the course uses this to motivate
   bceWithLogits. Use bceWithLogits in anything that trains. */
function bceLoss(prob, y) {
  const p = Array.isArray(prob) ? prob : [prob];
  const t = Array.isArray(y) ? y : [y];
  if (p.length !== t.length) {
    throw new Error("bceLoss: pred has " + p.length + " elements but target has " + t.length);
  }
  const terms = p.map((pi, i) => {
    const P = Value.wrap(pi);
    const yi = t[i];
    // -[y*log(p) + (1-y)*log(1-p)]
    return P.log().mul(yi).add(P.neg().add(1).log().mul(1 - yi)).neg();
  });
  return vsum(terms).div(p.length);
}

/* bceWithLogits(z, y) — the one to actually train with.
   L = max(z,0) - z*y + log(1 + exp(-|z|))  — the standard stable rearrangement.
   Same value as bceLoss(sigmoid(z), y), but never forms the probability, so it
   cannot underflow to log(0). Gradient is the famous dL/dz = sigmoid(z) - y. */
function bceWithLogits(z, y) {
  const zs = Array.isArray(z) ? z : [z];
  const ts = Array.isArray(y) ? y : [y];
  if (zs.length !== ts.length) {
    throw new Error("bceWithLogits: got " + zs.length + " logits but " + ts.length + " targets");
  }
  const terms = zs.map((zi, i) => {
    const Z = Value.wrap(zi);
    const yi = ts[i];
    // Build with primitive ops so autograd handles it — no hand-written grad.
    // softplus(-|z|) via log(1+exp(-|z|)): the exp argument is <= 0, so <= 1.
    // BOTH branches must test the SAME condition (>= 0). |z| and max(z,0) each
    // have a kink at z=0, and the kinks cancel exactly — L is really just
    // softplus(z) - z*y, which is smooth everywhere. Splitting the branches
    // differently (say `>= 0` here and `> 0` there) picks one-sided derivatives
    // from opposite sides at z=0 and the cancellation breaks: you get
    // dL/dz = -y - 0.5 instead of the correct sigmoid(0) - y = 0.5 - y.
    // Wrong only at exactly z=0 — invisible in training, caught by gradcheck.
    const absz = Z.data >= 0 ? Z : Z.neg();          // |z|
    const sp = absz.neg().exp().add(1).log();        // log(1 + e^{-|z|})
    const mx = Z.data >= 0 ? Z : Value.wrap(0);      // max(z, 0)
    return mx.sub(Z.mul(yi)).add(sp);
  });
  return vsum(terms).div(zs.length);
}

/* logSumExp(values) — stable: subtract the max as a CONSTANT.
   Subtracting a constant is exact, not an approximation: logsumexp(z) =
   m + logsumexp(z - m) identically. The constant carries no gradient, and it
   does not need to: the identity holds for any m. */
function logSumExp(values) {
  const vs = values.map(Value.wrap);
  let m = -Infinity;
  for (const v of vs) if (v.data > m) m = v.data;
  if (!isFinite(m)) m = 0; // all -Infinity (fully masked row): fall back gracefully
  const terms = vs.map(v => v.sub(m).exp());
  return vsum(terms).log().add(m);
}

/* crossEntropyLoss(logits, targetIdx) — softmax + NLL fused, stable.
   L = -z[t] + logsumexp(z). Never forms exp(z) at full scale, so logits of 1000
   are harmless. Matches torch.nn.CrossEntropyLoss (which takes LOGITS, not
   probabilities — a top-5 beginner error the course calls out). */
function crossEntropyLoss(logits, targetIdx) {
  if (!Array.isArray(logits)) throw new Error("crossEntropyLoss: logits must be an array of Values");
  if (!(targetIdx >= 0 && targetIdx < logits.length)) {
    throw new RangeError("crossEntropyLoss: target index " + targetIdx +
      " is out of range for " + logits.length + " classes (valid: 0.." + (logits.length - 1) + ")");
  }
  return logSumExp(logits).sub(Value.wrap(logits[targetIdx]));
}

/* softmaxValues(logits) -> Value[] of probabilities (stable). For demos that
   need to SHOW the probabilities; train with crossEntropyLoss instead. */
function softmaxValues(logits) {
  const lse = logSumExp(logits);
  return logits.map(z => Value.wrap(z).sub(lse).exp());
}

/* ==========================================================================
   4. MLP on the Value engine
   ========================================================================== */

const ACTIVATIONS = {
  tanh: v => v.tanh(),
  relu: v => v.relu(),
  sigmoid: v => v.sigmoid(),
  identity: v => v,
  linear: v => v
};

class Neuron {
  /* new Neuron(nin, {act, rng, init}) */
  constructor(nin, opts) {
    opts = opts || {};
    const rng = opts.rng || new RNG(1);
    const act = opts.act || "tanh";
    if (!ACTIVATIONS[act]) {
      throw new Error("Neuron: unknown activation '" + act + "'. Known: " + Object.keys(ACTIVATIONS).join(", "));
    }
    // Init scale: He for ReLU (gain 2), Xavier/Glorot otherwise. Getting this
    // wrong is invisible at 2 layers and fatal at 20 — the course shows both.
    let scale;
    if (opts.init === "zeros") scale = 0;
    else if (opts.initScale !== undefined) scale = opts.initScale;
    else scale = act === "relu" ? Math.sqrt(2 / nin) : Math.sqrt(1 / nin);
    this.w = [];
    for (let i = 0; i < nin; i++) this.w.push(new Value(rng.normal(0, 1) * scale));
    this.b = new Value(0);
    this.act = act;
    this.nin = nin;
  }
  /* forward(x: (Value|number)[]) -> Value */
  forward(x) {
    if (x.length !== this.nin) {
      throw new Error("Neuron.forward: expected " + this.nin + " inputs, got " + x.length);
    }
    let z = this.b;
    for (let i = 0; i < this.nin; i++) z = z.add(this.w[i].mul(x[i]));
    this.z = z; // exposed so a demo can print the pre-activation
    return ACTIVATIONS[this.act](z);
  }
  parameters() { return this.w.concat([this.b]); }
}

class Layer {
  /* new Layer(nin, nout, {act, rng, init}) */
  constructor(nin, nout, opts) {
    this.neurons = [];
    for (let i = 0; i < nout; i++) this.neurons.push(new Neuron(nin, opts));
    this.nin = nin; this.nout = nout;
  }
  forward(x) { return this.neurons.map(n => n.forward(x)); }
  parameters() {
    const p = [];
    for (const n of this.neurons) p.push.apply(p, n.parameters());
    return p;
  }
}

class MLP {
  /* new MLP(sizes, {act='tanh', outAct='identity', rng, init, initScale})
     sizes: [nin, h1, h2, ..., nout]. `act` applies to hidden layers, `outAct`
     to the final layer. Default outAct is identity, so the output is LOGITS —
     pair it with bceWithLogits / crossEntropyLoss. */
  constructor(sizes, opts) {
    opts = opts || {};
    if (!Array.isArray(sizes) || sizes.length < 2) {
      throw new Error("MLP(sizes): need at least [nin, nout], got " + JSON.stringify(sizes));
    }
    const rng = opts.rng || new RNG(1);
    const act = opts.act || "tanh";
    const outAct = opts.outAct || "identity";
    this.sizes = sizes.slice();
    this.layers = [];
    for (let i = 0; i < sizes.length - 1; i++) {
      const isLast = i === sizes.length - 2;
      this.layers.push(new Layer(sizes[i], sizes[i + 1], {
        act: isLast ? outAct : act,
        rng: rng,
        init: opts.init,
        initScale: opts.initScale
      }));
    }
  }
  /* forward(x: number[]|Value[]) -> Value[] (one per output unit) */
  forward(x) {
    let h = x;
    for (const l of this.layers) h = l.forward(h);
    return h;
  }
  /* forward1(x) -> Value — convenience for single-output nets */
  forward1(x) {
    const out = this.forward(x);
    if (out.length !== 1) throw new Error("MLP.forward1: network has " + out.length + " outputs, not 1");
    return out[0];
  }
  parameters() {
    const p = [];
    for (const l of this.layers) p.push.apply(p, l.parameters());
    return p;
  }
  /* THE zero_grad. Not called for you — same as PyTorch, and for the same
     teaching reason: the course wants a demo where you can leave it out. */
  zeroGrad() { for (const p of this.parameters()) p.grad = 0; return this; }
  /* Flat read/write of every parameter, in parameters() order. Lets a demo
     (and the test suite) plant the worked example's exact nine numbers. */
  getParameters() { return this.parameters().map(p => p.data); }
  setParameters(arr) {
    const ps = this.parameters();
    if (arr.length !== ps.length) {
      throw new Error("MLP.setParameters: network has " + ps.length + " parameters, got " + arr.length);
    }
    for (let i = 0; i < ps.length; i++) ps[i].data = arr[i];
    return this;
  }
  numParameters() { return this.parameters().length; }
}

/* ==========================================================================
   5. OPTIMIZERS
   ========================================================================== */

class SGD {
  /* new SGD(params, {lr=0.1, momentum=0, weightDecay=0, nesterov=false}) */
  constructor(params, opts) {
    opts = opts || {};
    this.params = params;
    this.lr = opts.lr === undefined ? 0.1 : opts.lr;
    this.momentum = opts.momentum || 0;
    this.weightDecay = opts.weightDecay || 0;
    this.nesterov = !!opts.nesterov;
    this.buf = new Float64Array(params.length); // velocity
    this.t = 0;
  }
  zeroGrad() { for (const p of this.params) p.grad = 0; }
  step() {
    this.t++;
    for (let i = 0; i < this.params.length; i++) {
      const p = this.params[i];
      let g = p.grad;
      if (this.weightDecay) g += this.weightDecay * p.data; // L2, coupled (classic SGD)
      if (this.momentum) {
        // PyTorch's formulation: buf = mu*buf + g  (dampening=0), then step by buf.
        this.buf[i] = this.momentum * this.buf[i] + g;
        g = this.nesterov ? g + this.momentum * this.buf[i] : this.buf[i];
      }
      p.data -= this.lr * g;
    }
  }
}

class Adam {
  /* new Adam(params, {lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
                       weightDecay=0, decoupled=false})
     decoupled:true == AdamW (decay applied straight to the weights, NOT folded
     into the gradient, so Adam's per-parameter scaling never touches it — that
     distinction is the entire content of the AdamW paper). */
  constructor(params, opts) {
    opts = opts || {};
    this.params = params;
    this.lr = opts.lr === undefined ? 1e-3 : opts.lr;
    this.beta1 = opts.beta1 === undefined ? 0.9 : opts.beta1;
    this.beta2 = opts.beta2 === undefined ? 0.999 : opts.beta2;
    this.eps = opts.eps === undefined ? 1e-8 : opts.eps;
    this.weightDecay = opts.weightDecay || 0;
    this.decoupled = !!opts.decoupled;
    this.m = new Float64Array(params.length);
    this.v = new Float64Array(params.length);
    this.t = 0;
  }
  zeroGrad() { for (const p of this.params) p.grad = 0; }
  step() {
    this.t++;
    const b1 = this.beta1, b2 = this.beta2;
    // Bias correction. m starts at 0, so m_1 = (1-b1)*g_1 is 10x too SMALL at
    // b1=0.9 — an artifact of initializing the EMA at zero, not a property of
    // the gradient. The 1/(1-b1^t) factor removes exactly that artifact and
    // fades to 1. Without it Adam's first step is enormous (at b2=0.999, v is
    // 1000x too small at t=1, so sqrt(v) needs a 31.6x correction).
    const c1 = 1 - Math.pow(b1, this.t);
    const c2 = 1 - Math.pow(b2, this.t);
    for (let i = 0; i < this.params.length; i++) {
      const p = this.params[i];
      let g = p.grad;
      if (this.weightDecay && !this.decoupled) g += this.weightDecay * p.data;
      this.m[i] = b1 * this.m[i] + (1 - b1) * g;
      this.v[i] = b2 * this.v[i] + (1 - b2) * g * g;
      const mh = this.m[i] / c1;
      const vh = this.v[i] / c2;
      if (this.weightDecay && this.decoupled) p.data -= this.lr * this.weightDecay * p.data;
      p.data -= this.lr * mh / (Math.sqrt(vh) + this.eps);
    }
  }
}

/* ==========================================================================
   6. TRAINER — a stepper, not a blocking loop
   A train(1000) that runs to completion inside a click handler freezes the tab
   and the learner sees nothing happen, then everything happen. The demo must
   drive this from requestAnimationFrame, a few steps per frame, so the loss
   curve draws itself and the page stays alive.
   ========================================================================== */

class Trainer {
  /* new Trainer({model, X, y, loss='bce'|'mse'|'ce', opt, batchSize=0, rng})
     batchSize 0 or >= N means full batch. `opt` is an SGD/Adam instance built
     over model.parameters(). */
  constructor(cfg) {
    this.model = cfg.model;
    this.X = cfg.X;
    this.y = cfg.y;
    this.lossKind = cfg.loss || "bce";
    this.opt = cfg.opt || new SGD(this.model.parameters(), { lr: 0.1 });
    this.batchSize = cfg.batchSize || 0;
    this.rng = cfg.rng || new RNG(7);
    this.skipZeroGrad = !!cfg.skipZeroGrad; // for the "forgot zero_grad()" demo
    this.step_ = 0;
    this.history = [];
    if (this.X.length !== this.y.length) {
      throw new Error("Trainer: X has " + this.X.length + " rows but y has " + this.y.length);
    }
  }
  _batch() {
    const n = this.X.length;
    if (!this.batchSize || this.batchSize >= n) {
      const idx = []; for (let i = 0; i < n; i++) idx.push(i);
      return idx;
    }
    const idx = [];
    for (let i = 0; i < this.batchSize; i++) idx.push(this.rng.int(n));
    return idx;
  }
  /* lossOn(indices) -> Value (mean over the batch) */
  lossOn(idx) {
    const terms = [];
    for (const i of idx) {
      const out = this.model.forward(this.X[i]);
      if (this.lossKind === "mse") terms.push(mseLoss(out, Array.isArray(this.y[i]) ? this.y[i] : [this.y[i]]));
      else if (this.lossKind === "ce") terms.push(crossEntropyLoss(out, this.y[i]));
      else if (this.lossKind === "bce") terms.push(bceWithLogits(out, Array.isArray(this.y[i]) ? this.y[i] : [this.y[i]]));
      else throw new Error("Trainer: unknown loss '" + this.lossKind + "' (use 'mse', 'bce' or 'ce')");
    }
    return vsum(terms).div(terms.length);
  }
  /* step() -> loss (number). One optimizer step. */
  step() {
    const L = this.lossOn(this._batch());
    if (!this.skipZeroGrad) this.opt.zeroGrad();
    L.backward();
    this.opt.step();
    this.step_++;
    this.history.push(L.data);
    return L.data;
  }
  /* steps(n) -> last loss. Call this from rAF with n ~ 1-20. */
  steps(n) {
    let L = NaN;
    for (let i = 0; i < n; i++) L = this.step();
    return L;
  }
  /* Full-dataset loss without touching gradients — for a val curve. */
  evalLoss(X, y) {
    const Xs = X || this.X, ys = y || this.y;
    let s = 0;
    for (let i = 0; i < Xs.length; i++) {
      const out = this.model.forward(Xs[i]);
      if (this.lossKind === "mse") s += mseLoss(out, Array.isArray(ys[i]) ? ys[i] : [ys[i]]).data;
      else if (this.lossKind === "ce") s += crossEntropyLoss(out, ys[i]).data;
      else s += bceWithLogits(out, Array.isArray(ys[i]) ? ys[i] : [ys[i]]).data;
    }
    return s / Xs.length;
  }
  /* Binary accuracy, threshold on the logit at 0 (== probability 0.5). */
  accuracy(X, y) {
    const Xs = X || this.X, ys = y || this.y;
    let c = 0;
    for (let i = 0; i < Xs.length; i++) {
      const z = this.model.forward(Xs[i])[0].data;
      const pred = z > 0 ? 1 : 0;
      const t = Array.isArray(ys[i]) ? ys[i][0] : ys[i];
      if (pred === t) c++;
    }
    return c / Xs.length;
  }
}

/* ==========================================================================
   7. Mat — minimal dense matrix layer
   Flat Float32Array + explicit [rows, cols]. Shape errors throw messages that
   name BOTH shapes, because "shapes (3,4) and (5,4) don't line up: matmul needs
   A's columns (4) to equal B's rows (5)" is a lesson and "RuntimeError: size
   mismatch" is not. Shape bookkeeping is the #1 practical skill the course
   teaches; the error text is teaching surface.
   ========================================================================== */

class Mat {
  /* new Mat(rows, cols, data?) — data is a flat, row-major array of length r*c */
  constructor(rows, cols, data) {
    if (!(rows > 0 && cols > 0 && Number.isInteger(rows) && Number.isInteger(cols))) {
      throw new Error("new Mat(" + rows + ", " + cols + "): dimensions must be positive integers");
    }
    this.rows = rows; this.cols = cols;
    if (data) {
      if (data.length !== rows * cols) {
        throw new Error("new Mat(" + rows + ", " + cols + "): expected " + (rows * cols) +
          " values for that shape, got " + data.length);
      }
      this.data = data instanceof Float32Array ? data : Float32Array.from(data);
    } else {
      this.data = new Float32Array(rows * cols);
    }
  }
  get shape() { return [this.rows, this.cols]; }
  get shapeStr() { return "(" + this.rows + "," + this.cols + ")"; }

  static zeros(r, c) { return new Mat(r, c); }
  static ones(r, c) { const m = new Mat(r, c); m.data.fill(1); return m; }
  static full(r, c, v) { const m = new Mat(r, c); m.data.fill(v); return m; }
  static eye(n) { const m = new Mat(n, n); for (let i = 0; i < n; i++) m.data[i * n + i] = 1; return m; }
  /* from([[1,2],[3,4]]) — the shape is inferred and rows are checked for raggedness */
  static from(arr2d) {
    if (!Array.isArray(arr2d) || !arr2d.length) throw new Error("Mat.from: expected a non-empty 2-D array");
    if (!Array.isArray(arr2d[0])) return new Mat(1, arr2d.length, arr2d); // a bare row
    const r = arr2d.length, c = arr2d[0].length;
    const d = new Float32Array(r * c);
    for (let i = 0; i < r; i++) {
      if (arr2d[i].length !== c) {
        throw new Error("Mat.from: ragged input — row 0 has " + c + " values but row " + i +
          " has " + arr2d[i].length + ". Every row must be the same length.");
      }
      for (let j = 0; j < c; j++) d[i * c + j] = arr2d[i][j];
    }
    return new Mat(r, c, d);
  }
  /* randn(r, c, rng, scale=1) — seeded, so demos reproduce */
  static randn(r, c, rng, scale) {
    scale = scale === undefined ? 1 : scale;
    const m = new Mat(r, c);
    for (let i = 0; i < m.data.length; i++) m.data[i] = rng.normal(0, 1) * scale;
    return m;
  }

  get(i, j) {
    if (i < 0 || i >= this.rows || j < 0 || j >= this.cols) {
      throw new RangeError("Mat.get(" + i + "," + j + "): out of bounds for shape " + this.shapeStr);
    }
    return this.data[i * this.cols + j];
  }
  set(i, j, v) {
    if (i < 0 || i >= this.rows || j < 0 || j >= this.cols) {
      throw new RangeError("Mat.set(" + i + "," + j + "): out of bounds for shape " + this.shapeStr);
    }
    this.data[i * this.cols + j] = v; return this;
  }
  row(i) { return Array.from(this.data.subarray(i * this.cols, (i + 1) * this.cols)); }
  col(j) { const o = []; for (let i = 0; i < this.rows; i++) o.push(this.data[i * this.cols + j]); return o; }
  clone() { return new Mat(this.rows, this.cols, this.data.slice()); }
  toArray() { const o = []; for (let i = 0; i < this.rows; i++) o.push(this.row(i)); return o; }

  /* reshape(r, c) — same buffer, new view of it. Total size must match. */
  reshape(r, c) {
    if (r * c !== this.data.length) {
      throw new Error("Mat.reshape" + this.shapeStr + " -> (" + r + "," + c + "): cannot reshape " +
        this.data.length + " values into " + (r * c) + ". The total number of elements must not change.");
    }
    return new Mat(r, c, this.data.slice());
  }

  T() {
    const o = new Mat(this.cols, this.rows);
    for (let i = 0; i < this.rows; i++)
      for (let j = 0; j < this.cols; j++)
        o.data[j * this.rows + i] = this.data[i * this.cols + j];
    return o;
  }

  /* matmul(B) -> (this.rows x B.cols). The inner dimensions must agree. */
  matmul(B) {
    if (!(B instanceof Mat)) throw new TypeError("Mat.matmul: expected a Mat, got " + typeof B);
    if (this.cols !== B.rows) {
      throw new Error("Mat.matmul: shapes " + this.shapeStr + " and " + B.shapeStr +
        " do not line up. Matmul needs the LEFT matrix's columns (" + this.cols +
        ") to equal the RIGHT matrix's rows (" + B.rows + "). " +
        "The result would be (" + this.rows + "," + B.cols + "). " +
        (this.cols === B.cols ? "Both have " + this.cols + " columns — did you mean B.T()?" : ""));
    }
    const o = new Mat(this.rows, B.cols);
    const n = this.cols;
    for (let i = 0; i < this.rows; i++) {
      for (let k = 0; k < n; k++) {
        const a = this.data[i * n + k];
        if (a === 0) continue;
        const boff = k * B.cols, ooff = i * B.cols;
        for (let j = 0; j < B.cols; j++) o.data[ooff + j] += a * B.data[boff + j];
      }
    }
    return o;
  }

  /* Elementwise binary op with sane broadcasting:
     same shape | (1,c) row vector | (r,1) column vector | scalar number. */
  _ew(other, fn, name) {
    if (typeof other === "number") {
      const o = new Mat(this.rows, this.cols);
      for (let i = 0; i < this.data.length; i++) o.data[i] = fn(this.data[i], other);
      return o;
    }
    if (!(other instanceof Mat)) throw new TypeError("Mat." + name + ": expected a Mat or a number, got " + typeof other);
    const br = other.rows, bc = other.cols;
    const rowBroadcast = br === 1 && bc === this.cols;
    const colBroadcast = bc === 1 && br === this.rows;
    const same = br === this.rows && bc === this.cols;
    if (!same && !rowBroadcast && !colBroadcast) {
      throw new Error("Mat." + name + ": shapes " + this.shapeStr + " and " + other.shapeStr +
        " are not compatible. Elementwise ops need identical shapes, or the right operand to be " +
        "a row vector (1," + this.cols + "), a column vector (" + this.rows + ",1), or a plain number.");
    }
    const o = new Mat(this.rows, this.cols);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        const b = same ? other.data[i * bc + j] : (rowBroadcast ? other.data[j] : other.data[i]);
        o.data[i * this.cols + j] = fn(this.data[i * this.cols + j], b);
      }
    }
    return o;
  }
  add(o) { return this._ew(o, (a, b) => a + b, "add"); }
  sub(o) { return this._ew(o, (a, b) => a - b, "sub"); }
  mul(o) { return this._ew(o, (a, b) => a * b, "mul"); }   // HADAMARD, not matmul
  div(o) { return this._ew(o, (a, b) => a / b, "div"); }
  scale(k) { return this._ew(k, (a, b) => a * b, "scale"); }
  map(fn) {
    const o = new Mat(this.rows, this.cols);
    for (let i = 0; i < this.data.length; i++) o.data[i] = fn(this.data[i], (i / this.cols) | 0, i % this.cols);
    return o;
  }
  sum() { let s = 0; for (let i = 0; i < this.data.length; i++) s += this.data[i]; return s; }
  max() { let m = -Infinity; for (let i = 0; i < this.data.length; i++) if (this.data[i] > m) m = this.data[i]; return m; }
  min() { let m = Infinity; for (let i = 0; i < this.data.length; i++) if (this.data[i] < m) m = this.data[i]; return m; }
  mean() { return this.sum() / this.data.length; }
  rowSums() { const o = []; for (let i = 0; i < this.rows; i++) { let s = 0; for (let j = 0; j < this.cols; j++) s += this.data[i * this.cols + j]; o.push(s); } return o; }
  rowMaxes() { const o = []; for (let i = 0; i < this.rows; i++) { let m = -Infinity; for (let j = 0; j < this.cols; j++) { const v = this.data[i * this.cols + j]; if (v > m) m = v; } o.push(m); } return o; }
  toString() { return "Mat" + this.shapeStr + " " + JSON.stringify(this.toArray().map(r => r.map(v => +v.toFixed(4)))); }
}

/* softmax(arr, {stable=true}) — plain-number softmax over a 1-D array.
   stable:false is provided ON PURPOSE so a demo can show it produce NaN on
   [1000,1001,1002]. Do not "fix" it. */
function softmax(arr, opts) {
  const stable = !opts || opts.stable !== false;
  const m = stable ? Math.max.apply(null, arr) : 0;
  const e = arr.map(v => Math.exp(v - m));
  let s = 0; for (const x of e) s += x;
  return e.map(x => x / s); // s = 0 or Infinity -> NaN, and that is the lesson
}

/* softmaxRows(M, {stable=true, temperature=1}) -> Mat, each row sums to 1. */
function softmaxRows(M, opts) {
  opts = opts || {};
  const stable = opts.stable !== false;
  const tau = opts.temperature === undefined ? 1 : opts.temperature;
  const o = new Mat(M.rows, M.cols);
  for (let i = 0; i < M.rows; i++) {
    const off = i * M.cols;
    let mx = 0;
    if (stable) {
      mx = -Infinity;
      for (let j = 0; j < M.cols; j++) { const v = M.data[off + j] / tau; if (v > mx) mx = v; }
      // A fully-masked row is all -Infinity; exp(-Inf - -Inf) = NaN. Guard it.
      if (!isFinite(mx)) mx = 0;
    }
    let s = 0;
    for (let j = 0; j < M.cols; j++) { const e = Math.exp(M.data[off + j] / tau - mx); o.data[off + j] = e; s += e; }
    for (let j = 0; j < M.cols; j++) o.data[off + j] /= s;
  }
  return o;
}

/* layerNorm(M, {eps=1e-5, gamma, beta}) — normalizes each ROW (the feature
   dimension), which is what LayerNorm means in a transformer: statistics are
   per-token, never across the batch. gamma/beta are length-cols arrays. */
function layerNorm(M, opts) {
  opts = opts || {};
  const eps = opts.eps === undefined ? 1e-5 : opts.eps;
  const gamma = opts.gamma, beta = opts.beta;
  if (gamma && gamma.length !== M.cols) {
    throw new Error("layerNorm: gamma has " + gamma.length + " values but the feature dimension of " +
      M.shapeStr + " is " + M.cols + ". LayerNorm normalizes each row, so gamma must be length " + M.cols + ".");
  }
  if (beta && beta.length !== M.cols) {
    throw new Error("layerNorm: beta has " + beta.length + " values but the feature dimension of " +
      M.shapeStr + " is " + M.cols + ".");
  }
  const o = new Mat(M.rows, M.cols);
  for (let i = 0; i < M.rows; i++) {
    const off = i * M.cols;
    let mu = 0;
    for (let j = 0; j < M.cols; j++) mu += M.data[off + j];
    mu /= M.cols;
    let va = 0;
    for (let j = 0; j < M.cols; j++) { const d = M.data[off + j] - mu; va += d * d; }
    va /= M.cols; // biased (1/N), matching PyTorch's LayerNorm — not 1/(N-1)
    const inv = 1 / Math.sqrt(va + eps);
    for (let j = 0; j < M.cols; j++) {
      let v = (M.data[off + j] - mu) * inv;
      if (gamma) v *= gamma[j];
      if (beta) v += beta[j];
      o.data[off + j] = v;
    }
  }
  return o;
}

/* rmsNorm(M, {eps, gamma}) — LayerNorm minus the mean-centering. What Llama
   and most modern LLMs actually use. */
function rmsNorm(M, opts) {
  opts = opts || {};
  const eps = opts.eps === undefined ? 1e-6 : opts.eps;
  const gamma = opts.gamma;
  if (gamma && gamma.length !== M.cols) {
    throw new Error("rmsNorm: gamma has " + gamma.length + " values but the feature dimension of " +
      M.shapeStr + " is " + M.cols + ".");
  }
  const o = new Mat(M.rows, M.cols);
  for (let i = 0; i < M.rows; i++) {
    const off = i * M.cols;
    let ss = 0;
    for (let j = 0; j < M.cols; j++) ss += M.data[off + j] * M.data[off + j];
    const inv = 1 / Math.sqrt(ss / M.cols + eps);
    for (let j = 0; j < M.cols; j++) o.data[off + j] = M.data[off + j] * inv * (gamma ? gamma[j] : 1);
  }
  return o;
}

/* ==========================================================================
   8. ATTENTION — softmax(QK^T / sqrt(dk) + M) V
   Returns the attention weights, not just the output: the heatmap IS the demo.
   ========================================================================== */

/* attention(Q, K, V, {causal=false, temperature, scale}) ->
     {O, A, scores, entropy, effKeys, gradHealth, maxJac, dk}
   Q:(n,dk) K:(m,dk) V:(m,dv). temperature overrides the default sqrt(dk)
   scaling so a demo can drag it off the detent and watch attention snap to a
   hard argmax (entropy -> 0, gradient health -> ~0) or melt to uniform. */
function attention(Q, K, V, opts) {
  opts = opts || {};
  if (Q.cols !== K.cols) {
    throw new Error("attention: Q is " + Q.shapeStr + " and K is " + K.shapeStr +
      ". Q and K must share the head dimension d_k (Q has " + Q.cols + ", K has " + K.cols +
      ") because every score is a dot product of a query row with a key row.");
  }
  if (K.rows !== V.rows) {
    throw new Error("attention: K is " + K.shapeStr + " and V is " + V.shapeStr +
      ". K and V must have the same number of rows (one key and one value per source position; " +
      "K has " + K.rows + ", V has " + V.rows + ").");
  }
  const dk = Q.cols;
  const tau = opts.temperature !== undefined ? opts.temperature : Math.sqrt(dk);
  if (!(tau > 0)) throw new Error("attention: temperature must be > 0, got " + tau);

  const S = Q.matmul(K.T());          // (n,m) raw scores
  const Sc = S.scale(1 / tau);        // scaled scores
  if (opts.causal) {
    // Mask the future with -Infinity BEFORE the softmax, so exp() sends it to
    // exactly 0. Masking after the softmax would leak: the denominator would
    // already contain the future's contribution.
    for (let i = 0; i < Sc.rows; i++)
      for (let j = 0; j < Sc.cols; j++)
        if (j > i) Sc.data[i * Sc.cols + j] = -Infinity;
  }
  if (opts.mask) {
    const M = opts.mask;
    if (M.rows !== Sc.rows || M.cols !== Sc.cols) {
      throw new Error("attention: mask is " + M.shapeStr + " but the score matrix is " +
        Sc.shapeStr + ". The mask must match the scores exactly (one entry per query-key pair).");
    }
    for (let i = 0; i < Sc.data.length; i++) if (M.data[i] === 0) Sc.data[i] = -Infinity;
  }
  const A = softmaxRows(Sc, { stable: true }); // rows sum to 1
  const O = A.matmul(V);                        // (n,dv)

  // Readouts. These are what make it a lesson instead of a picture.
  const entropy = [], effKeys = [];
  let maxJac = 0;
  for (let i = 0; i < A.rows; i++) {
    let H = 0;
    for (let j = 0; j < A.cols; j++) {
      const p = A.data[i * A.cols + j];
      if (p > 0) H -= p * Math.log(p);       // 0*log0 := 0
      const jac = p * (1 - p);               // softmax diagonal Jacobian
      if (jac > maxJac) maxJac = jac;
    }
    entropy.push(H);
    effKeys.push(Math.exp(H)); // perplexity of the attention row = "how many keys am I really using"
  }
  return {
    O: O, A: A, scores: S, scaled: Sc, dk: dk, tau: tau,
    entropy: entropy, effKeys: effKeys,
    maxEntropy: Math.log(A.cols),
    gradHealth: maxJac, maxJac: maxJac
  };
}

/* ==========================================================================
   9. DATASETS — deterministic given an RNG
   ========================================================================== */

/* makeDataset(kind, n, {rng, noise, labelNoise})
   kind: 'moons' | 'circles' | 'xor' | 'spiral' | 'blobs'
   -> {X: [[x1,x2],...], y: [0|1,...], kind, n}
   All are scaled to roughly [-2,2] so one plot window fits every dataset. */
function makeDataset(kind, n, opts) {
  opts = opts || {};
  const rng = opts.rng || new RNG(42);
  const noise = opts.noise === undefined ? 0.15 : opts.noise;
  n = n || 100;
  const X = [], y = [];
  if (kind === "moons") {
    const half = Math.floor(n / 2);
    for (let i = 0; i < n; i++) {
      const c = i < half ? 0 : 1;
      const t = Math.PI * (i < half ? i / half : (i - half) / (n - half));
      let px, py;
      if (c === 0) { px = Math.cos(t); py = Math.sin(t); }
      else { px = 1 - Math.cos(t); py = 0.5 - Math.sin(t); }
      X.push([(px - 0.5) * 1.6 + rng.normal(0, noise), (py - 0.25) * 1.6 + rng.normal(0, noise)]);
      y.push(c);
    }
  } else if (kind === "circles") {
    for (let i = 0; i < n; i++) {
      const c = i % 2;
      const r = (c === 0 ? 0.5 : 1.4) + rng.normal(0, noise);
      const t = rng.uniform(0, 2 * Math.PI);
      X.push([r * Math.cos(t), r * Math.sin(t)]);
      y.push(c);
    }
  } else if (kind === "xor") {
    for (let i = 0; i < n; i++) {
      const a = rng.uniform(-1.5, 1.5), b = rng.uniform(-1.5, 1.5);
      X.push([a + rng.normal(0, noise * 0.3), b + rng.normal(0, noise * 0.3)]);
      y.push((a > 0) !== (b > 0) ? 1 : 0);
    }
  } else if (kind === "spiral") {
    const half = Math.floor(n / 2);
    for (let i = 0; i < n; i++) {
      const c = i < half ? 0 : 1;
      const k = i < half ? i / half : (i - half) / (n - half);
      const r = 0.2 + 1.6 * k;
      const t = 2.6 * Math.PI * k + (c === 1 ? Math.PI : 0);
      X.push([r * Math.cos(t) + rng.normal(0, noise * 0.6), r * Math.sin(t) + rng.normal(0, noise * 0.6)]);
      y.push(c);
    }
  } else if (kind === "blobs") {
    for (let i = 0; i < n; i++) {
      const c = i % 2;
      const cx = c === 0 ? -1 : 1, cy = c === 0 ? -0.6 : 0.6;
      X.push([cx + rng.normal(0, noise * 2), cy + rng.normal(0, noise * 2)]);
      y.push(c);
    }
  } else {
    throw new Error("makeDataset: unknown kind '" + kind + "'. Known: moons, circles, xor, spiral, blobs");
  }
  // Label noise: flip a fraction of labels. This is what makes overfitting
  // visible — a model with enough capacity will memorize the flipped points,
  // and the val curve will turn up while the train curve keeps falling.
  if (opts.labelNoise) {
    for (let i = 0; i < y.length; i++) if (rng.random() < opts.labelNoise) y[i] = 1 - y[i];
  }
  return { X: X, y: y, kind: kind, n: X.length };
}

/* XOR's four corners exactly — no noise, no sampling. The canonical test case. */
function xorData() {
  return {
    X: [[0, 0], [0, 1], [1, 0], [1, 1]],
    y: [0, 1, 1, 0],
    kind: "xor-exact", n: 4
  };
}

/* trainTestSplit(ds, frac=0.7, rng) -> {train:{X,y}, test:{X,y}} */
function trainTestSplit(ds, frac, rng) {
  frac = frac === undefined ? 0.7 : frac;
  rng = rng || new RNG(5);
  const idx = []; for (let i = 0; i < ds.X.length; i++) idx.push(i);
  rng.shuffle(idx);
  const k = Math.floor(idx.length * frac);
  const pick = ids => ({ X: ids.map(i => ds.X[i]), y: ids.map(i => ds.y[i]) });
  return { train: pick(idx.slice(0, k)), test: pick(idx.slice(k)) };
}

/* ==========================================================================
   10. GRADIENT CHECKING — central differences
   This is both a demo (the course verifies autograd on screen) and this file's
   own conscience: nn.test.html runs it over every op.
   ========================================================================== */

/* gradCheck(buildLoss, params, {eps=1e-6, tol=1e-6})
     buildLoss: () => Value   — must REBUILD the graph from params each call,
                                because perturbing params[k].data has no effect
                                on an already-built graph's stored intermediates.
     params:    Value[]       — the leaves to check.
   -> {ok, maxRelErr, maxAbsErr, results:[{i,label,analytic,numeric,absErr,relErr,ok}]}

   Central, not forward, differences: error is O(eps^2) rather than O(eps), which
   is the difference between agreeing to 8 decimals and agreeing to 4. Note that
   .data is a plain JS number (float64) — in float32, eps=1e-6 sits right on top
   of machine epsilon (~1.19e-7) and L(x+eps) - L(x-eps) catastrophically
   cancels into noise. That is why torch.autograd.gradcheck defaults to float64,
   and why this engine keeps parameters in float64 even though Mat is float32. */
function gradCheck(buildLoss, params, opts) {
  opts = opts || {};
  const eps = opts.eps === undefined ? 1e-6 : opts.eps;
  const tol = opts.tol === undefined ? 1e-6 : opts.tol;

  for (const p of params) p.grad = 0;
  const L = buildLoss();
  L.backward();
  const analytic = params.map(p => p.grad);

  const results = [];
  let maxRelErr = 0, maxAbsErr = 0;
  for (let i = 0; i < params.length; i++) {
    const p = params[i];
    const orig = p.data;
    p.data = orig + eps; const Lp = buildLoss().data;
    p.data = orig - eps; const Lm = buildLoss().data;
    p.data = orig; // restore EXACTLY — orig is kept, not recomputed by +eps-eps
    const numeric = (Lp - Lm) / (2 * eps);
    const absErr = Math.abs(numeric - analytic[i]);
    // Relative error normalized by the larger magnitude, floored at 1 so that a
    // gradient of ~0 is judged on absolute error and doesn't divide by ~0.
    const denom = Math.max(1, Math.abs(numeric), Math.abs(analytic[i]));
    const relErr = absErr / denom;
    if (relErr > maxRelErr) maxRelErr = relErr;
    if (absErr > maxAbsErr) maxAbsErr = absErr;
    results.push({
      i: i, label: p.label || ("p[" + i + "]"),
      analytic: analytic[i], numeric: numeric,
      absErr: absErr, relErr: relErr, ok: relErr <= tol
    });
  }
  return { ok: maxRelErr <= tol, maxRelErr: maxRelErr, maxAbsErr: maxAbsErr, results: results, loss: L.data };
}

/* ==========================================================================
   11. The 2-2-1 worked example (brief-training.md §5.4)
   Hard-coded as a first-class object because a dozen pages refer back to it and
   they must all show the SAME nine numbers. Built on the same Value engine
   everything else uses — not a special case, just a small case.
   ========================================================================== */

/* worked221({W1,b1,W2,b2,x,y}) -> {params, loss, forward:{z1,a1,z2,yhat}, grads, net}
   Defaults are §5.4's exact constants. Every field is a live Value. */
function worked221(cfg) {
  cfg = cfg || {};
  const W1 = cfg.W1 || [[0.5, -0.3], [0.2, 0.8]];
  const b1 = cfg.b1 || [0.1, -0.1];
  const W2 = cfg.W2 || [0.7, -0.4];
  const b2 = cfg.b2 === undefined ? 0.2 : cfg.b2;
  const x = cfg.x || [0.6, -0.2];
  const y = cfg.y === undefined ? 1 : cfg.y;

  const P = {
    W1_11: V(W1[0][0], "W1_11"), W1_12: V(W1[0][1], "W1_12"),
    W1_21: V(W1[1][0], "W1_21"), W1_22: V(W1[1][1], "W1_22"),
    b1_1: V(b1[0], "b1_1"), b1_2: V(b1[1], "b1_2"),
    W2_1: V(W2[0], "W2_1"), W2_2: V(W2[1], "W2_2"),
    b2: V(b2, "b2")
  };
  const order = ["W1_11", "W1_12", "W1_21", "W1_22", "b1_1", "b1_2", "W2_1", "W2_2", "b2"];
  const params = order.map(k => P[k]);

  function build() {
    const z1_1 = P.W1_11.mul(x[0]).add(P.W1_12.mul(x[1])).add(P.b1_1);
    const z1_2 = P.W1_21.mul(x[0]).add(P.W1_22.mul(x[1])).add(P.b1_2);
    const a1_1 = z1_1.tanh(), a1_2 = z1_2.tanh();
    const z2 = P.W2_1.mul(a1_1).add(P.W2_2.mul(a1_2)).add(P.b2);
    const yhat = z2.sigmoid();
    // BCE written out in full. Stable enough here (yhat is nowhere near 0 or 1)
    // and it keeps the graph matching the equations on the page, term for term.
    const L = yhat.log().mul(y).add(yhat.neg().add(1).log().mul(1 - y)).neg();
    return { z1_1, z1_2, a1_1, a1_2, z2, yhat, L };
  }

  for (const p of params) p.grad = 0;
  const f = build();
  f.L.backward();

  const grads = {};
  order.forEach(k => { grads[k] = P[k].grad; });

  return {
    P: P, params: params, order: order,
    x: x, y: y,
    forward: { z1: [f.z1_1.data, f.z1_2.data], a1: [f.a1_1.data, f.a1_2.data], z2: f.z2.data, yhat: f.yhat.data },
    nodes: f,
    loss: f.L.data,
    grads: grads,
    gradArray: order.map(k => grads[k]),
    /* rebuild() -> the loss Value, fresh — what gradCheck's buildLoss wants. */
    rebuild: () => build().L,
    /* sgdStep(eta) -> the new loss, after updating in place. */
    sgdStep: function (eta) {
      eta = eta === undefined ? 0.5 : eta;
      for (const p of params) p.data -= eta * p.grad;
      for (const p of params) p.grad = 0;
      const f2 = build();
      f2.L.backward();
      order.forEach(k => { grads[k] = P[k].grad; });
      this.loss = f2.L.data;
      this.forward = { z1: [f2.z1_1.data, f2.z1_2.data], a1: [f2.a1_1.data, f2.a1_2.data], z2: f2.z2.data, yhat: f2.yhat.data };
      this.nodes = f2;
      this.gradArray = order.map(k => grads[k]);
      return this.loss;
    }
  };
}

/* ==========================================================================
   12. Small numeric helpers the demos reuse
   ========================================================================== */

function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }
function lerp(a, b, t) { return a + (b - a) * t; }
function argmax(arr) { let bi = 0, bv = -Infinity; for (let i = 0; i < arr.length; i++) if (arr[i] > bv) { bv = arr[i]; bi = i; } return bi; }
function mean(arr) { let s = 0; for (const v of arr) s += v; return s / arr.length; }
function std(arr) { const m = mean(arr); let s = 0; for (const v of arr) s += (v - m) * (v - m); return Math.sqrt(s / arr.length); }
/* entropy of a probability vector, in nats */
function entropy(p) { let H = 0; for (const q of p) if (q > 0) H -= q * Math.log(q); return H; }

/* ==========================================================================
   EXPORTS — plain globals, no modules (the course is opened over file://,
   where ES module imports are blocked by CORS).
   ========================================================================== */
window.NN = {
  // rng
  mulberry32, RNG,
  // autograd
  Value, V, vsum,
  // losses
  mseLoss, bceLoss, bceWithLogits, crossEntropyLoss, logSumExp, softmaxValues,
  // nets
  Neuron, Layer, MLP, ACTIVATIONS,
  // optim
  SGD, Adam, Trainer,
  // tensors
  Mat, softmax, softmaxRows, layerNorm, rmsNorm, attention,
  // data
  makeDataset, xorData, trainTestSplit,
  // verification
  gradCheck, worked221,
  // misc
  clamp, lerp, argmax, mean, std, entropy
};

/* Deliberately NOT also assigning window.Value / window.Mat / window.RNG:
   a page that then writes `const {Value} = NN;` would shadow them confusingly,
   and the four-character saving is not worth the ambiguity. Demos write
   `const {Value, MLP} = NN;` inside their own IIFE and everything is explicit
   about where it came from. */

})(window);
