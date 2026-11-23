// Z-map stock sweep on the GPU (crates/cam/src/simulate.rs, `StockMap::cut` and
// `StockMap::rapid_hits`). The stock heights are kept as order-preserving u32 keys so every move
// can lower its cells with atomicMin (lowering is a minimum, so the moves' order does not change
// the heights). A rapid is checked in its own dispatch, after the cuts before it. Coordinates are
// relative to the grid's origin.

struct Grid {
    nx: u32,
    ny: u32,
    cell: f32,
    r: f32,
    rc: f32,
    flat: f32,
    tol: f32,
    _p: u32,
};

struct Step {
    first: u32,      // cut: first segment; rapid: rapid index
    count: u32,      // cut: segments
    i0: u32,         // rapid: window
    j0: u32,
    wi: u32,
    wj: u32,
    _p0: u32,
    _p1: u32,
    a: vec4<f32>,    // rapid: from
    b: vec4<f32>,    // rapid: to
};

@group(0) @binding(0) var<uniform> G: Grid;
@group(0) @binding(1) var<uniform> S: Step;
@group(0) @binding(2) var<storage, read> segs: array<vec4<f32>>;   // a, b per segment
@group(0) @binding(3) var<storage, read_write> stock: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> start: array<f32>;        // heights before the sweep
@group(0) @binding(5) var<storage, read> lows: array<f32>;         // each column's floor
@group(0) @binding(6) var<storage, read_write> hits: array<atomic<u32>>;

fn key(f: f32) -> u32 {
    let b = bitcast<u32>(f);
    if ((b & 0x80000000u) != 0u) {
        return ~b;
    }
    return b | 0x80000000u;
}

fn unkey(u: u32) -> f32 {
    if ((u & 0x80000000u) != 0u) {
        return bitcast<f32>(u & 0x7fffffffu);
    }
    return bitcast<f32>(~u);
}

fn profile(rho: f32) -> f32 {
    let x = clamp(rho, 0.0, G.r);
    if (x <= G.flat || G.rc <= 0.0) {
        return 0.0;
    }
    let d = x - G.flat;
    return G.rc - sqrt(max(G.rc * G.rc - d * d, 0.0));
}

// The tip's height plus the profile at p's distance from the axis, the tip a fraction t of the
// way from a to b.
fn height(p: vec2<f32>, a: vec3<f32>, b: vec3<f32>, t: f32) -> f32 {
    return a.z + (b.z - a.z) * t + profile(length(p - (a.xy + (b.xy - a.xy) * t)));
}

// The lowest tool-bottom height over p as the tip goes from a to b; 3.0e38 when never covered.
// As `lowest_bottom` in simulate.rs: the height is convex along the move, lowest at the closest
// point on a level move, at the downhill end of the stretch under a flat end, a little downhill
// of the closest point under a ball, and found by golden-section search under a bull nose.
fn lowest(p: vec2<f32>, a: vec3<f32>, b: vec3<f32>) -> f32 {
    let u = b.xy - a.xy;
    let w = p - a.xy;
    let uu = dot(u, u);
    let dz = b.z - a.z;
    let r = G.r;
    if (uu < 1.0e-12) {
        if (length(w) > r) {
            return 3.0e38;
        }
        return select(a.z, b.z, dz < 0.0) + profile(length(w));
    }
    let half_b = dot(w, u);
    let disc = half_b * half_b - uu * (dot(w, w) - r * r);
    if (disc < 0.0) {
        return 3.0e38;
    }
    let root = sqrt(disc);
    let lo = max((half_b - root) / uu, 0.0);
    let hi = min((half_b + root) / uu, 1.0);
    if (lo > hi) {
        return 3.0e38;
    }
    let closest = clamp(half_b / uu, lo, hi);
    if (abs(dz) < 1.0e-9) {
        return height(p, a, b, closest);
    }
    if (G.rc <= 0.0) {
        return height(p, a, b, select(lo, hi, dz < 0.0));
    }
    if (G.flat <= 0.0) {
        let len = sqrt(uu);
        let slope = dz / len;
        let along = half_b / len;
        let across2 = max(dot(w, w) - along * along, 0.0);
        let shift = slope * sqrt(max(r * r - across2, 0.0) / (1.0 + slope * slope));
        return height(p, a, b, clamp((along - shift) / len, lo, hi));
    }
    var x0 = lo;
    var x1 = hi;
    let g = 0.6180339887;
    var c = x1 - g * (x1 - x0);
    var d = x0 + g * (x1 - x0);
    var fc = height(p, a, b, c);
    var fd = height(p, a, b, d);
    let len = sqrt(uu);
    for (var k = 0; k < 40; k = k + 1) {
        if ((x1 - x0) * len < 1.0e-5) {
            break;
        }
        if (fc < fd) {
            x1 = d;
            d = c;
            fd = fc;
            c = x1 - g * (x1 - x0);
            fc = height(p, a, b, c);
        } else {
            x0 = c;
            c = d;
            fc = fd;
            d = x0 + g * (x1 - x0);
            fd = height(p, a, b, d);
        }
    }
    return min(min(min(height(p, a, b, lo), height(p, a, b, hi)), min(height(p, a, b, closest), fc)), fd);
}

@compute @workgroup_size(64, 1, 1)
fn cut(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let s = S.first + wg.x;
    let a = segs[2u * s].xyz;
    let b = segs[2u * s + 1u].xyz;
    let lo = vec2<f32>(min(a.x, b.x), min(a.y, b.y)) - vec2<f32>(G.r, G.r);
    let hi = vec2<f32>(max(a.x, b.x), max(a.y, b.y)) + vec2<f32>(G.r, G.r);
    let i0 = u32(max(floor(lo.x / G.cell) - 1.0, 0.0));
    let j0 = u32(max(floor(lo.y / G.cell) - 1.0, 0.0));
    let i1f = ceil(hi.x / G.cell) + 1.0;
    let j1f = ceil(hi.y / G.cell) + 1.0;
    if (i1f < 0.0 || j1f < 0.0) {
        return;
    }
    let i1 = min(u32(i1f), G.nx - 1u);
    let j1 = min(u32(j1f), G.ny - 1u);
    if (i0 > i1 || j0 > j1) {
        return;
    }
    let w = i1 - i0 + 1u;
    let n = w * (j1 - j0 + 1u);
    for (var k = lid; k < n; k = k + 64u) {
        let i = i0 + k % w;
        let j = j0 + k / w;
        let z = lowest(vec2<f32>(f32(i), f32(j)) * G.cell, a, b);
        if (z < 1.0e38) {
            atomicMin(&stock[j * G.nx + i], key(z));
        }
    }
}

@compute @workgroup_size(256, 1, 1)
fn rapid(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = S.wi * S.wj;
    let k = id.x;
    if (k >= n) {
        return;
    }
    let i = S.i0 + k % S.wi;
    let j = S.j0 + k / S.wi;
    if (i >= G.nx || j >= G.ny) {
        return;
    }
    let c = j * G.nx + i;
    let z = lowest(vec2<f32>(f32(i), f32(j)) * G.cell, S.a.xyz, S.b.xyz);
    if (z >= 1.0e38) {
        return;
    }
    let low = lows[c];
    let s = unkey(atomicLoad(&stock[c]));
    let s0 = start[c];
    var cur = s0;
    if (s < s0) {
        cur = max(s, low);
    }
    let zb = max(z, low);
    // -inf (outside the stock) fails `cur > low + tol`.
    if (cur > low + G.tol && zb < cur - G.tol) {
        atomicOr(&hits[S.first], 1u);
    }
}
