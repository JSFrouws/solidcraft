// Cutter-location map on the GPU: for every grid cell, the highest tip height at which an end
// mill (flat, ball or bull-nose), coming down the z axis, touches a triangle mesh. The same
// exact contacts as the CPU drop cutter (crates/cam/src/dropcutter.rs): vertices, facets and
// edges. Triangles are binned in XY; each bin's triangles are stored relative to the bin's
// reference grid node (so the f32 offsets stay small) and sorted by their top, highest first,
// so a cell stops reading a bin once nothing left in it can reach above what it has.

struct Params {
    cell: f32,
    floor: f32,       // relative to z_ref
    r: f32,
    rc: f32,
    flat: f32,
    nx: u32,
    ny: u32,
    row0: u32,        // first row of this dispatch
    rows: u32,
    bin_cells: u32,   // a bin is bin_cells x bin_cells grid cells
    shift: u32,       // bin k covers cells [k*bin_cells - shift, (k+1)*bin_cells - shift)
    nbx: u32,
    nby: u32,
    reach: u32,       // cells within the tool radius (rounded up, plus one)
    _p0: u32,
    _p1: u32,
};

struct Tri {
    a: vec4<f32>,
    b: vec4<f32>,
    c: vec4<f32>,
    n: vec4<f32>,   // unit normal, w = the triangle's top (z max)
    bb: vec4<f32>,  // XY bounds: min x, min y, max x, max y
};

struct Bin {
    start: u32,
    end: u32,
    ref_i: i32,     // reference grid node of the bin (triangles are stored relative to it)
    ref_j: i32,
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> tris: array<Tri>;
@group(0) @binding(2) var<storage, read> bins: array<Bin>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

fn profile(rho: f32) -> f32 {
    let x = clamp(rho, 0.0, P.r);
    if (x <= P.flat || P.rc <= 0.0) {
        return 0.0;
    }
    let d = x - P.flat;
    return P.rc - sqrt(max(P.rc * P.rc - d * d, 0.0));
}

fn cross2(u: vec2<f32>, v: vec2<f32>) -> f32 {
    return u.x * v.y - u.y * v.x;
}

fn in_tri(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> bool {
    let d1 = cross2(b - a, p - a);
    let d2 = cross2(c - b, p - b);
    let d3 = cross2(a - c, p - c);
    let eps = -1e-6 * (length(b - a) + length(c - b) + length(a - c));
    return (d1 >= eps && d2 >= eps && d3 >= eps) || (d1 <= -eps && d2 <= -eps && d3 <= -eps);
}

// Ball/bull edge contact for a ball centre on the circle of radius `flat` at angle `ang`:
// returns (tip z, ok) for the line a + t d, t in [0, 1]. Heights are solved relative to a.z.
fn edge_eval(xy: vec2<f32>, ang: f32, a: vec3<f32>, d: vec3<f32>) -> vec2<f32> {
    let q = xy + vec2<f32>(cos(ang), sin(ang)) * P.flat;
    let w = vec3<f32>(q.x - a.x, q.y - a.y, 0.0);
    let wxd = cross(w, d);
    let exd = vec3<f32>(-d.y, d.x, 0.0);
    let aa = dot(exd, exd);
    if (aa < 1e-14) {
        return vec2<f32>(0.0, 0.0);
    }
    let bb = 2.0 * dot(wxd, exd);
    let cc = dot(wxd, wxd) - P.rc * P.rc * dot(d, d);
    let disc = bb * bb - 4.0 * aa * cc;
    if (disc < 0.0) {
        return vec2<f32>(0.0, 0.0);
    }
    let cz = (-bb + sqrt(disc)) / (2.0 * aa);        // centre height above a.z
    let t = dot(vec3<f32>(w.x, w.y, cz), d) / dot(d, d);
    if (t < 0.0 || t > 1.0) {
        return vec2<f32>(0.0, 0.0);
    }
    return vec2<f32>(a.z + cz - P.rc, 1.0);
}

fn seg_dist(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let d = b - a;
    let l2 = dot(d, d);
    var t = 0.0;
    if (l2 > 0.0) {
        t = clamp(dot(p - a, d) / l2, 0.0, 1.0);
    }
    return length(p - (a + d * t));
}

fn edge(xy: vec2<f32>, a: vec3<f32>, b: vec3<f32>, best: f32) -> f32 {
    var z = best;
    if (seg_dist(xy, a.xy, b.xy) > P.r + 1e-6) {
        return z;
    }
    let d = b - a;
    if (P.rc <= 1e-9) {
        // Flat end: where the edge crosses the rim of the bottom disc.
        let f = a.xy - xy;
        let A = dot(d.xy, d.xy);
        if (A < 1e-14) {
            return z;
        }
        let B = 2.0 * dot(f, d.xy);
        let C = dot(f, f) - P.r * P.r;
        let disc = B * B - 4.0 * A * C;
        if (disc < 0.0) {
            return z;
        }
        let s = sqrt(disc);
        let t1 = (-B - s) / (2.0 * A);
        let t2 = (-B + s) / (2.0 * A);
        if (t1 >= 0.0 && t1 <= 1.0) {
            z = max(z, a.z + d.z * t1);
        }
        if (t2 >= 0.0 && t2 <= 1.0) {
            z = max(z, a.z + d.z * t2);
        }
        return z;
    }
    if (P.flat <= 1e-9) {
        let e = edge_eval(xy, 0.0, a, d);
        if (e.y > 0.5) {
            z = max(z, e.x);
        }
        return z;
    }
    // Bull-nose: 24 samples round the circle, then golden-section refinement.
    let n = 24;
    var best_ang = 0.0;
    var best_z = -3.0e38;
    for (var k = 0; k < n; k = k + 1) {
        let ang = f32(k) / f32(n) * 6.2831853;
        let e = edge_eval(xy, ang, a, d);
        if (e.y > 0.5 && e.x > best_z) {
            best_z = e.x;
            best_ang = ang;
        }
    }
    if (best_z <= -3.0e38) {
        return z;
    }
    let step = 6.2831853 / f32(n);
    var lo = best_ang - step;
    var hi = best_ang + step;
    let gr = 0.618034;
    for (var k = 0; k < 30; k = k + 1) {
        let m1 = hi - gr * (hi - lo);
        let m2 = lo + gr * (hi - lo);
        let e1 = edge_eval(xy, m1, a, d);
        let e2 = edge_eval(xy, m2, a, d);
        var f1 = -3.0e38;
        var f2 = -3.0e38;
        if (e1.y > 0.5) { f1 = e1.x; }
        if (e2.y > 0.5) { f2 = e2.x; }
        if (f1 < f2) { lo = m1; } else { hi = m2; }
    }
    let e = edge_eval(xy, (lo + hi) * 0.5, a, d);
    if (e.y > 0.5 && e.x > best_z) {
        best_z = e.x;
    }
    return max(z, best_z);
}

fn drop_tri(xy: vec2<f32>, t: Tri, best_in: f32) -> f32 {
    var best = best_in;
    let a = t.a.xyz;
    let b = t.b.xyz;
    let c = t.c.xyz;
    // Vertices.
    for (var k = 0; k < 3; k = k + 1) {
        var v = a;
        if (k == 1) { v = b; }
        if (k == 2) { v = c; }
        let rho = length(v.xy - xy);
        if (rho <= P.r + 1e-6) {
            best = max(best, v.z - profile(rho));
        }
    }
    // Facet (facing up).
    let n = t.n.xyz;
    if (n.z > 1e-9) {
        let nxy = n.xy;
        let l = length(nxy);
        var q = xy;
        if (l > 1e-9) {
            q = xy - nxy * (P.flat / l);
        }
        // Ball centre height relative to the vertex a: n.(Q - a) = rc.
        let cz = a.z + (P.rc - n.x * (q.x - a.x) - n.y * (q.y - a.y)) / n.z;
        let contact = vec3<f32>(q.x, q.y, cz) - n * P.rc;
        if (in_tri(contact.xy, a.xy, b.xy, c.xy)) {
            best = max(best, cz - P.rc);
        }
    }
    // Edges.
    best = edge(xy, a, b, best);
    best = edge(xy, b, c, best);
    best = edge(xy, c, a, best);
    return best;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = P.row0 + id.y;
    if (i >= P.nx || id.y >= P.rows || j >= P.ny) {
        return;
    }
    var best = P.floor;
    let ii = i32(i);
    let jj = i32(j);
    let reach = i32(P.reach);
    let bc = i32(P.bin_cells);
    let sh = i32(P.shift);
    let bx0 = max((ii - reach + sh) / bc, 0);
    let bx1 = min((ii + reach + sh) / bc, i32(P.nbx) - 1);
    let by0 = max((jj - reach + sh) / bc, 0);
    let by1 = min((jj + reach + sh) / bc, i32(P.nby) - 1);
    for (var by = by0; by <= by1; by = by + 1) {
        for (var bx = bx0; bx <= bx1; bx = bx + 1) {
            let bin = bins[u32(by) * P.nbx + u32(bx)];
            // This cell relative to the bin's reference node (small, exact integers).
            let xy = vec2<f32>(f32(ii - bin.ref_i), f32(jj - bin.ref_j)) * P.cell;
            for (var k = bin.start; k < bin.end; k = k + 1) {
                let t = tris[k];
                if (t.n.w <= best) {
                    break; // sorted by top: nothing further in this bin reaches higher
                }
                // Nearest point of the triangle's XY box; past the tool's reach, or unable to
                // lift the tip above what we have, it is skipped.
                let g = max(max(t.bb.xy - xy, xy - t.bb.zw), vec2<f32>(0.0, 0.0));
                let dmin = length(g);
                if (dmin > P.r + 1e-6 || t.n.w - profile(dmin) <= best) {
                    continue;
                }
                best = drop_tri(xy, t, best);
            }
        }
    }
    out[id.y * P.nx + i] = best;
}
