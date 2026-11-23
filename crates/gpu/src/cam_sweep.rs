//! The CAM stock sweep on the GPU (`cam_sweep.wgsl`), installed with the cutter-location map
//! ([`crate::cam_cl::install_cam_accelerator`]) and used only when `SOLIDCRAFT_CAM_GPU=1` asks
//! for it. One dispatch for the cuts between two rapids (a workgroup per move, lowering its cells
//! with atomic minimums), one for each rapid's check, in program order. The dispatches go in
//! submissions of at most [`MOVES_PER_SUBMIT`] moves, each waited for with a timeout, so a long
//! program stays under the system's GPU watchdog. Heights the program cut come back in f32
//! (micrometre-level, against a simulation tolerance of hundredths of a millimetre); heights it
//! did not touch keep their exact value. Everything runs inside the device's error scopes
//! ([`crate::cam_cl`]); any failure declines, and the CPU sweeps instead.

use std::sync::{Mutex, OnceLock};

use solidcore_record::Vec3;
use solidcraft_cam::simulate::StockMap;
use solidcraft_cam::tool::Tool;
use wgpu::util::DeviceExt;

const SHADER: &str = include_str!("cam_sweep.wgsl");
/// Bytes of one step's parameters (`Step` in the shader).
const STEP_BYTES: usize = 64;
/// Most workgroups in one dispatch dimension.
const MAX_GROUPS: usize = 65_535;
/// Most moves (cuts and rapid checks) in one submission (fewer in the tests, so they cross
/// submissions).
#[cfg(not(test))]
const MOVES_PER_SUBMIT: usize = 32_768;
#[cfg(test)]
const MOVES_PER_SUBMIT: usize = 700;
/// Most dispatches in one submission (each takes one parameter slot).
const STEPS_PER_SUBMIT: usize = 4096;

struct Sweep {
    shared: &'static crate::cam_cl::Device,
    cut: wgpu::ComputePipeline,
    rapid: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    align: usize,
}

struct GpuSweep {
    gpu: OnceLock<Option<Sweep>>,
    busy: Mutex<()>,
}

pub(crate) fn install() -> bool {
    solidcraft_cam::accel::set_sweep_accelerator(Box::new(GpuSweep { gpu: OnceLock::new(), busy: Mutex::new(()) }))
}

fn open(shared: &'static crate::cam_cl::Device) -> Option<Sweep> {
    match shared.scoped(|| Ok(open_in(shared))) {
        Ok(s) => Some(s),
        Err(e) => {
            log::warn!("CAM stock sweeps on the CPU: {e}");
            None
        }
    }
}

fn open_in(shared: &'static crate::cam_cl::Device) -> Sweep {
    let device = &shared.device;
    let module =
        device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("cam sweep"), source: wgpu::ShaderSource::Wgsl(SHADER.into()) });
    let entry = |binding: u32, ty: wgpu::BufferBindingType, dynamic: bool| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: dynamic, min_binding_size: None },
        count: None,
    };
    let storage = |read_only| wgpu::BufferBindingType::Storage { read_only };
    let uniform = wgpu::BufferBindingType::Uniform;
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("cam sweep"),
        entries: &[
            entry(0, uniform, false),
            entry(1, uniform, true),
            entry(2, storage(true), false),
            entry(3, storage(false), false),
            entry(4, storage(true), false),
            entry(5, storage(true), false),
            entry(6, storage(false), false),
        ],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("cam sweep"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = |entry: &str| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cam sweep"),
            layout: Some(&pl),
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let (cut, rapid) = (pipeline("cut"), pipeline("rapid"));
    let align = (device.limits().min_uniform_buffer_offset_alignment as usize).max(STEP_BYTES);
    Sweep { shared, cut, rapid, layout, align }
}

/// f32 to the order-preserving key the shader keeps (as `key` there).
fn key(f: f32) -> u32 {
    let b = f.to_bits();
    if b & 0x8000_0000 != 0 { !b } else { b | 0x8000_0000 }
}

fn unkey(u: u32) -> f32 {
    if u & 0x8000_0000 != 0 { f32::from_bits(u & 0x7fff_ffff) } else { f32::from_bits(!u) }
}

impl solidcraft_cam::accel::SweepAccelerator for GpuSweep {
    fn sweep(&self, stock: &mut StockMap, tool: &Tool, cuts: &[(Vec3, Vec3)], rapids: &[(usize, Vec3, Vec3)], tol: f64) -> Option<Vec<bool>> {
        if !tool.is_end_mill() {
            return None;
        }
        let gpu = self.gpu.get_or_init(|| crate::cam_cl::device().and_then(open)).as_ref()?;
        let _one = self.busy.lock().ok()?;
        match gpu.shared.scoped(|| run(gpu, stock, tool, cuts, rapids, tol)) {
            Ok(h) => Some(h),
            Err(e) => {
                log::warn!("CAM stock sweep on the GPU failed, using the CPU: {e}");
                None
            }
        }
    }

    fn device_name(&self) -> String {
        crate::cam_cl::device().map_or_else(|| "GPU".to_string(), |d| d.name.clone())
    }
}

enum Step {
    Cut { first: u32, count: u32 },
    Rapid { index: u32, i0: u32, j0: u32, wi: u32, wj: u32, a: Vec3, b: Vec3 },
}

fn run(gpu: &Sweep, stock: &mut StockMap, tool: &Tool, cuts: &[(Vec3, Vec3)], rapids: &[(usize, Vec3, Vec3)], tol: f64) -> Result<Vec<bool>, String> {
    let (nx, ny) = (stock.map.nx, stock.map.ny);
    let (o, cell) = (stock.map.origin, stock.map.cell);
    let r = tool.radius();
    if nx == 0 || ny == 0 || !(cell > 0.0) {
        return Err("empty stock".into());
    }
    let n = nx.checked_mul(ny).ok_or("grid")?;
    // Per column: the stock's key, its height before the sweep, and its floor.
    let start: Vec<f32> = stock.map.z.iter().map(|z| *z as f32).collect();
    let keys: Vec<u32> = start.iter().map(|z| key(*z)).collect();
    let lows64: Vec<f64> = (0..n).map(|k| stock.floor_at(k % nx, k / nx)).collect();
    let lows: Vec<f32> = lows64.iter().map(|z| *z as f32).collect();
    let rel = |p: Vec3| [(p.x - o.x) as f32, (p.y - o.y) as f32, p.z as f32, 0.0f32];
    let mut seg_words: Vec<f32> = Vec::with_capacity(cuts.len() * 8 + 8);
    for (a, b) in cuts {
        seg_words.extend_from_slice(&rel(*a));
        seg_words.extend_from_slice(&rel(*b));
    }
    if seg_words.is_empty() {
        seg_words.resize(8, 0.0);
    }
    // The steps in program order: the cuts before each rapid, then the rapid.
    let mut steps: Vec<Step> = Vec::new();
    let push_cuts = |from: usize, to: usize, steps: &mut Vec<Step>| {
        let mut f = from;
        while f < to {
            let c = (to - f).min(MAX_GROUPS).min(MOVES_PER_SUBMIT);
            steps.push(Step::Cut { first: f as u32, count: c as u32 });
            f += c;
        }
    };
    let mut done = 0usize;
    for (k, (before, a, b)) in rapids.iter().enumerate() {
        let before = (*before).min(cuts.len());
        push_cuts(done, before, &mut steps);
        done = done.max(before);
        let lo = ((a.x.min(b.x) - r - o.x) / cell).floor() - 1.0;
        let hi = ((a.x.max(b.x) + r - o.x) / cell).ceil() + 1.0;
        let lo_j = ((a.y.min(b.y) - r - o.y) / cell).floor() - 1.0;
        let hi_j = ((a.y.max(b.y) + r - o.y) / cell).ceil() + 1.0;
        let i0 = lo.max(0.0);
        let j0 = lo_j.max(0.0);
        let i1 = hi.min(nx as f64 - 1.0);
        let j1 = hi_j.min(ny as f64 - 1.0);
        if i1 < i0 || j1 < j0 {
            continue; // off the stock: nothing to hit
        }
        let (wi, wj) = ((i1 - i0) as u32 + 1, (j1 - j0) as u32 + 1);
        if (wi as usize * wj as usize).div_ceil(256) > MAX_GROUPS {
            return Err("rapid window too large".into());
        }
        steps.push(Step::Rapid { index: k as u32, i0: i0 as u32, j0: j0 as u32, wi, wj, a: *a, b: *b });
    }
    push_cuts(done, cuts.len(), &mut steps);
    if steps.is_empty() {
        return Ok(vec![false; rapids.len()]);
    }
    let shared = gpu.shared;
    let d = &shared.device;
    shared.fits("the moves", (seg_words.len() * 4) as u64)?;
    shared.fits("the stock", (n * 4) as u64)?;
    let bytes_u32 = |v: &[u32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let bytes_f32 = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let init = |label: &str, contents: &[u8], usage: wgpu::BufferUsages| {
        d.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents, usage })
    };
    let grid = [
        nx as u32,
        ny as u32,
        (cell as f32).to_bits(),
        (r as f32).to_bits(),
        (tool.corner_radius() as f32).to_bits(),
        (tool.flat_radius() as f32).to_bits(),
        (tol as f32).to_bits(),
        0,
    ];
    let grid_buf = init("cam sweep grid", &bytes_u32(&grid), wgpu::BufferUsages::UNIFORM);
    let seg_buf = init("cam sweep segments", &bytes_f32(&seg_words), wgpu::BufferUsages::STORAGE);
    let stock_buf = init("cam sweep stock", &bytes_u32(&keys), wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
    let start_buf = init("cam sweep start", &bytes_f32(&start), wgpu::BufferUsages::STORAGE);
    let low_buf = init("cam sweep floor", &bytes_f32(&lows), wgpu::BufferUsages::STORAGE);
    let hits0 = vec![0u32; rapids.len().max(1)];
    let hit_buf = init("cam sweep hits", &bytes_u32(&hits0), wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
    let read_buf = |label: &str, size: u64| {
        d.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };
    let (stock_bytes, hit_bytes) = ((n * 4) as u64, (hits0.len() * 4) as u64);
    let stock_read = read_buf("cam sweep stock read", stock_bytes);
    let hit_read = read_buf("cam sweep hits read", hit_bytes);
    // The steps in submissions of at most MOVES_PER_SUBMIT moves and STEPS_PER_SUBMIT dispatches,
    // in order (the stock buffer carries the heights from one to the next); the last one also
    // copies the stock and the rapid verdicts out.
    let mut chunks: Vec<&[Step]> = Vec::new();
    let (mut from, mut moves) = (0usize, 0usize);
    for (k, st) in steps.iter().enumerate() {
        let m = match st {
            Step::Cut { count, .. } => *count as usize,
            Step::Rapid { .. } => 1,
        };
        if k > from && (moves + m > MOVES_PER_SUBMIT || k - from >= STEPS_PER_SUBMIT) {
            chunks.push(steps.get(from..k).ok_or("steps")?);
            (from, moves) = (k, 0);
        }
        moves += m;
    }
    chunks.push(steps.get(from..).ok_or("steps")?);
    let last = chunks.len() - 1;
    let mut submission = None;
    for (c, chunk) in chunks.iter().enumerate() {
        // Step parameters, one aligned slot each.
        let mut step_bytes = vec![0u8; chunk.len() * gpu.align];
        for (k, st) in chunk.iter().enumerate() {
            let mut w = [0u32; 16];
            match st {
                Step::Cut { first, count } => {
                    w[0] = *first;
                    w[1] = *count;
                }
                Step::Rapid { index, i0, j0, wi, wj, a, b } => {
                    w[0] = *index;
                    w[2] = *i0;
                    w[3] = *j0;
                    w[4] = *wi;
                    w[5] = *wj;
                    let (pa, pb) = (rel(*a), rel(*b));
                    for q in 0..4 {
                        w[8 + q] = pa[q].to_bits();
                        w[12 + q] = pb[q].to_bits();
                    }
                }
            }
            let at = k * gpu.align;
            for (q, v) in w.iter().enumerate() {
                if let Some(slot) = step_bytes.get_mut(at + 4 * q..at + 4 * q + 4) {
                    slot.copy_from_slice(&v.to_le_bytes());
                }
            }
        }
        let step_buf = init("cam sweep steps", &step_bytes, wgpu::BufferUsages::UNIFORM);
        let bind = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cam sweep"),
            layout: &gpu.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: grid_buf.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &step_buf,
                        offset: 0,
                        size: wgpu::BufferSize::new(STEP_BYTES as u64),
                    }),
                },
                wgpu::BindGroupEntry { binding: 2, resource: seg_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: stock_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: start_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: low_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: hit_buf.as_entire_binding() },
            ],
        });
        let mut enc = d.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("cam sweep") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("cam sweep"), timestamp_writes: None });
            for (k, st) in chunk.iter().enumerate() {
                let off = (k * gpu.align) as u32;
                match st {
                    Step::Cut { count, .. } => {
                        pass.set_pipeline(&gpu.cut);
                        pass.set_bind_group(0, &bind, &[off]);
                        pass.dispatch_workgroups(*count, 1, 1);
                    }
                    Step::Rapid { wi, wj, .. } => {
                        pass.set_pipeline(&gpu.rapid);
                        pass.set_bind_group(0, &bind, &[off]);
                        pass.dispatch_workgroups((*wi * *wj).div_ceil(256), 1, 1);
                    }
                }
            }
        }
        if c == last {
            enc.copy_buffer_to_buffer(&stock_buf, 0, &stock_read, 0, stock_bytes);
            enc.copy_buffer_to_buffer(&hit_buf, 0, &hit_read, 0, hit_bytes);
        }
        let sub = shared.queue.submit([enc.finish()]);
        if c != last {
            shared.wait(sub)?;
        } else {
            submission = Some(sub);
        }
    }
    let submission = submission.ok_or("no submission")?;
    let words = |b: Vec<u8>| -> Vec<u32> { b.as_chunks::<4>().0.iter().map(|w| u32::from_le_bytes(*w)).collect() };
    let out = words(shared.read(&stock_read, stock_bytes, submission.clone())?);
    let hits = words(shared.read(&hit_read, hit_bytes, submission)?);
    if out.len() != n {
        return Err("stock size".into());
    }
    // A column the program lowered gets its new height, clamped to its floor as `cut` does; one
    // it did not touch keeps its exact height.
    for (k, (z, u)) in stock.map.z.iter_mut().zip(&out).enumerate() {
        let s = unkey(*u);
        if let Some(s0) = start.get(k)
            && s < *s0
        {
            let low = lows64.get(k).copied().unwrap_or(f64::NEG_INFINITY);
            *z = f64::from(s).max(low);
        }
    }
    Ok(hits.iter().take(rapids.len()).map(|h| *h != 0).collect())
}

#[cfg(test)]
mod tests {
    use solidcore_record::Vec3;
    use solidcraft_cam::accel::SweepAccelerator;
    use solidcraft_cam::setup::StockShape;
    use solidcraft_cam::simulate::StockMap;
    use solidcraft_cam::tool::{Tool, ToolKind};

    /// The GPU sweep against the CPU's: random cuts with rapids among them, some of the rapids
    /// through the material, and moves that climb and descend. Skipped on a machine without a GPU.
    #[test]
    fn gpu_sweep_matches_the_cpu() {
        let acc = super::GpuSweep { gpu: std::sync::OnceLock::new(), busy: std::sync::Mutex::new(()) };
        let shape = StockShape::Box { min: Vec3::new(0.0, 0.0, -20.0), max: Vec3::new(60.0, 45.0, 0.0) };
        let mut seed = 99u64;
        let mut rnd = |lo: f64, hi: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            lo + (hi - lo) * ((seed >> 11) as f64 / (1u64 << 53) as f64)
        };
        let mut cuts = Vec::new();
        let mut rapids = Vec::new();
        for k in 0..4000 {
            let a = Vec3::new(rnd(-5.0, 65.0), rnd(-5.0, 50.0), rnd(-15.0, 1.0));
            let b = Vec3::new(a.x + rnd(-8.0, 8.0), a.y + rnd(-8.0, 8.0), a.z + rnd(-1.0, 1.0));
            cuts.push((a, b));
            if k % 97 == 0 {
                // Half above the stock, half through it.
                let z = if k % 2 == 0 { 5.0 } else { rnd(-12.0, -2.0) };
                rapids.push((cuts.len(), Vec3::new(rnd(0.0, 60.0), rnd(0.0, 45.0), z), Vec3::new(rnd(0.0, 60.0), rnd(0.0, 45.0), z)));
            }
        }
        for tool in [
            Tool::new(1, "ball", ToolKind::BallEnd, 6.0),
            Tool::new(2, "flat", ToolKind::FlatEnd, 8.0),
            Tool::new(3, "bull", ToolKind::BullNose { corner_radius: 1.5 }, 10.0),
        ] {
            let tol = 0.05;
            let mut cpu = StockMap::new(&shape, 0.25).unwrap();
            let mut gpu = cpu.clone();
            let mut done = 0;
            let mut cpu_hits = Vec::new();
            for (before, a, b) in &rapids {
                for (p, q) in &cuts[done..*before] {
                    cpu.cut(&tool, *p, *q);
                }
                done = *before;
                cpu_hits.push(cpu.rapid_hits(&tool, *a, *b, tol));
            }
            for (p, q) in &cuts[done..] {
                cpu.cut(&tool, *p, *q);
            }
            let Some(gpu_hits) = acc.sweep(&mut gpu, &tool, &cuts, &rapids, tol) else {
                eprintln!("no GPU: skipped");
                return;
            };
            assert!(cpu_hits.iter().any(|h| *h) && cpu_hits.iter().any(|h| !*h), "the case has rapids of both kinds");
            assert_eq!(gpu_hits, cpu_hits, "{}", tool.name);
            let worst = cpu.map.z.iter().zip(&gpu.map.z).filter(|(a, _)| a.is_finite()).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            assert!(worst < 1e-4, "{}: heights differ by {worst}", tool.name);
        }
    }
}
