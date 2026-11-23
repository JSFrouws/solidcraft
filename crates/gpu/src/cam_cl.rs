//! The CAM cutter-location map on the GPU (`cam_cl.wgsl`): the accelerator the apps install into
//! `solidcraft-cam` ([`install_cam_accelerator`]). The CAM crate uses it only when
//! `SOLIDCRAFT_CAM_GPU=1` asks for it. The device opens on first use; any failure declines, and
//! the CAM crate then computes the map on the CPU as before.
//!
//! Failures never reach wgpu's uncaptured-error handler, which panics by default: buffers,
//! pipelines and submissions run inside error scopes ([`Device::scoped`]), every wait has a
//! timeout ([`Device::wait`]), and a buffer larger than the device can bind is declined before
//! it is made. A lost device, a timeout or an error outside a scope retires the device for the
//! rest of the process (the CPU does the work from then on).
//!
//! The shader works in f32. Triangles are stored per XY bin relative to the bin's reference grid
//! node and heights relative to the mesh's middle, so the offsets it works with stay small: the
//! heights agree with the CPU's f64 drop cutter to about a micrometre.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use solidcraft_cam::dropcutter::DropMesh;
use solidcraft_cam::heightmap::HeightMap;
use solidcraft_cam::tool::Tool;
use wgpu::util::DeviceExt;

const SHADER: &str = include_str!("cam_cl.wgsl");
/// Floats per triangle (a, b, c, n with the top in w, XY box), as `Tri` in the shader.
const TRI_WORDS: usize = 20;
/// Cells per dispatch, so one dispatch stays far below the system's GPU watchdog.
const CHUNK_CELLS: usize = 1 << 18;

/// The longest one submission may take before the device is given up on. A chunk takes well
/// under a second; this only catches a hung device.
const WAIT: Duration = Duration::from_secs(30);

/// The device the CAM accelerators share, opened on first use (`None` when there is none).
pub(crate) struct Device {
    pub(crate) name: String,
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    /// Cleared for good on a lost device, an uncaptured error or a timeout.
    healthy: Arc<AtomicBool>,
}

impl Device {
    /// Whether the device can still be used.
    pub(crate) fn healthy(&self) -> bool {
        self.healthy.load(Ordering::Acquire)
    }

    /// Runs `work` (buffer and pipeline creation, encoding, submissions) inside validation,
    /// out-of-memory and internal error scopes: an error there comes back as `Err` instead of
    /// going to the uncaptured-error handler.
    pub(crate) fn scoped<T>(&self, work: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
        if !self.healthy() {
            return Err("the GPU was retired after an earlier failure".into());
        }
        let scopes =
            [wgpu::ErrorFilter::Validation, wgpu::ErrorFilter::OutOfMemory, wgpu::ErrorFilter::Internal].map(|f| self.device.push_error_scope(f));
        let result = work();
        let mut error = None;
        for scope in scopes.into_iter().rev() {
            if let Some(e) = pollster::block_on(scope.pop()) {
                error = Some(e.to_string());
            }
        }
        match error {
            Some(e) => Err(e),
            None => result,
        }
    }

    /// Waits for `submission`, at most [`WAIT`]; a timeout or a lost device retires the device.
    pub(crate) fn wait(&self, submission: wgpu::SubmissionIndex) -> Result<(), String> {
        let done = self.device.poll(wgpu::PollType::Wait { submission_index: Some(submission), timeout: Some(WAIT) });
        if let Err(e) = done {
            self.healthy.store(false, Ordering::Release);
            return Err(format!("the GPU did not finish ({e}); CAM stays on the CPU from now on"));
        }
        if !self.healthy() {
            return Err("the GPU was lost".into());
        }
        Ok(())
    }

    /// Reads the first `size` bytes of a mappable buffer back once the copy into it was
    /// submitted as `submission`.
    pub(crate) fn read(&self, buf: &wgpu::Buffer, size: u64, submission: wgpu::SubmissionIndex) -> Result<Vec<u8>, String> {
        let slice = buf.slice(..size);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.wait(submission)?;
        rx.recv_timeout(Duration::from_secs(1)).map_err(|e| e.to_string())?.map_err(|e| e.to_string())?;
        let bytes = slice.get_mapped_range().map_err(|e| e.to_string())?.to_vec();
        buf.unmap();
        Ok(bytes)
    }

    /// Declines a storage buffer of `bytes` the device cannot bind.
    pub(crate) fn fits(&self, what: &str, bytes: u64) -> Result<(), String> {
        let l = self.device.limits();
        let most = l.max_storage_buffer_binding_size.min(l.max_buffer_size);
        if bytes > most { Err(format!("{what} needs {bytes} bytes; the GPU binds at most {most}")) } else { Ok(()) }
    }
}

static DEVICE: OnceLock<Option<Device>> = OnceLock::new();

/// The shared CAM device: the best hardware adapter, with its full buffer limits.
pub(crate) fn device() -> Option<&'static Device> {
    DEVICE
        .get_or_init(|| match open_device() {
            Ok(d) => {
                log::info!("CAM on the GPU: {}", d.name);
                Some(d)
            }
            Err(e) => {
                log::info!("CAM on the CPU (no GPU: {e})");
                None
            }
        })
        .as_ref()
}

fn open_device() -> Result<Device, String> {
    let (adapter, _) =
        solidcraft_viewport_gpu::select::pick(solidcraft_viewport_gpu::select::Policy::Hardware).ok_or_else(|| "no hardware adapter".to_string())?;
    let name = adapter.get_info().name;
    let limits = adapter.limits();
    let required = wgpu::Limits {
        max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
        max_buffer_size: limits.max_buffer_size,
        max_storage_buffers_per_shader_stage: limits.max_storage_buffers_per_shader_stage.max(6),
        ..wgpu::Limits::downlevel_defaults()
    };
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("solidcraft cam"),
        required_limits: required,
        ..Default::default()
    }))
    .map_err(|e| e.to_string())?;
    // Errors outside a scope (there should be none) and a lost device retire the device instead
    // of panicking; the CPU takes over.
    let healthy = Arc::new(AtomicBool::new(true));
    let uncaptured = healthy.clone();
    device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
        uncaptured.store(false, Ordering::Release);
        log::error!("CAM GPU retired after an uncaptured error: {e}");
    }));
    let lost = healthy.clone();
    device.set_device_lost_callback(move |reason, message| {
        lost.store(false, Ordering::Release);
        log::warn!("CAM GPU lost ({reason:?}): {message}");
    });
    Ok(Device { name, device, queue, healthy })
}

struct Gpu {
    shared: &'static Device,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

/// The accelerator: the device is opened the first time a map is asked for.
struct GpuClMap {
    gpu: OnceLock<Option<Gpu>>,
    /// The GPU path is used by one map at a time (the maps are large; queueing them would only
    /// split the device between them).
    busy: Mutex<()>,
}

/// Installs the GPU accelerators into the CAM crate (native builds): the cutter-location map and
/// the stock sweep. `false` when they were installed already. Cheap: no device is opened until
/// one is used.
pub fn install_cam_accelerator() -> bool {
    let a = solidcraft_cam::accel::set_cl_map_accelerator(Box::new(GpuClMap { gpu: OnceLock::new(), busy: Mutex::new(()) }));
    let b = crate::cam_sweep::install();
    a && b
}

impl GpuClMap {
    fn gpu(&self) -> Option<&Gpu> {
        self.gpu
            .get_or_init(|| match device().map(open) {
                Some(Ok(g)) => Some(g),
                Some(Err(e)) => {
                    log::warn!("CAM cutter-location maps on the CPU: {e}");
                    None
                }
                None => None,
            })
            .as_ref()
    }
}

fn open(shared: &'static Device) -> Result<Gpu, String> {
    shared.scoped(|| open_in(shared))
}

fn open_in(shared: &'static Device) -> Result<Gpu, String> {
    let device = &shared.device;
    let module =
        device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("cam cl map"), source: wgpu::ShaderSource::Wgsl(SHADER.into()) });
    let entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    };
    let storage = |read_only| wgpu::BufferBindingType::Storage { read_only };
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("cam cl map"),
        entries: &[entry(0, wgpu::BufferBindingType::Uniform), entry(1, storage(true)), entry(2, storage(true)), entry(3, storage(false))],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("cam cl map"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("cam cl map"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    Ok(Gpu { shared, pipeline, layout })
}

impl solidcraft_cam::accel::ClMapAccelerator for GpuClMap {
    fn cl_map(&self, mesh: &DropMesh, tool: &Tool, grid: &HeightMap, floor: f64) -> Option<Vec<f64>> {
        if !tool.is_end_mill() || mesh.is_empty() || grid.nx == 0 || grid.ny == 0 {
            return None;
        }
        let gpu = self.gpu()?;
        let _one = self.busy.lock().ok()?;
        match gpu.shared.scoped(|| run(gpu, mesh, tool, grid, floor)) {
            Ok(z) => Some(z),
            Err(e) => {
                log::warn!("CAM cutter-location map on the GPU failed, using the CPU: {e}");
                None
            }
        }
    }

    fn device_name(&self) -> String {
        device().map_or_else(|| "GPU".to_string(), |d| d.name.clone())
    }
}

fn run(gpu: &Gpu, mesh: &DropMesh, tool: &Tool, grid: &HeightMap, floor: f64) -> Result<Vec<f64>, String> {
    let (nx, ny, cell, origin) = (grid.nx, grid.ny, grid.cell, grid.origin);
    let r = tool.radius();
    let (rc, flat) = (tool.corner_radius(), tool.flat_radius());
    if !(r > 0.0 && cell > 0.0) {
        return Err("tool radius or cell".into());
    }
    let z_ref = (mesh.bounds.min.z + mesh.bounds.max.z) * 0.5;
    // Bins of `bc` x `bc` cells, at least the tool's reach wide, padded by `shift` cells round
    // the grid so triangles just outside it are found too.
    let reach = (r / cell).ceil() as usize + 1;
    let bc = reach.max(4);
    let shift = bc;
    let nbx = (nx + 2 * shift).div_ceil(bc) + 1;
    let nby = (ny + 2 * shift).div_ceil(bc) + 1;
    if nbx.checked_mul(nby).is_none_or(|n| n > 50_000_000) {
        return Err("too many bins".into());
    }
    let col = |x: f64| ((x - origin.x) / cell).floor();
    let row = |y: f64| ((y - origin.y) / cell).floor();
    // Triangles per bin.
    let mut lists: Vec<Vec<u32>> = vec![Vec::new(); nbx * nby];
    for (t, tri) in mesh.tris.iter().enumerate() {
        let (x0, x1) = (tri[0].x.min(tri[1].x).min(tri[2].x), tri[0].x.max(tri[1].x).max(tri[2].x));
        let (y0, y1) = (tri[0].y.min(tri[1].y).min(tri[2].y), tri[0].y.max(tri[1].y).max(tri[2].y));
        let top = tri[0].z.max(tri[1].z).max(tri[2].z);
        if top < floor {
            continue; // cannot lift the tool off the floor
        }
        // Farther than the padding from the grid: no cell's tool reaches it.
        let s = shift as f64;
        if col(x1) < -s || row(y1) < -s || col(x0) >= nx as f64 + s || row(y0) >= ny as f64 + s {
            continue;
        }
        let clampb = |c: f64, nb: usize| ((c + s) / bc as f64).floor().clamp(0.0, nb as f64 - 1.0) as usize;
        let (bx0, bx1) = (clampb(col(x0), nbx), clampb(col(x1), nbx));
        let (by0, by1) = (clampb(row(y0), nby), clampb(row(y1), nby));
        for by in by0..=by1 {
            for bx in bx0..=bx1 {
                if let Some(l) = lists.get_mut(by * nbx + bx) {
                    l.push(t as u32);
                }
            }
        }
    }
    // Pack: each bin's triangles relative to its reference node, highest first.
    let top_of = |t: u32| mesh.tris.get(t as usize).map_or(f64::NEG_INFINITY, |p| p[0].z.max(p[1].z).max(p[2].z));
    let total: usize = lists.iter().map(Vec::len).sum();
    let mut tris_f: Vec<f32> = Vec::with_capacity(total.max(1) * TRI_WORDS);
    let mut bins_w: Vec<u32> = Vec::with_capacity(nbx * nby * 4);
    for by in 0..nby {
        for bx in 0..nbx {
            let list = lists.get_mut(by * nbx + bx).ok_or("bin")?;
            list.sort_by(|a, b| top_of(*b).total_cmp(&top_of(*a)));
            let ref_i = (bx * bc) as i64 - shift as i64 + (bc / 2) as i64;
            let ref_j = (by * bc) as i64 - shift as i64 + (bc / 2) as i64;
            let (rx, ry) = (origin.x + ref_i as f64 * cell, origin.y + ref_j as f64 * cell);
            let start = tris_f.len() / TRI_WORDS;
            for &t in list.iter() {
                let (Some(p), Some(n)) = (mesh.tris.get(t as usize), mesh.normals.get(t as usize)) else { continue };
                let top = p[0].z.max(p[1].z).max(p[2].z);
                for v in p {
                    tris_f.extend_from_slice(&[(v.x - rx) as f32, (v.y - ry) as f32, (v.z - z_ref) as f32, 0.0]);
                }
                tris_f.extend_from_slice(&[n.x as f32, n.y as f32, n.z as f32, (top - z_ref) as f32]);
                let (x0, x1) = (p[0].x.min(p[1].x).min(p[2].x), p[0].x.max(p[1].x).max(p[2].x));
                let (y0, y1) = (p[0].y.min(p[1].y).min(p[2].y), p[0].y.max(p[1].y).max(p[2].y));
                tris_f.extend_from_slice(&[(x0 - rx) as f32, (y0 - ry) as f32, (x1 - rx) as f32, (y1 - ry) as f32]);
            }
            let end = tris_f.len() / TRI_WORDS;
            bins_w.extend_from_slice(&[start as u32, end as u32, ref_i as i32 as u32, ref_j as i32 as u32]);
        }
    }
    if tris_f.is_empty() {
        tris_f.resize(TRI_WORDS, 0.0);
    }
    let d = &gpu.shared.device;
    gpu.shared.fits("the triangle list", (tris_f.len() * 4) as u64)?;
    gpu.shared.fits("the bin table", (bins_w.len() * 4) as u64)?;
    let to_bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let tris_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cam tris"),
        contents: &to_bytes(&tris_f),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let bins_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cam bins"),
        contents: &bins_w.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>(),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let rows_per = (CHUNK_CELLS / nx).clamp(1, ny);
    let out_size = (nx * rows_per * 4) as u64;
    let out = d.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cam cl out"),
        size: out_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read = d.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cam cl read"),
        size: out_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let floor_rel = (floor - z_ref) as f32;
    let mut result = vec![floor; nx * ny];
    let mut row0 = 0usize;
    while row0 < ny {
        let rows = rows_per.min(ny - row0);
        let words: [u32; 16] = [
            (cell as f32).to_bits(),
            floor_rel.to_bits(),
            (r as f32).to_bits(),
            (rc as f32).to_bits(),
            (flat as f32).to_bits(),
            nx as u32,
            ny as u32,
            row0 as u32,
            rows as u32,
            bc as u32,
            shift as u32,
            nbx as u32,
            nby as u32,
            reach as u32,
            0,
            0,
        ];
        let params = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cam cl params"),
            contents: &words.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>(),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cam cl map"),
            layout: &gpu.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: tris_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: bins_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: out.as_entire_binding() },
            ],
        });
        let mut enc = d.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("cam cl map") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("cam cl map"), timestamp_writes: None });
            pass.set_pipeline(&gpu.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups((nx as u32).div_ceil(16), (rows as u32).div_ceil(16), 1);
        }
        let bytes = (nx * rows * 4) as u64;
        enc.copy_buffer_to_buffer(&out, 0, &read, 0, bytes);
        let submission = gpu.shared.queue.submit([enc.finish()]);
        let data = gpu.shared.read(&read, bytes, submission)?;
        for (k, b) in data.as_chunks::<4>().0.iter().enumerate() {
            if let Some(slot) = result.get_mut(row0 * nx + k) {
                let v = f64::from(f32::from_le_bytes(*b));
                *slot = if v > f64::from(floor_rel) { v + z_ref } else { floor };
            }
        }
        row0 += rows;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    /// A validation error on the CAM device comes back from `scoped` as an error (the GPU path
    /// then declines and the CPU does the work) instead of reaching wgpu's uncaptured-error
    /// handler, which panics; the device stays usable. Skipped on a machine without a GPU.
    #[test]
    fn gpu_errors_come_back_instead_of_panicking() {
        let Some(dev) = super::device() else {
            eprintln!("no GPU: skipped");
            return;
        };
        let bad = dev.scoped(|| {
            // MAP_READ with MAP_WRITE is not a valid buffer.
            let _b = dev.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("invalid on purpose"),
                size: 16,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::MAP_WRITE,
                mapped_at_creation: false,
            });
            Ok(())
        });
        assert!(bad.is_err(), "the validation error is reported");
        assert!(dev.healthy(), "a captured error does not retire the device");
        assert!(dev.fits("a huge buffer", u64::MAX).is_err());
        let good = dev.scoped(|| {
            let _b = dev.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fine"),
                size: 16,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            Ok(7)
        });
        assert_eq!(good, Ok(7));
    }
}
