//! The CAM cutter-location map on the GPU (`cam_cl.wgsl`): the accelerator the apps install into
//! `solidcraft-cam` ([`install_cam_accelerator`]). The device opens on first use; any failure
//! declines, and the CAM crate then computes the map on the CPU as before.
//!
//! The shader works in f32. Triangles are stored per XY bin relative to the bin's reference grid
//! node and heights relative to the mesh's middle, so the offsets it works with stay small: the
//! heights agree with the CPU's f64 drop cutter to about a micrometre.

use std::sync::{Mutex, OnceLock};

use solidcraft_cam::dropcutter::DropMesh;
use solidcraft_cam::heightmap::HeightMap;
use solidcraft_cam::tool::Tool;
use wgpu::util::DeviceExt;

const SHADER: &str = include_str!("cam_cl.wgsl");
/// Floats per triangle (a, b, c, n with the top in w, XY box), as `Tri` in the shader.
const TRI_WORDS: usize = 20;
/// Cells per dispatch, so one dispatch stays far below the system's GPU watchdog.
const CHUNK_CELLS: usize = 1 << 18;

/// The device the CAM accelerators share, opened on first use (`None` when there is none).
pub(crate) struct Device {
    pub(crate) name: String,
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
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
    Ok(Device { name, device, queue })
}

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
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

fn open(shared: &Device) -> Result<Gpu, String> {
    let (device, queue) = (shared.device.clone(), shared.queue.clone());
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
    Ok(Gpu { device, queue, pipeline, layout })
}

impl solidcraft_cam::accel::ClMapAccelerator for GpuClMap {
    fn cl_map(&self, mesh: &DropMesh, tool: &Tool, grid: &HeightMap, floor: f64) -> Option<Vec<f64>> {
        if !tool.is_end_mill() || mesh.is_empty() || grid.nx == 0 || grid.ny == 0 {
            return None;
        }
        let gpu = self.gpu()?;
        let _one = self.busy.lock().ok()?;
        match run(gpu, mesh, tool, grid, floor) {
            Ok(z) => Some(z),
            Err(e) => {
                log::warn!("CAM cutter-location map on the GPU failed, using the CPU: {e}");
                None
            }
        }
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
    let d = &gpu.device;
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
        gpu.queue.submit([enc.finish()]);
        let slice = read.slice(..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        d.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| e.to_string())?;
        rx.recv().map_err(|e| e.to_string())?.map_err(|e| e.to_string())?;
        {
            let data = slice.get_mapped_range().map_err(|e| e.to_string())?;
            for (k, b) in data.as_chunks::<4>().0.iter().enumerate() {
                if let Some(slot) = result.get_mut(row0 * nx + k) {
                    let v = f64::from(f32::from_le_bytes(*b));
                    *slot = if v > f64::from(floor_rel) { v + z_ref } else { floor };
                }
            }
        }
        read.unmap();
        row0 += rows;
    }
    Ok(result)
}
