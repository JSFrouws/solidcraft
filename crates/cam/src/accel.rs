//! Optional accelerators for the cutter-location map and the stock sweep (for example a GPU), installed by a
//! higher layer (the apps install the GPU one from `solidcraft-gpu`). The CAM crate stays
//! geometry only: without an accelerator, or when it declines, the map is computed on the CPU.
//!
//! Off unless asked for: an installed accelerator is used only when `SOLIDCRAFT_CAM_GPU=1` (or
//! `on`, `true`, `yes`). Its f32 arithmetic gives heights that depend on the device, so the same
//! design gives the same G-code everywhere only on the CPU path. When it is used, the
//! operation's notice says so and names the device ([`Used`]).

use std::sync::{Mutex, OnceLock};

use solidcore_record::Vec3;

use crate::dropcutter::DropMesh;
use crate::heightmap::HeightMap;
use crate::simulate::StockMap;
use crate::tool::Tool;

/// Computes cutter-location heights: for every cell of `grid` (row-major, `grid.point(i, j)`),
/// the tip height at which `tool`, coming down the world's z axis, first touches `mesh`, or
/// `floor` where it touches nothing above it. `None` declines (the CPU computes it instead).
pub trait ClMapAccelerator: Send + Sync {
    fn cl_map(&self, mesh: &DropMesh, tool: &Tool, grid: &HeightMap, floor: f64) -> Option<Vec<f64>>;
    /// The device that computed the last map, for the operation's notice.
    fn device_name(&self) -> String;
}

/// Sweeps a 3-axis toolpath through a Z-map stock: `cuts` lower the stock under the tool (in
/// order; their order does not change the heights), and each of `rapids` — `(cuts before it,
/// from, to)` — is checked against the stock as the cuts before it left it, with the stock's
/// `tol`. Lowers `stock` as [`StockMap::cut`] does and returns, per rapid, whether it passes
/// through material ([`StockMap::rapid_hits`]). `None` declines (the CPU sweeps instead).
pub trait SweepAccelerator: Send + Sync {
    fn sweep(&self, stock: &mut StockMap, tool: &Tool, cuts: &[(Vec3, Vec3)], rapids: &[(usize, Vec3, Vec3)], tol: f64) -> Option<Vec<bool>>;
    /// The device that ran the last sweep, for the simulation report.
    fn device_name(&self) -> String;
}

/// Which accelerator, if any, did part of an operation's work: the operation's notice reports it.
#[derive(Debug, Default)]
pub struct Used(Mutex<Option<String>>);

impl Used {
    pub(crate) fn record(&self, device: String) {
        if let Ok(mut d) = self.0.lock() {
            *d = Some(device);
        }
    }
    /// The device's name, when an accelerator was used.
    pub fn device(&self) -> Option<String> {
        self.0.lock().ok().and_then(|d| d.clone())
    }
}

static ACCELERATOR: OnceLock<Box<dyn ClMapAccelerator>> = OnceLock::new();
static SWEEP: OnceLock<Box<dyn SweepAccelerator>> = OnceLock::new();

/// Installs the stock-sweep accelerator for this process; `false` when one is installed already.
pub fn set_sweep_accelerator(a: Box<dyn SweepAccelerator>) -> bool {
    SWEEP.set(a).is_ok()
}

/// The installed sweep accelerator, when `SOLIDCRAFT_CAM_GPU` asks for it.
pub(crate) fn sweep_accelerator() -> Option<&'static dyn SweepAccelerator> {
    if !wanted() {
        return None;
    }
    SWEEP.get().map(|a| a.as_ref())
}

/// `SOLIDCRAFT_CAM_GPU=1` (or `on`, `true`, `yes`): accelerators are opt-in.
fn wanted() -> bool {
    std::env::var("SOLIDCRAFT_CAM_GPU").is_ok_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "on" | "true" | "yes"))
}

/// Installs the accelerator for this process; `false` when one is installed already.
pub fn set_cl_map_accelerator(a: Box<dyn ClMapAccelerator>) -> bool {
    ACCELERATOR.set(a).is_ok()
}

/// The installed accelerator, when `SOLIDCRAFT_CAM_GPU` asks for it.
pub(crate) fn accelerator() -> Option<&'static dyn ClMapAccelerator> {
    if !wanted() {
        return None;
    }
    ACCELERATOR.get().map(|a| a.as_ref())
}
