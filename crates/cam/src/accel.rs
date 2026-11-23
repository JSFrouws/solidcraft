//! Optional accelerators for the cutter-location map and the stock sweep (for example a GPU), installed by a
//! higher layer (the apps install the GPU one from `solidcraft-gpu`). The CAM crate stays
//! geometry only: without an accelerator, or when it declines, the map is computed on the CPU.
//! `SOLIDCRAFT_CAM_GPU=0` (or `off`) turns the accelerator off for a run.

use std::sync::OnceLock;

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
}

/// Sweeps a 3-axis toolpath through a Z-map stock: `cuts` lower the stock under the tool (in
/// order; their order does not change the heights), and each of `rapids` — `(cuts before it,
/// from, to)` — is checked against the stock as the cuts before it left it, with the stock's
/// `tol`. Lowers `stock` as [`StockMap::cut`] does and returns, per rapid, whether it passes
/// through material ([`StockMap::rapid_hits`]). `None` declines (the CPU sweeps instead).
pub trait SweepAccelerator: Send + Sync {
    fn sweep(&self, stock: &mut StockMap, tool: &Tool, cuts: &[(Vec3, Vec3)], rapids: &[(usize, Vec3, Vec3)], tol: f64) -> Option<Vec<bool>>;
}

static ACCELERATOR: OnceLock<Box<dyn ClMapAccelerator>> = OnceLock::new();
static SWEEP: OnceLock<Box<dyn SweepAccelerator>> = OnceLock::new();

/// Installs the stock-sweep accelerator for this process; `false` when one is installed already.
pub fn set_sweep_accelerator(a: Box<dyn SweepAccelerator>) -> bool {
    SWEEP.set(a).is_ok()
}

/// The installed sweep accelerator, unless `SOLIDCRAFT_CAM_GPU` turns accelerators off.
pub(crate) fn sweep_accelerator() -> Option<&'static dyn SweepAccelerator> {
    if off() {
        return None;
    }
    SWEEP.get().map(|a| a.as_ref())
}

fn off() -> bool {
    std::env::var("SOLIDCRAFT_CAM_GPU").is_ok_and(|v| matches!(v.trim(), "0" | "off" | "false" | "no"))
}

/// Installs the accelerator for this process; `false` when one is installed already.
pub fn set_cl_map_accelerator(a: Box<dyn ClMapAccelerator>) -> bool {
    ACCELERATOR.set(a).is_ok()
}

/// The installed accelerator, unless `SOLIDCRAFT_CAM_GPU` turns it off.
pub(crate) fn accelerator() -> Option<&'static dyn ClMapAccelerator> {
    if off() {
        return None;
    }
    ACCELERATOR.get().map(|a| a.as_ref())
}
