//! Independent work spread over the cores: the rows of a grid (height maps, drop-cutter
//! queries), passes and Z levels, on Rayon's pool in desktop builds, serially in the browser build, which has no
//! threads. Results are the same either way: each row is written by exactly one call.

/// Calls `f(j, row)` for every row `j` of `data` (rows of `width` items, the last one possibly
/// shorter).
pub(crate) fn rows<T: Send>(data: &mut [T], width: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    if width == 0 {
        return;
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        data.par_chunks_mut(width).enumerate().for_each(|(j, row)| f(j, row));
    }
    #[cfg(target_arch = "wasm32")]
    for (j, row) in data.chunks_mut(width).enumerate() {
        f(j, row);
    }
}

/// `f` over `items`, results in item order: on Rayon's pool in desktop builds, serially in the
/// browser build. For independent items (passes, Z levels), so the output is the same either way.
pub(crate) fn map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        let f = &f;
        items.par_iter().map(f).collect()
    }
    #[cfg(target_arch = "wasm32")]
    {
        items.iter().map(f).collect()
    }
}
