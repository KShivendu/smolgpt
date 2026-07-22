//! Post-training INT8 quantization for storage: "quantize for storage,
//! dequantize for compute". This codebase's models (`Gpt`/`BigramLM`/`NgramLM`)
//! all store their trainable weights in a `candle_nn::VarMap` of f32 tensors,
//! and every forward pass is written in plain f32 candle ops. Rewriting the
//! forward pass to operate on int8 tensors directly (true int8 inference) is
//! out of scope for models this small (a few thousand parameters) — the win
//! there is compute/memory bandwidth, which doesn't matter here. What DOES
//! matter for this experiment series is disk footprint, so this module only
//! quantizes the on-disk representation:
//!
//! - `save_var_map_quantized` walks every named tensor in a `VarMap`, computes
//!   a per-tensor symmetric scale (`scale = max(abs(x)) / 127`), quantizes
//!   each value to `i8` (`round(x / scale)`, clamped to `[-127, 127]`), and
//!   writes a small custom binary file (see the format doc below).
//! - `load_into_var_map` is the mirror image: if the file at `path` starts
//!   with our magic, every tensor is dequantized (`x = i8_value * scale`)
//!   back to f32 and written into the caller's already-constructed `VarMap`
//!   (which must have been built with the same architecture, exactly like the
//!   existing `var_map.load(path)` safetensors path this replaces). If the
//!   file is NOT one of our quantized files (no magic match), this function
//!   falls back to the original `VarMap::load` (candle's safetensors format),
//!   so a single call site in each model's `load` works for both a regular
//!   `.bin` (safetensors, f32) and a quantized `.bin` (our custom format)
//!   completely transparently to the rest of the codebase (`--eval`,
//!   `--serve`, etc. don't need to know or care which kind of file they're
//!   loading).
//!
//! # File format
//!
//! Chose a small bespoke binary format over reusing candle's `VarMap::save`/
//! `load` (safetensors) because that format is f32-only — there's no
//! standard place to stash a per-tensor scale + i8 payload in it without
//! fighting the format. A per-tensor JSON sidecar was considered too, but a
//! single self-contained binary file is simpler to keep in sync with the
//! model `.bin` (one path, one file, no risk of the sidecar going missing).
//!
//! Layout (all integers little-endian):
//! ```text
//! magic:        4 bytes   b"SGQZ"
//! version:      u8        1
//! num_tensors:  u32
//! for each tensor:
//!   name_len:   u32
//!   name:       name_len bytes (UTF-8)
//!   num_dims:   u32
//!   dims:       num_dims x u64
//!   scale:      f32
//!   num_elems:  u64        (== product of dims; stored explicitly rather
//!                           than re-derived, so a corrupt/truncated dims
//!                           list is still caught by a length mismatch)
//!   data:       num_elems x i8 (1 byte each)
//! ```

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use candle_core::{Device, Tensor};
use candle_nn::VarMap;

use crate::error::{SmolError, SmolResult};

const MAGIC: &[u8; 4] = b"SGQZ";
const FORMAT_VERSION: u8 = 1;

/// One tensor's quantized payload: its shape (needed to reshape back to a
/// tensor on load), the single per-tensor scale factor, and the raw `i8`
/// values in row-major (the same order `Tensor::flatten_all().to_vec1()`
/// produces).
struct QuantizedTensor {
    shape: Vec<usize>,
    scale: f32,
    data: Vec<i8>,
}

/// Symmetric per-tensor quantization: `scale = max(abs(x)) / 127`, each value
/// mapped to `round(x / scale)` and clamped to `[-127, 127]`. An all-zero (or
/// empty) tensor gets `scale = 1.0` (arbitrary but harmless — every value
/// quantizes to 0 either way) to avoid a divide-by-zero.
fn quantize_f32(values: &[f32]) -> (f32, Vec<i8>) {
    let max_abs = values.iter().fold(0f32, |acc, &v| acc.max(v.abs()));
    let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
    let data = values
        .iter()
        .map(|&v| {
            let q = (v / scale).round();
            q.clamp(-127.0, 127.0) as i8
        })
        .collect();
    (scale, data)
}

/// Inverse of `quantize_f32`: `x = i8_value * scale`.
fn dequantize_f32(scale: f32, data: &[i8]) -> Vec<f32> {
    data.iter().map(|&q| q as f32 * scale).collect()
}

fn write_u8(w: &mut impl Write, v: u8) -> std::io::Result<()> {
    w.write_all(&[v])
}
fn write_u32(w: &mut impl Write, v: u32) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn write_u64(w: &mut impl Write, v: u64) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn write_f32(w: &mut impl Write, v: f32) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn read_u8(r: &mut impl Read) -> std::io::Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}
fn read_u32(r: &mut impl Read) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}
fn read_u64(r: &mut impl Read) -> std::io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}
fn read_f32(r: &mut impl Read) -> std::io::Result<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

/// `true` if `path` exists, is readable, and starts with our magic bytes.
/// Used by `load_into_var_map` to decide whether to parse our custom format
/// or fall back to candle's safetensors `VarMap::load`. Any I/O error (file
/// doesn't exist, too short to hold the magic, etc.) is treated as "not a
/// quantized file" rather than propagated — the caller's subsequent fallback
/// path (`var_map.load`) will surface the real error with its own message if
/// the file is genuinely unreadable/invalid.
fn looks_quantized(path: &Path) -> bool {
    let Ok(mut f) = File::open(path) else {
        return false;
    };
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf).is_ok() && &buf == MAGIC
}

/// Quantize every named tensor in `var_map` and write it to `path` in the
/// format documented on this module. Does NOT touch `var_map` itself (the
/// caller's in-memory model is left exactly as it was) — this only produces
/// an on-disk artifact, mirroring how `VarMap::save` doesn't mutate the map
/// either.
pub fn save_var_map_quantized(var_map: &VarMap, path: &Path) -> SmolResult<()> {
    let tensor_data = var_map.data().lock().map_err(|e| {
        SmolError::custom_error(&format!("save_var_map_quantized: VarMap lock poisoned: {e}"))
    })?;

    // Sort by name for a deterministic file layout (nice for diffing/tests;
    // HashMap iteration order is otherwise unspecified).
    let mut entries: Vec<(&String, &candle_core::Var)> = tensor_data.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));

    let mut quantized: Vec<(String, QuantizedTensor)> = Vec::with_capacity(entries.len());
    for (name, var) in entries {
        let tensor = var.as_tensor();
        let shape: Vec<usize> = tensor.dims().to_vec();
        let flat = tensor
            .flatten_all()
            .map_err(|e| SmolError::custom_error(&format!("flatten {name}: {e}")))?
            .to_dtype(candle_core::DType::F32)
            .map_err(|e| SmolError::custom_error(&format!("to_dtype {name}: {e}")))?
            .to_vec1::<f32>()
            .map_err(|e| SmolError::custom_error(&format!("to_vec1 {name}: {e}")))?;
        let (scale, data) = quantize_f32(&flat);
        quantized.push((name.clone(), QuantizedTensor { shape, scale, data }));
    }
    drop(tensor_data);

    let file = File::create(path).map_err(|e| {
        SmolError::custom_error(&format!("save_var_map_quantized: create {}: {e}", path.display()))
    })?;
    let mut w = BufWriter::new(file);

    (|| -> std::io::Result<()> {
        w.write_all(MAGIC)?;
        write_u8(&mut w, FORMAT_VERSION)?;
        write_u32(&mut w, quantized.len() as u32)?;
        for (name, qt) in &quantized {
            let name_bytes = name.as_bytes();
            write_u32(&mut w, name_bytes.len() as u32)?;
            w.write_all(name_bytes)?;
            write_u32(&mut w, qt.shape.len() as u32)?;
            for &d in &qt.shape {
                write_u64(&mut w, d as u64)?;
            }
            write_f32(&mut w, qt.scale)?;
            write_u64(&mut w, qt.data.len() as u64)?;
            // i8 -> u8 byte-for-byte (same bit pattern); read side reverses.
            let bytes: Vec<u8> = qt.data.iter().map(|&v| v as u8).collect();
            w.write_all(&bytes)?;
        }
        w.flush()
    })()
    .map_err(|e| SmolError::custom_error(&format!("save_var_map_quantized: write {}: {e}", path.display())))?;

    Ok(())
}

/// Read the quantized file at `path`, dequantize every tensor back to f32,
/// and return `(name, Tensor)` pairs on `device`, shaped per the stored
/// dims. Pure parsing/dequantization — does not touch any `VarMap`.
fn load_quantized_tensors(path: &Path, device: &Device) -> SmolResult<Vec<(String, Tensor)>> {
    let file = File::open(path).map_err(|e| {
        SmolError::custom_error(&format!("load_quantized_tensors: open {}: {e}", path.display()))
    })?;
    let mut r = BufReader::new(file);

    let result = (|| -> std::io::Result<Vec<(String, Vec<usize>, f32, Vec<i8>)>> {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "bad magic (not a smolgpt quantized file)",
            ));
        }
        let _version = read_u8(&mut r)?; // only version 1 exists so far
        let num_tensors = read_u32(&mut r)?;
        let mut out = Vec::with_capacity(num_tensors as usize);
        for _ in 0..num_tensors {
            let name_len = read_u32(&mut r)? as usize;
            let mut name_bytes = vec![0u8; name_len];
            r.read_exact(&mut name_bytes)?;
            let name = String::from_utf8(name_bytes).map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
            })?;
            let num_dims = read_u32(&mut r)? as usize;
            let mut shape = Vec::with_capacity(num_dims);
            for _ in 0..num_dims {
                shape.push(read_u64(&mut r)? as usize);
            }
            let scale = read_f32(&mut r)?;
            let num_elems = read_u64(&mut r)? as usize;
            let mut raw = vec![0u8; num_elems];
            r.read_exact(&mut raw)?;
            let data: Vec<i8> = raw.into_iter().map(|b| b as i8).collect();
            out.push((name, shape, scale, data));
        }
        Ok(out)
    })()
    .map_err(|e| SmolError::custom_error(&format!("load_quantized_tensors: {}: {e}", path.display())))?;

    let mut tensors = Vec::with_capacity(result.len());
    for (name, shape, scale, data) in result {
        let values = dequantize_f32(scale, &data);
        let tensor = Tensor::from_vec(values, shape, device)
            .map_err(|e| SmolError::custom_error(&format!("rebuild tensor {name}: {e}")))?;
        tensors.push((name, tensor));
    }
    Ok(tensors)
}

/// Populate `var_map`'s existing variables (already constructed with the
/// right architecture/shapes, e.g. via `Gpt::load`'s placeholder build) from
/// `path`. If `path` is one of our quantized files (magic match), every
/// tensor is dequantized and written in via `VarMap::set_one`. Otherwise
/// falls back to candle's own `VarMap::load` (safetensors, f32) — the
/// original behavior — so this function is a drop-in replacement for the
/// `var_map.load(path)?` call in every model's `load` constructor, and
/// callers don't need to know ahead of time which format `path` is in.
pub fn load_into_var_map(var_map: &mut VarMap, path: &Path, device: &Device) -> SmolResult<()> {
    if looks_quantized(path) {
        let tensors = load_quantized_tensors(path, device)?;
        for (name, tensor) in tensors {
            var_map
                .set_one(&name, &tensor)
                .map_err(|e| SmolError::custom_error(&format!("apply quantized tensor {name}: {e}")))?;
        }
        Ok(())
    } else {
        var_map.load(path).map_err(|e| e.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use candle_nn::{Init, VarBuilder};
    use temp_dir::TempDir;

    #[test]
    fn test_quantize_dequantize_round_trip_close_to_original() {
        let values: Vec<f32> = vec![-2.5, -1.0, -0.001, 0.0, 0.3, 1.0, 2.5, 0.9999];
        let (scale, data) = quantize_f32(&values);
        let recovered = dequantize_f32(scale, &data);
        assert_eq!(recovered.len(), values.len());
        // Max quantization error for a symmetric int8 scheme is scale/2; allow
        // a little slack for float rounding.
        let max_err = scale / 2.0 + 1e-6;
        for (orig, rec) in values.iter().zip(recovered.iter()) {
            assert!(
                (orig - rec).abs() <= max_err,
                "quantization error too large: orig={orig}, rec={rec}, max_err={max_err}"
            );
        }
    }

    #[test]
    fn test_quantize_all_zero_tensor_does_not_panic() {
        let values = vec![0.0f32; 8];
        let (scale, data) = quantize_f32(&values);
        assert!(scale > 0.0, "scale must be positive to avoid NaN on dequantize");
        let recovered = dequantize_f32(scale, &data);
        assert!(recovered.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_quantize_clamps_extreme_values() {
        // max_abs is 10.0 -> scale = 10/127. A value of exactly max_abs must
        // quantize to +-127, never overflow i8's range.
        let values = vec![10.0f32, -10.0f32, 0.0f32];
        let (scale, data) = quantize_f32(&values);
        assert_eq!(data[0], 127);
        assert_eq!(data[1], -127);
        assert!((scale - 10.0 / 127.0).abs() < 1e-6);
    }

    #[test]
    fn test_save_and_load_quantized_var_map_round_trip() {
        let device = Device::Cpu;
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("quant_test.bin");

        // Build a small VarMap with a couple of named tensors, like a real
        // model's constructor would.
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
        let a = vb
            .get_with_hints((4, 3), "weight_a", Init::Randn { mean: 0.0, stdev: 1.0 })
            .unwrap();
        let b = vb
            .get_with_hints((5,), "weight_b", Init::Const(0.0))
            .unwrap();
        // Give "weight_b" some non-trivial values to quantize.
        let b_values = Tensor::from_vec(vec![1.0f32, -2.0, 0.5, 3.25, -3.25], (5,), &device).unwrap();
        {
            let data = var_map.data().lock().unwrap();
            data.get("weight_b").unwrap().set(&b_values).unwrap();
        }

        save_var_map_quantized(&var_map, &path).unwrap();
        assert!(looks_quantized(&path), "saved file should start with our magic");

        // Load into a FRESH var_map built with the same shapes (mirrors how
        // Gpt::load builds placeholders before applying saved values).
        let mut loaded_var_map = VarMap::new();
        let vb2 = VarBuilder::from_varmap(&loaded_var_map, DType::F32, &device);
        let _ = vb2
            .get_with_hints((4, 3), "weight_a", Init::Const(0.0))
            .unwrap();
        let _ = vb2
            .get_with_hints((5,), "weight_b", Init::Const(0.0))
            .unwrap();

        load_into_var_map(&mut loaded_var_map, &path, &device).unwrap();

        let orig_a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let orig_b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let data = loaded_var_map.data().lock().unwrap();
        let loaded_a = data
            .get("weight_a")
            .unwrap()
            .as_tensor()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let loaded_b = data
            .get("weight_b")
            .unwrap()
            .as_tensor()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        // int8 quantization is lossy; check "close" rather than exact.
        for (o, l) in orig_a.iter().zip(loaded_a.iter()) {
            assert!((o - l).abs() < 0.05, "weight_a drifted too much: {o} vs {l}");
        }
        for (o, l) in orig_b.iter().zip(loaded_b.iter()) {
            assert!((o - l).abs() < 0.05, "weight_b drifted too much: {o} vs {l}");
        }
    }

    #[test]
    fn test_looks_quantized_false_for_non_quantized_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("not_quantized.bin");
        std::fs::write(&path, b"not a quantized file at all").unwrap();
        assert!(!looks_quantized(&path));
    }
}
