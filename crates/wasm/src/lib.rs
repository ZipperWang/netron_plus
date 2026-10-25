//! Minimal C ABI so the web build can drive the core library from a Web
//! Worker without any JS glue generator.
//!
//! Protocol: JS allocates input buffers with `np_alloc`, the entry points take
//! ownership of them, return a status (0 = ok, 1 = error) and leave the
//! response bytes (JSON or exported data, or the error text) in an output
//! buffer read through `np_out_ptr` / `np_out_len`.

use netron_core::{NoSource, Workspace};
use std::cell::RefCell;

thread_local! {
    static WS: Workspace = Workspace::new();
    static OUT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static PANIC: RefCell<String> = const { RefCell::new(String::new()) };
}

fn init_panic_hook() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        std::panic::set_hook(Box::new(|info| {
            let msg = info.to_string();
            PANIC.with(|p| {
                if let Ok(mut p) = p.try_borrow_mut() {
                    *p = msg;
                }
            });
        }));
    });
}

fn finish(result: Result<Vec<u8>, String>) -> u32 {
    let (status, bytes) = match result {
        Ok(b) => (0, b),
        Err(e) => (1, e.into_bytes()),
    };
    OUT.with(|o| *o.borrow_mut() = bytes);
    status
}

/// Take ownership of a buffer created by [`np_alloc`].
unsafe fn take(ptr: *mut u8, len: usize) -> Vec<u8> {
    if ptr.is_null() {
        return Vec::new();
    }
    Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len)).into_vec()
}

unsafe fn take_str(ptr: *mut u8, len: usize) -> String {
    String::from_utf8_lossy(&take(ptr, len)).into_owned()
}

#[no_mangle]
pub extern "C" fn np_alloc(len: usize) -> *mut u8 {
    let mut buf = vec![0u8; len].into_boxed_slice();
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

/// # Safety
/// `ptr`/`len` must come from [`np_alloc`] and not have been passed on.
#[no_mangle]
pub unsafe extern "C" fn np_dealloc(ptr: *mut u8, len: usize) {
    drop(take(ptr, len));
}

/// # Safety
/// Buffers must come from [`np_alloc`]; they are consumed.
#[no_mangle]
pub unsafe extern "C" fn np_open(
    name_ptr: *mut u8,
    name_len: usize,
    data_ptr: *mut u8,
    data_len: usize,
) -> u32 {
    init_panic_hook();
    let name = take_str(name_ptr, name_len);
    let data = take(data_ptr, data_len);
    let r = WS.with(|ws| ws.open_bytes(name, None, data, Box::new(NoSource)));
    finish(r.map(String::into_bytes).map_err(|e| format!("{:#}", e)))
}

/// # Safety
/// Buffers must come from [`np_alloc`]; they are consumed.
#[no_mangle]
pub unsafe extern "C" fn np_call(
    method_ptr: *mut u8,
    method_len: usize,
    params_ptr: *mut u8,
    params_len: usize,
) -> u32 {
    init_panic_hook();
    let method = take_str(method_ptr, method_len);
    let params = take_str(params_ptr, params_len);
    let r = WS.with(|ws| ws.call_str(&method, &params));
    finish(r.map(String::into_bytes).map_err(|e| format!("{:#}", e)))
}

/// # Safety
/// Buffers must come from [`np_alloc`]; they are consumed.
#[no_mangle]
pub unsafe extern "C" fn np_export(params_ptr: *mut u8, params_len: usize) -> u32 {
    init_panic_hook();
    let params = take_str(params_ptr, params_len);
    let r = WS.with(|ws| ws.export_str(&params));
    finish(r.map_err(|e| format!("{:#}", e)))
}

#[no_mangle]
pub extern "C" fn np_out_ptr() -> *const u8 {
    OUT.with(|o| o.borrow().as_ptr())
}

#[no_mangle]
pub extern "C" fn np_out_len() -> usize {
    OUT.with(|o| o.borrow().len())
}

#[no_mangle]
pub extern "C" fn np_panic_ptr() -> *const u8 {
    PANIC.with(|p| p.borrow().as_ptr())
}

#[no_mangle]
pub extern "C" fn np_panic_len() -> usize {
    PANIC.with(|p| p.borrow().len())
}
