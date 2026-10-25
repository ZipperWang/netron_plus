// Hide the console window on Windows release builds.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use netron_core::Workspace;
use std::sync::Arc;
use tauri::ipc::Response;

struct AppState {
    ws: Arc<Workspace>,
}

/// Run blocking work off the IPC thread and turn errors into strings.
async fn blocking<T, F>(f: F) -> Result<T, String>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, String> + Send + 'static,
{
    tauri::async_runtime::spawn_blocking(f)
        .await
        .map_err(|e| e.to_string())?
}

/// Open a model file from disk; returns the summary JSON as raw bytes.
#[tauri::command]
async fn np_open_path(state: tauri::State<'_, AppState>, path: String) -> Result<Response, String> {
    let ws = state.ws.clone();
    let json = blocking(move || ws.open_path(&path).map_err(|e| format!("{:#}", e))).await?;
    Ok(Response::new(json.into_bytes()))
}

/// Generic API call; see `netron_core::api::Workspace::call`.
#[tauri::command]
async fn np_call(
    state: tauri::State<'_, AppState>,
    method: String,
    params: serde_json::Value,
) -> Result<Response, String> {
    let ws = state.ws.clone();
    let json = blocking(move || ws.call(&method, &params).map_err(|e| format!("{:#}", e))).await?;
    Ok(Response::new(json.into_bytes()))
}

/// Export a tensor straight to a file chosen by the user.
#[tauri::command]
async fn np_export(
    state: tauri::State<'_, AppState>,
    params: serde_json::Value,
    path: String,
) -> Result<(), String> {
    let ws = state.ws.clone();
    blocking(move || {
        let bytes = ws.export(&params).map_err(|e| format!("{:#}", e))?;
        std::fs::write(&path, bytes).map_err(|e| format!("Cannot write {}: {}", path, e))
    })
    .await
}

#[tauri::command]
async fn np_write_text(path: String, content: String) -> Result<(), String> {
    blocking(move || std::fs::write(&path, content).map_err(|e| format!("Cannot write {}: {}", path, e))).await
}

#[tauri::command]
async fn np_write_bytes(path: String, data: Vec<u8>) -> Result<(), String> {
    blocking(move || std::fs::write(&path, data).map_err(|e| format!("Cannot write {}: {}", path, e))).await
}

/// A model path passed on the command line (e.g. via "Open with").
#[tauri::command]
fn np_startup_file() -> Option<String> {
    std::env::args()
        .skip(1)
        .find(|a| !a.starts_with('-') && std::path::Path::new(a).is_file())
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .manage(AppState {
            ws: Arc::new(Workspace::new()),
        })
        .invoke_handler(tauri::generate_handler![
            np_open_path,
            np_call,
            np_export,
            np_write_text,
            np_write_bytes,
            np_startup_file
        ])
        .run(tauri::generate_context!())
        .expect("error while running Netron Plus");
}
