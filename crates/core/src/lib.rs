//! Core of Netron Plus: ONNX parsing, shape inference, statistics, health
//! checks and graph layout. The crate has no platform dependencies so it is
//! shared by the Tauri desktop app and the WebAssembly web build.

pub mod analysis;
pub mod api;
pub mod checks;
pub mod cost;
pub mod dtype;
pub mod format;
pub mod infer;
pub mod layout;
pub mod model;
pub mod ops;
pub mod proto;
pub mod shape;
pub mod summary;
pub mod tensor;
pub mod view;

pub use api::Workspace;
pub use tensor::{DataSource, FileSource, NoSource};
