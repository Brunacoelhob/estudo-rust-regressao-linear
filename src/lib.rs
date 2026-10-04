//! Regressão linear simples, sem dependências externas.
//!
//! ```
//! use rust_linear_regression::regression::linear::LinearModel;
//!
//! let modelo = LinearModel::fit(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]).unwrap();
//! assert!((modelo.slope - 2.0).abs() < 1e-9);
//! assert!((modelo.predict_one(10.0) - 20.0).abs() < 1e-9);
//! ```

pub mod regression;
