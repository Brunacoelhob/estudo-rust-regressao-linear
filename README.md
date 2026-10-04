# Regressão Linear em Rust

[![CI](https://github.com/Brunacoelhob/estudo-rust-regressao-linear/actions/workflows/ci.yml/badge.svg)](https://github.com/Brunacoelhob/estudo-rust-regressao-linear/actions/workflows/ci.yml)
![Rust](https://img.shields.io/badge/rust-2021-orange)
![License](https://img.shields.io/badge/license-MIT-blue)

Regressão linear simples (mínimos quadrados) **sem dependências externas**: biblioteca reutilizável + CLI que lê um CSV.

## Destaques

- **Numericamente estável**: usa somas centradas na média (a fórmula "de livro" perde precisão com `x` grandes, como timestamps).
- **Erros tipados** (`RegressionError`), sem *panic*: tamanhos diferentes, dados vazios, menos de 2 pontos, `NaN`/infinito e `x` constante.
- **Casos de borda tratados** no R² (`y` constante).
- API de alto nível: `LinearModel::fit(&x, &y)` devolve coeficientes, R² e MSE.
- 12 testes de unidade + doctest; CI com `fmt`, `clippy -D warnings` e testes.

## Uso como biblioteca

```rust
use rust_linear_regression::regression::linear::LinearModel;

let modelo = LinearModel::fit(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0])?;
println!("y = {} + {}·x  (R² = {})", modelo.intercept, modelo.slope, modelo.r_squared);
println!("previsão para x = 10: {}", modelo.predict_one(10.0));
```

Funções avulsas também estão disponíveis: `linear_regression`, `r_squared`, `mean_squared_error` e `predict`.

## Uso como programa

```bash
cargo run                                   # dados de exemplo
cargo run -- examples/exemplo.csv           # lê um CSV "x,y" (cabeçalho opcional, # comenta)
cargo run -- examples/exemplo.csv 6 7 8     # e prevê y para x = 6, 7 e 8
```

## Desenvolvimento

```bash
cargo test
cargo fmt --check
cargo clippy --all-targets -- -D warnings
```

Requer o [Rust](https://www.rust-lang.org/tools/install) estável.

## Estrutura

```
src/
├── lib.rs                  # biblioteca (com doctest)
├── main.rs                 # CLI
└── regression/linear.rs    # algoritmo, métricas, erros e testes
examples/exemplo.csv
```

## Licença

[MIT](LICENSE)
