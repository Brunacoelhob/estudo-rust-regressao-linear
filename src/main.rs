//! CLI: ajusta uma reta a partir de um CSV (`x,y` por linha) ou, sem argumentos, usa dados de exemplo.
//!
//! Uso:
//!   rust-linear-regression                      # dados de exemplo
//!   rust-linear-regression dados.csv            # lê o arquivo
//!   rust-linear-regression dados.csv 6 7 8      # e prevê y para x = 6, 7, 8

use std::{env, fs, process::ExitCode};

use rust_linear_regression::regression::linear::LinearModel;

fn ler_csv(caminho: &str) -> Result<(Vec<f64>, Vec<f64>), String> {
    let texto = fs::read_to_string(caminho)
        .map_err(|e| format!("não foi possível ler '{caminho}': {e}"))?;
    let (mut x, mut y) = (Vec::new(), Vec::new());

    for (i, linha) in texto.lines().enumerate() {
        let linha = linha.trim();
        if linha.is_empty() || linha.starts_with('#') {
            continue;
        }
        let campos: Vec<&str> = linha.split(',').map(str::trim).collect();
        let [a, b] = campos[..] else {
            return Err(format!(
                "linha {}: esperado 'x,y', encontrado '{linha}'",
                i + 1
            ));
        };
        match (a.parse::<f64>(), b.parse::<f64>()) {
            (Ok(a), Ok(b)) => {
                x.push(a);
                y.push(b);
            }
            // Permite um cabeçalho (ex.: "x,y") apenas na primeira linha do arquivo.
            _ if i == 0 => continue,
            _ => {
                return Err(format!(
                    "linha {}: valores numéricos inválidos em '{linha}'",
                    i + 1
                ))
            }
        }
    }
    Ok((x, y))
}

fn executar() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();

    let (x, y) = match args.first() {
        Some(caminho) => ler_csv(caminho)?,
        None => (
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![2.0, 4.0, 6.0, 8.0, 10.0],
        ),
    };

    let modelo = LinearModel::fit(&x, &y).map_err(|e| e.to_string())?;
    println!("Pontos:      {}", x.len());
    println!("Intercepto:  {}", modelo.intercept);
    println!("Inclinação:  {}", modelo.slope);
    println!("R²:          {}", modelo.r_squared);
    println!("MSE:         {}", modelo.mse);

    let novos: Vec<f64> = if args.len() > 1 {
        args[1..]
            .iter()
            .map(|a| {
                a.parse::<f64>()
                    .map_err(|_| format!("valor inválido para previsão: '{a}'"))
            })
            .collect::<Result<_, _>>()?
    } else if args.is_empty() {
        vec![6.0, 7.0, 8.0]
    } else {
        Vec::new()
    };

    for (xi, yi) in novos.iter().zip(modelo.predict(&novos)) {
        println!("Previsão: x = {xi} -> y = {yi}");
    }
    Ok(())
}

fn main() -> ExitCode {
    match executar() {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("erro: {msg}");
            ExitCode::FAILURE
        }
    }
}
