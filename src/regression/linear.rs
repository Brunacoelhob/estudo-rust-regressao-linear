//! Regressão linear simples (y = intercepto + inclinação · x) pelo método dos mínimos quadrados.
//!
//! Os coeficientes são calculados com somas **centradas na média**, que é numericamente
//! mais estável que a forma clássica `n·Σx² − (Σx)²`: esta perde precisão quando os valores
//! de `x` são grandes e próximos entre si (ex.: timestamps, anos).

use std::error::Error;
use std::fmt;

/// Erros possíveis ao ajustar ou avaliar o modelo.
#[derive(Debug, Clone, PartialEq)]
pub enum RegressionError {
    /// `x` e `y` têm tamanhos diferentes.
    LengthMismatch { x: usize, y: usize },
    /// Não há dados.
    Empty,
    /// São necessários ao menos 2 pontos para traçar uma reta.
    NotEnoughPoints(usize),
    /// Algum valor é `NaN` ou infinito.
    NonFinite,
    /// Todos os `x` são iguais: a reta não é definida (inclinação infinita).
    ZeroVariance,
}

impl fmt::Display for RegressionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch { x, y } => {
                write!(f, "x e y devem ter o mesmo tamanho (x tem {x}, y tem {y})")
            }
            Self::Empty => write!(f, "os dados não podem estar vazios"),
            Self::NotEnoughPoints(n) => {
                write!(f, "são necessários ao menos 2 pontos (recebido: {n})")
            }
            Self::NonFinite => write!(f, "os dados contêm valores não finitos (NaN ou infinito)"),
            Self::ZeroVariance => write!(f, "todos os valores de x são iguais; não há reta única"),
        }
    }
}

impl Error for RegressionError {}

fn validar(x: &[f64], y: &[f64]) -> Result<(), RegressionError> {
    if x.len() != y.len() {
        return Err(RegressionError::LengthMismatch {
            x: x.len(),
            y: y.len(),
        });
    }
    if x.is_empty() {
        return Err(RegressionError::Empty);
    }
    if x.len() < 2 {
        return Err(RegressionError::NotEnoughPoints(x.len()));
    }
    if x.iter().chain(y).any(|v| !v.is_finite()) {
        return Err(RegressionError::NonFinite);
    }
    Ok(())
}

fn media(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

/// Calcula `(intercepto, inclinação)` da reta de mínimos quadrados.
pub fn linear_regression(x: &[f64], y: &[f64]) -> Result<(f64, f64), RegressionError> {
    validar(x, y)?;

    let (mx, my) = (media(x), media(y));
    let sxx: f64 = x.iter().map(|xi| (xi - mx).powi(2)).sum();
    if sxx == 0.0 {
        return Err(RegressionError::ZeroVariance);
    }
    let sxy: f64 = x.iter().zip(y).map(|(xi, yi)| (xi - mx) * (yi - my)).sum();

    let slope = sxy / sxx;
    Ok((my - slope * mx, slope))
}

/// Coeficiente de determinação R². Vale 1,0 para um ajuste perfeito.
///
/// Se `y` é constante (variância zero) e o modelo a prevê exatamente, devolve 1,0;
/// se é constante mas o modelo erra, o R² não é definido e devolve `f64::NEG_INFINITY`.
pub fn r_squared(x: &[f64], y: &[f64], intercept: f64, slope: f64) -> Result<f64, RegressionError> {
    validar(x, y)?;

    let my = media(y);
    let ss_total: f64 = y.iter().map(|yi| (yi - my).powi(2)).sum();
    let ss_residual: f64 = residuos_quadrados(x, y, intercept, slope).sum();

    if ss_total == 0.0 {
        return Ok(if ss_residual == 0.0 {
            1.0
        } else {
            f64::NEG_INFINITY
        });
    }
    Ok(1.0 - ss_residual / ss_total)
}

/// Erro quadrático médio (MSE).
pub fn mean_squared_error(
    x: &[f64],
    y: &[f64],
    intercept: f64,
    slope: f64,
) -> Result<f64, RegressionError> {
    validar(x, y)?;
    Ok(residuos_quadrados(x, y, intercept, slope).sum::<f64>() / x.len() as f64)
}

/// Previsões `intercepto + inclinação · x` para cada valor de `x`.
pub fn predict(x: &[f64], intercept: f64, slope: f64) -> Vec<f64> {
    x.iter().map(|&xi| intercept + slope * xi).collect()
}

fn residuos_quadrados<'a>(
    x: &'a [f64],
    y: &'a [f64],
    intercept: f64,
    slope: f64,
) -> impl Iterator<Item = f64> + 'a {
    x.iter()
        .zip(y)
        .map(move |(xi, yi)| (yi - (intercept + slope * xi)).powi(2))
}

/// Modelo ajustado: reúne os coeficientes e as métricas de qualidade.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearModel {
    pub intercept: f64,
    pub slope: f64,
    pub r_squared: f64,
    pub mse: f64,
}

impl LinearModel {
    /// Ajusta o modelo e já calcula R² e MSE sobre os dados de treino.
    pub fn fit(x: &[f64], y: &[f64]) -> Result<Self, RegressionError> {
        let (intercept, slope) = linear_regression(x, y)?;
        Ok(Self {
            intercept,
            slope,
            r_squared: r_squared(x, y, intercept, slope)?,
            mse: mean_squared_error(x, y, intercept, slope)?,
        })
    }

    pub fn predict_one(&self, x: f64) -> f64 {
        self.intercept + self.slope * x
    }

    pub fn predict(&self, x: &[f64]) -> Vec<f64> {
        predict(x, self.intercept, self.slope)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const X: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    const Y: [f64; 5] = [2.0, 4.0, 6.0, 8.0, 10.0];

    fn quase(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn reta_perfeita() {
        let (b0, b1) = linear_regression(&X, &Y).unwrap();
        assert!(quase(b0, 0.0) && quase(b1, 2.0));
    }

    #[test]
    fn dados_com_ruido_conhecido() {
        // Referência feita à mão: mx = 2,5; my = 5,0; Sxx = 5; Sxy = 9,9
        // -> inclinação 1,98 e intercepto 5,0 - 1,98 * 2,5 = 0,05.
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [2.0, 4.1, 5.9, 8.0];
        let (b0, b1) = linear_regression(&x, &y).unwrap();
        assert!(quase(b1, 1.98));
        assert!(quase(b0, 0.05));
        let r2 = r_squared(&x, &y, b0, b1).unwrap();
        assert!(r2 > 0.99 && r2 < 1.0);
    }

    #[test]
    fn r2_e_mse_de_ajuste_perfeito() {
        let (b0, b1) = linear_regression(&X, &Y).unwrap();
        assert!(quase(r_squared(&X, &Y, b0, b1).unwrap(), 1.0));
        assert!(quase(mean_squared_error(&X, &Y, b0, b1).unwrap(), 0.0));
    }

    #[test]
    fn estabilidade_numerica_com_x_grandes() {
        // Anos/timestamps: a fórmula ingênua perde precisão aqui.
        let x: Vec<f64> = (0..10).map(|i| 1.7e9 + i as f64).collect();
        let y: Vec<f64> = x.iter().map(|xi| 3.0 * xi + 5.0).collect();
        let (_, b1) = linear_regression(&x, &y).unwrap();
        assert!((b1 - 3.0).abs() < 1e-6);
    }

    #[test]
    fn previsao() {
        assert_eq!(predict(&X, 0.0, 2.0), Y.to_vec());
    }

    #[test]
    fn modelo_agrupa_tudo() {
        let m = LinearModel::fit(&X, &Y).unwrap();
        assert!(quase(m.slope, 2.0) && quase(m.r_squared, 1.0));
        assert!(quase(m.predict_one(6.0), 12.0));
        assert_eq!(m.predict(&[6.0, 7.0]).len(), 2);
    }

    #[test]
    fn tamanhos_diferentes() {
        assert_eq!(
            linear_regression(&[1.0, 2.0], &[3.0]),
            Err(RegressionError::LengthMismatch { x: 2, y: 1 })
        );
    }

    #[test]
    fn vazio_e_ponto_unico() {
        assert_eq!(linear_regression(&[], &[]), Err(RegressionError::Empty));
        assert_eq!(
            linear_regression(&[1.0], &[1.0]),
            Err(RegressionError::NotEnoughPoints(1))
        );
    }

    #[test]
    fn rejeita_nan_e_infinito() {
        assert_eq!(
            linear_regression(&[1.0, f64::NAN], &[1.0, 2.0]),
            Err(RegressionError::NonFinite)
        );
        assert_eq!(
            linear_regression(&[1.0, 2.0], &[1.0, f64::INFINITY]),
            Err(RegressionError::NonFinite)
        );
    }

    #[test]
    fn x_constante_nao_define_reta() {
        assert_eq!(
            linear_regression(&[2.0, 2.0, 2.0], &[1.0, 2.0, 3.0]),
            Err(RegressionError::ZeroVariance)
        );
    }

    #[test]
    fn y_constante_tem_inclinacao_zero_e_r2_definido() {
        let y = [4.0, 4.0, 4.0, 4.0, 4.0];
        let (b0, b1) = linear_regression(&X, &y).unwrap();
        assert!(quase(b1, 0.0) && quase(b0, 4.0));
        assert!(quase(r_squared(&X, &y, b0, b1).unwrap(), 1.0));
        assert_eq!(r_squared(&X, &y, 0.0, 1.0).unwrap(), f64::NEG_INFINITY);
    }

    #[test]
    fn mensagens_de_erro_legiveis() {
        let msg = RegressionError::LengthMismatch { x: 3, y: 2 }.to_string();
        assert!(msg.contains("x tem 3") && msg.contains("y tem 2"));
    }
}
