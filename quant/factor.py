"""Tradex Factor Analysis Module.

Provides factor exposure analysis and multi-factor modeling for equity returns,
following the Fama-French style approach with regime-aware factor adjustment.

팩터 분석 모듈 - 주식 수익률을 설명하는 팩터 분석 및 모델링을 제공하며,
Fama-French 스타일의 멀티팩터 모델과 레짐 기반 팩터 조정을 지원합니다.

Features:
    - Factor exposure analysis (beta, t-stat, p-value per factor)
    - Factor return attribution and decomposition
    - Multi-factor model fitting and risk decomposition (Fama-French style)
    - Rolling factor analysis over configurable windows
    - Regime-aware factor adjustment via RegimeForecaster integration
    - Learned-pattern-based factor weight tuning

Usage:
    from tradex.quant.factor import FactorAnalyzer, FactorModel

    analyzer = FactorAnalyzer()
    analyzer.setFactors(factor_returns_df)
    result = analyzer.analyze(portfolio_returns)
    print(result.summary())

    model = FactorModel()
    model.fit(asset_returns_df, factor_returns_df)
    risk = model.riskDecomposition()
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from enum import Enum
import pandas as pd
import numpy as np
from scipy import stats


class Factor(Enum):
    """Standard equity risk factors used in multi-factor models.

    표준 주식 리스크 팩터 열거형 (멀티팩터 모델에서 사용).

    Attributes:
        MARKET: Market risk factor (시장 팩터).
        SIZE: Size factor, small-minus-big (규모 팩터).
        VALUE: Value factor, high-minus-low book-to-market (가치 팩터).
        MOMENTUM: Momentum factor (모멘텀 팩터).
        QUALITY: Quality factor (퀄리티 팩터).
        VOLATILITY: Low-volatility factor (저변동성 팩터).
        LIQUIDITY: Liquidity factor (유동성 팩터).
    """
    MARKET = "market"
    SIZE = "size"
    VALUE = "value"
    MOMENTUM = "momentum"
    QUALITY = "quality"
    VOLATILITY = "volatility"
    LIQUIDITY = "liquidity"


@dataclass
class FactorExposure:
    """Single factor exposure result from regression analysis.

    회귀 분석에서 산출된 단일 팩터 노출도 결과.

    Attributes:
        factor: Factor name (팩터 이름).
        beta: Factor loading / regression coefficient (팩터 베타 계수).
        tStat: t-statistic for the beta estimate (베타 추정치의 t-통계량).
        pValue: p-value for statistical significance (통계적 유의성 p-값).
        contribution: Annualized return contribution of this factor (연율 수익 기여도).
    """
    factor: str
    beta: float
    tStat: float
    pValue: float
    contribution: float

    def isSignificant(self, alpha: float = 0.05) -> bool:
        """Check whether the factor exposure is statistically significant.

        Args:
            alpha: Significance level (default 0.05). 유의 수준.

        Returns:
            True if pValue is below the significance level.
        """
        return self.pValue < alpha


@dataclass
class FactorResult:
    """Complete result of a factor regression analysis.

    팩터 회귀 분석의 전체 결과를 담는 데이터 클래스.

    Attributes:
        exposures: Mapping of factor name to FactorExposure (팩터별 노출도 딕셔너리).
        alpha: Daily alpha (intercept) from regression (일별 알파).
        alphaTStat: t-statistic for the alpha estimate (알파의 t-통계량).
        rSquared: R-squared of the regression (결정 계수).
        adjRSquared: Adjusted R-squared (수정 결정 계수).
        residualVol: Daily residual volatility (일별 잔차 변동성).
        factorContribution: Total annualized factor return contribution (연율 팩터 기여 수익).
        specificReturn: Annualized alpha / specific return (연율 고유 수익).
    """
    exposures: Dict[str, FactorExposure]
    alpha: float
    alphaTStat: float
    rSquared: float
    adjRSquared: float
    residualVol: float
    factorContribution: float
    specificReturn: float

    def summary(self) -> str:
        """Generate a human-readable summary of the factor analysis result.

        Returns:
            Formatted multi-line string with alpha, R-squared, and per-factor exposures.
        """
        lines = [
            "=== 팩터 분석 결과 ===",
            f"알파 (연율): {self.alpha * 252:.2%} (t={self.alphaTStat:.2f})",
            f"R²: {self.rSquared:.2%}",
            f"조정 R²: {self.adjRSquared:.2%}",
            f"잔차 변동성: {self.residualVol * np.sqrt(252):.2%}",
            "",
            "팩터 노출도:",
            f"{'팩터':<15} {'베타':>10} {'t-stat':>10} {'기여도':>10}",
            "-" * 50,
        ]

        for name, exp in self.exposures.items():
            sig = "*" if exp.isSignificant() else ""
            lines.append(
                f"{name:<15} {exp.beta:>10.3f} {exp.tStat:>9.2f}{sig} {exp.contribution:>9.2%}"
            )

        lines.extend([
            "-" * 50,
            f"팩터 기여: {self.factorContribution:.2%}",
            f"고유 수익: {self.specificReturn:.2%}",
        ])

        return "\n".join(lines)


class FactorAnalyzer:
    """Factor exposure analyzer for portfolios and strategies.

    Performs OLS regression of asset/portfolio returns against a set of factor
    returns to estimate factor loadings, alpha, and goodness-of-fit metrics.

    포트폴리오/전략의 팩터 노출도를 분석합니다. OLS 회귀를 통해 팩터 베타,
    알파, R-squared 등을 산출합니다.

    Attributes:
        riskFreeRate: Annualized risk-free rate (연율 무위험 이자율).
        factorReturns: DataFrame of daily factor returns (일별 팩터 수익률).
        factorNames: List of factor column names (팩터 이름 목록).

    Example:
        >>> analyzer = FactorAnalyzer(riskFreeRate=0.03)
        >>> analyzer.setFactors(factor_returns_df)
        >>> result = analyzer.analyze(portfolio_returns)
        >>> print(result.summary())
        >>> rolling = analyzer.rollingAnalysis(portfolio_returns, window=60)
    """

    def __init__(self, riskFreeRate: float = 0.02):
        """Initialize the factor analyzer.

        Args:
            riskFreeRate: Annualized risk-free rate (default 0.02).
                          연율 무위험 이자율.
        """
        self.riskFreeRate = riskFreeRate
        self.factorReturns: pd.DataFrame = None
        self.factorNames: List[str] = []

    def setFactors(self, factorReturns: pd.DataFrame) -> 'FactorAnalyzer':
        """Set the factor return data for subsequent analyses.

        Args:
            factorReturns: DataFrame of daily factor returns.
                           Columns should be factor names, e.g. ['market', 'size', 'value'].
                           팩터별 일별 수익률 DataFrame.

        Returns:
            Self for method chaining.
        """
        self.factorReturns = factorReturns.dropna()
        self.factorNames = list(factorReturns.columns)
        return self

    def createFactorsFromPrices(
        self,
        priceData: pd.DataFrame,
        marketCol: str = None,
        sizeCol: str = None,
        bookToMarketCol: str = None,
    ) -> pd.DataFrame:
        """Construct factor return series from raw price data.

        Derives momentum and volatility factors from close prices, and optionally
        a market factor from the specified index column.

        Args:
            priceData: DataFrame with OHLCV and optional fundamental data.
                       OHLCV 및 재무 데이터 DataFrame.
            marketCol: Column name for the market index (시장 지수 컬럼).
            sizeCol: Column name for market cap, used for size factor (시가총액 컬럼).
            bookToMarketCol: Column name for book-to-market ratio (장부가/시가 비율 컬럼).

        Returns:
            DataFrame of computed factor returns (산출된 팩터 수익률 DataFrame).
        """
        factors = pd.DataFrame(index=priceData.index)

        if 'close' in priceData.columns:
            returns = priceData['close'].pct_change()

            factors['momentum'] = returns.rolling(252).mean() - returns.rolling(21).mean()

            factors['volatility'] = -returns.rolling(21).std()

        if marketCol and marketCol in priceData.columns:
            factors['market'] = priceData[marketCol].pct_change()

        return factors.dropna()

    def analyze(
        self,
        returns: pd.Series,
        factors: pd.DataFrame = None
    ) -> FactorResult:
        """Run OLS factor regression on the given return series.

        Args:
            returns: Daily return series of a portfolio or strategy.
                     포트폴리오/전략의 일별 수익률 시리즈.
            factors: Factor returns DataFrame. If None, uses the previously set factors.
                     팩터 수익률 DataFrame (None이면 기존 설정 사용).

        Returns:
            FactorResult containing exposures, alpha, R-squared, and attribution.

        Raises:
            ValueError: If no factor returns have been set or provided.
        """
        if factors is None:
            factors = self.factorReturns

        if factors is None:
            raise ValueError("팩터 수익률을 설정하세요")

        commonIdx = returns.index.intersection(factors.index)
        y = returns.loc[commonIdx].values
        X = factors.loc[commonIdx].values

        X = np.column_stack([np.ones(len(y)), X])

        try:
            beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            beta = np.zeros(X.shape[1])
            residuals = np.array([np.sum((y - X @ beta) ** 2)])

        alpha = beta[0]
        factorBetas = beta[1:]

        yPred = X @ beta
        ssRes = np.sum((y - yPred) ** 2)
        ssTot = np.sum((y - np.mean(y)) ** 2)
        rSquared = 1 - ssRes / ssTot if ssTot > 0 else 0

        n = len(y)
        k = len(factorBetas)
        adjRSquared = 1 - (1 - rSquared) * (n - 1) / (n - k - 1) if n > k + 1 else rSquared

        residualVol = np.std(y - yPred)

        mse = ssRes / (n - k - 1) if n > k + 1 else ssRes / n
        varBeta = mse * np.linalg.inv(X.T @ X).diagonal()
        seBeta = np.sqrt(varBeta)
        tStats = beta / seBeta

        exposures = {}
        totalFactorContrib = 0

        for i, name in enumerate(self.factorNames):
            factorReturn = factors[name].loc[commonIdx].mean() * 252
            contribution = factorBetas[i] * factorReturn

            pValue = 2 * (1 - stats.t.cdf(abs(tStats[i + 1]), n - k - 1))

            exposures[name] = FactorExposure(
                factor=name,
                beta=factorBetas[i],
                tStat=tStats[i + 1],
                pValue=pValue,
                contribution=contribution,
            )
            totalFactorContrib += contribution

        alphaPValue = 2 * (1 - stats.t.cdf(abs(tStats[0]), n - k - 1))

        return FactorResult(
            exposures=exposures,
            alpha=alpha,
            alphaTStat=tStats[0],
            rSquared=rSquared,
            adjRSquared=adjRSquared,
            residualVol=residualVol,
            factorContribution=totalFactorContrib,
            specificReturn=alpha * 252,
        )

    def rollingAnalysis(
        self,
        returns: pd.Series,
        window: int = 60,
        factors: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Perform rolling-window factor analysis.

        Args:
            returns: Daily return series (일별 수익률 시리즈).
            window: Rolling window size in trading days (롤링 윈도우 크기, 거래일 기준).
            factors: Factor returns DataFrame. If None, uses the previously set factors.
                     팩터 수익률 DataFrame (None이면 기존 설정 사용).

        Returns:
            DataFrame with date-indexed rolling alpha, R-squared, and per-factor
            beta/t-stat columns (롤링 팩터 노출도 DataFrame).
        """
        if factors is None:
            factors = self.factorReturns

        results = []

        for i in range(window, len(returns)):
            startIdx = i - window
            endIdx = i

            windowReturns = returns.iloc[startIdx:endIdx]

            result = self.analyze(windowReturns, factors)

            row = {
                'date': returns.index[i],
                'alpha': result.alpha,
                'rSquared': result.rSquared,
            }

            for name, exp in result.exposures.items():
                row[f'{name}_beta'] = exp.beta
                row[f'{name}_tstat'] = exp.tStat

            results.append(row)

        return pd.DataFrame(results)

    def factorAttribution(
        self,
        returns: pd.Series,
        factors: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Decompose daily returns into factor contributions.

        Args:
            returns: Daily return series (일별 수익률 시리즈).
            factors: Factor returns DataFrame. If None, uses the previously set factors.
                     팩터 수익률 DataFrame (None이면 기존 설정 사용).

        Returns:
            DataFrame with columns for total return, alpha, each factor's contribution,
            and residual (일별 팩터 기여도 DataFrame).
        """
        if factors is None:
            factors = self.factorReturns

        result = self.analyze(returns, factors)

        commonIdx = returns.index.intersection(factors.index)
        attribution = pd.DataFrame(index=commonIdx)

        attribution['total'] = returns.loc[commonIdx]
        attribution['alpha'] = result.alpha

        for name, exp in result.exposures.items():
            attribution[name] = exp.beta * factors[name].loc[commonIdx]

        attribution['residual'] = (
            attribution['total']
            - attribution['alpha']
            - attribution[[name for name in result.exposures.keys()]].sum(axis=1)
        )

        return attribution


class FactorModel:
    """Multi-factor model for expected return estimation and risk decomposition.

    Fits an OLS regression per asset to estimate factor betas, then provides
    expected returns based on factor premiums and decomposes total risk into
    factor risk and specific (idiosyncratic) risk.

    멀티팩터 모델 - 자산별 OLS 회귀를 통해 팩터 베타를 추정하고,
    기대 수익률 산출 및 리스크 분해를 수행합니다.

    Attributes:
        factorBetas: DataFrame of asset-by-factor loadings (자산별 팩터 베타).
        factorCov: Annualized factor covariance matrix (연율 팩터 공분산 행렬).
        residualVol: Per-asset residual (idiosyncratic) volatility (자산별 잔차 변동성).
        factorNames: List of factor names (팩터 이름 목록).
        assetNames: List of asset names (자산 이름 목록).

    Example:
        >>> model = FactorModel()
        >>> model.fit(asset_returns_df, factor_returns_df)
        >>> expected = model.expectedReturn({'market': 0.08, 'size': 0.03})
        >>> risk = model.riskDecomposition()
    """

    def __init__(self):
        self.factorBetas: pd.DataFrame = None
        self.factorCov: pd.DataFrame = None
        self.residualVol: pd.Series = None
        self.factorNames: List[str] = []
        self.assetNames: List[str] = []

    def fit(
        self,
        assetReturns: pd.DataFrame,
        factorReturns: pd.DataFrame
    ) -> 'FactorModel':
        """Fit the multi-factor model via per-asset OLS regressions.

        Args:
            assetReturns: DataFrame of daily returns per asset (자산별 일별 수익률).
            factorReturns: DataFrame of daily returns per factor (팩터별 일별 수익률).

        Returns:
            Self for method chaining.
        """
        commonIdx = assetReturns.index.intersection(factorReturns.index)
        assets = assetReturns.loc[commonIdx]
        factors = factorReturns.loc[commonIdx]

        self.factorNames = list(factors.columns)
        self.assetNames = list(assets.columns)

        betas = {}
        residuals = {}

        for asset in self.assetNames:
            y = assets[asset].values
            X = np.column_stack([np.ones(len(y)), factors.values])

            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            betas[asset] = beta[1:]

            residuals[asset] = np.std(y - X @ beta)

        self.factorBetas = pd.DataFrame(
            betas,
            index=self.factorNames
        ).T

        self.residualVol = pd.Series(residuals)

        self.factorCov = factors.cov() * 252

        return self

    def expectedReturn(
        self,
        factorPremiums: Dict[str, float] = None
    ) -> pd.Series:
        """Compute expected returns for each asset based on factor premiums.

        Args:
            factorPremiums: Annualized factor premium per factor name.
                            If None, defaults to 5% for each factor.
                            팩터별 기대 프리미엄 (연율). None이면 5% 기본값 사용.

        Returns:
            Series of expected annualized returns per asset (자산별 기대 수익률).
        """
        if factorPremiums is None:
            factorPremiums = {f: 0.05 for f in self.factorNames}

        premiums = pd.Series([factorPremiums.get(f, 0) for f in self.factorNames],
                            index=self.factorNames)

        return self.factorBetas @ premiums

    def riskDecomposition(self) -> pd.DataFrame:
        """Decompose total risk into factor risk and specific (idiosyncratic) risk.

        Returns:
            DataFrame indexed by asset with columns: totalRisk, factorRisk,
            specificRisk, factorPct (자산별 팩터 리스크 vs 고유 리스크 비율).
        """
        result = []

        for asset in self.assetNames:
            betas = self.factorBetas.loc[asset].values

            factorVar = betas @ self.factorCov.values @ betas
            specificVar = self.residualVol[asset] ** 2
            totalVar = factorVar + specificVar

            result.append({
                'asset': asset,
                'totalRisk': np.sqrt(totalVar),
                'factorRisk': np.sqrt(factorVar),
                'specificRisk': self.residualVol[asset],
                'factorPct': factorVar / totalVar if totalVar > 0 else 0,
            })

        return pd.DataFrame(result).set_index('asset')

    def factorContributionToRisk(self, weights: pd.Series) -> Dict[str, float]:
        """Compute each factor's contribution to portfolio-level factor risk.

        Args:
            weights: Asset weight Series indexed by asset name (자산별 가중치).

        Returns:
            Dict mapping factor name to its proportional risk contribution
            (팩터별 리스크 기여 비율).
        """
        portBetas = (self.factorBetas.T * weights).sum(axis=1)

        totalFactorVar = portBetas @ self.factorCov @ portBetas
        totalFactorRisk = np.sqrt(totalFactorVar)

        contributions = {}
        for factor in self.factorNames:
            factorVar = portBetas[factor] ** 2 * self.factorCov.loc[factor, factor]
            contributions[factor] = np.sqrt(factorVar) / totalFactorRisk if totalFactorRisk > 0 else 0

        return contributions
