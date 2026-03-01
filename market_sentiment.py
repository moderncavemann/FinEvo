"""
Market Sentiment Module - ECCV 2026

Implements endogenous market sentiment index affecting group behavior (GAP 1 & 2 Fix)

Sentiment Index Formula:
    S_t = alpha * S_{t-1} + beta * (delta_P/P) + gamma * delta_u + delta * (delta_GDP/GDP) + epsilon * news_shock

Features:
- Endogenous sentiment based on economic indicators
- Sentiment diffusion among agents
- Behavioral effects on consumption and work decisions
"""

import numpy as np
from typing import Dict, List


class MarketSentiment:
    """
    Market Sentiment Module

    Computes aggregate market sentiment based on economic indicators
    and diffuses sentiment among individual agents.
    """

    def __init__(
        self,
        n_agents: int,
        sentiment_inertia: float = 0.7,
        inflation_sensitivity: float = 0.15,
        unemployment_sensitivity: float = -0.2,
        gdp_sensitivity: float = 0.1,
        news_shock_std: float = 0.05,
        sentiment_diffusion: float = 0.3,
    ):
        """
        Initialize market sentiment module

        Args:
            n_agents: Number of agents in simulation
            sentiment_inertia: Weight of previous sentiment (alpha)
            inflation_sensitivity: Sensitivity to price changes (beta)
            unemployment_sensitivity: Sensitivity to unemployment (gamma)
            gdp_sensitivity: Sensitivity to GDP growth (delta)
            news_shock_std: Std of random news shocks (epsilon)
            sentiment_diffusion: Rate of sentiment diffusion among agents
        """
        self.n_agents = n_agents
        self.sentiment_inertia = sentiment_inertia
        self.inflation_sensitivity = inflation_sensitivity
        self.unemployment_sensitivity = unemployment_sensitivity
        self.gdp_sensitivity = gdp_sensitivity
        self.news_shock_std = news_shock_std
        self.sentiment_diffusion = sentiment_diffusion

        # State
        self.sentiment_history: List[float] = [0.0]
        self.news_shocks: List[float] = []
        self.individual_sentiments: np.ndarray = np.zeros(n_agents)
        self.current_sentiment: float = 0.0

    def reset(self):
        """Reset sentiment state"""
        self.sentiment_history = [0.0]
        self.news_shocks = []
        self.individual_sentiments = np.random.normal(0, 0.1, self.n_agents)
        self.current_sentiment = 0.0

    def update(
        self,
        price_history: List[float],
        unemployment_history: List[int],
        gdp_inflation_history: List[float],
        timestep: int,
        period: int,
    ) -> float:
        """
        Update market sentiment based on economic indicators

        Args:
            price_history: History of price levels
            unemployment_history: History of unemployment counts
            gdp_inflation_history: History of GDP inflation rates
            timestep: Current simulation timestep
            period: Periods per year

        Returns:
            Updated sentiment index
        """
        prev_sentiment = self.sentiment_history[-1]

        # Price change factor (negative relationship - inflation hurts sentiment)
        if len(price_history) >= 2:
            price_change = (price_history[-1] - price_history[-2]) / price_history[-2]
        else:
            price_change = 0

        # Unemployment change factor
        year = (timestep - 1) // period
        if year > 0 and len(unemployment_history) > year:
            current_u = unemployment_history[year] / period / self.n_agents
            prev_u = unemployment_history[year-1] / period / self.n_agents
            unemployment_change = current_u - prev_u
        else:
            unemployment_change = 0

        # GDP growth factor
        if len(gdp_inflation_history) > 0:
            gdp_growth = gdp_inflation_history[-1]
        else:
            gdp_growth = 0

        # Random news shock
        news_shock = np.random.normal(0, self.news_shock_std)
        self.news_shocks.append(news_shock)

        # Calculate new sentiment
        new_sentiment = (
            self.sentiment_inertia * prev_sentiment +
            self.inflation_sensitivity * (-price_change) +  # Negative: inflation hurts sentiment
            self.unemployment_sensitivity * unemployment_change +
            self.gdp_sensitivity * gdp_growth +
            news_shock
        )

        # Clip to [-1, 1] range
        new_sentiment = np.clip(new_sentiment, -1, 1)

        self.sentiment_history.append(new_sentiment)
        self.current_sentiment = new_sentiment

        # Diffuse sentiment to individual agents
        self._diffuse_sentiment(new_sentiment)

        return new_sentiment

    def _diffuse_sentiment(self, aggregate_sentiment: float):
        """Diffuse aggregate sentiment to individual agents with noise"""
        for i in range(self.n_agents):
            individual = self.individual_sentiments[i]
            new_individual = (
                (1 - self.sentiment_diffusion) * individual +
                self.sentiment_diffusion * aggregate_sentiment +
                np.random.normal(0, 0.02)
            )
            self.individual_sentiments[i] = np.clip(new_individual, -1, 1)

    def get_sentiment_effects(self) -> Dict[str, float]:
        """
        Get sentiment effects on agent behavior

        Returns:
            Dict with multipliers for consumption, work, and risk preference
        """
        sentiment = self.current_sentiment
        return {
            "consumption_multiplier": 1 + 0.15 * sentiment,
            "work_multiplier": 1 + 0.05 * sentiment,
            "risk_preference_shift": 0.1 * sentiment,
        }

    def get_sentiment_prompt(self) -> str:
        """Generate sentiment description for LLM prompt"""
        sentiment = self.current_sentiment

        if sentiment > 0.5:
            return (
                f"Market sentiment is highly optimistic (index: {sentiment:.2f}). "
                "People are confident and spending more."
            )
        elif sentiment > 0.2:
            return (
                f"Market sentiment is moderately positive (index: {sentiment:.2f}). "
                "Cautious optimism prevails."
            )
        elif sentiment > -0.2:
            return (
                f"Market sentiment is neutral (index: {sentiment:.2f}). "
                "Economic outlook is mixed."
            )
        elif sentiment > -0.5:
            return (
                f"Market sentiment is moderately negative (index: {sentiment:.2f}). "
                "Growing concerns about economy."
            )
        else:
            return (
                f"Market sentiment is pessimistic (index: {sentiment:.2f}). "
                "Many are worried and cutting spending."
            )

    def get_individual_sentiment(self, agent_idx: int) -> float:
        """Get individual agent sentiment"""
        if 0 <= agent_idx < self.n_agents:
            return self.individual_sentiments[agent_idx]
        return 0.0

    def to_dict(self) -> Dict:
        """Serialize sentiment state to dict"""
        return {
            "current_sentiment": self.current_sentiment,
            "sentiment_history": self.sentiment_history,
            "news_shocks": self.news_shocks,
            "individual_sentiments": self.individual_sentiments.tolist(),
        }
