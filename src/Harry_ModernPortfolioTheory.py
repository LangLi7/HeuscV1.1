import pandas as pd
import numpy as np

class ModernPortfolioTheory:
    def __init__(self, returns_df):
        """
        returns_df: DataFrame mit historischen Renditen der Anlagen, Spalten = Anlagen
        """
        self.returns = returns_df
        self.mean_returns = returns_df.mean()
        self.cov_matrix = returns_df.cov()
        self.num_assets = len(self.mean_returns)
    
    def portfolio_performance(self, weights):
        """
        Berechnung der erwarteten Rendite und Varianz des Portfolios
        weights: Array oder Liste mit Gewichten für jede Anlage
        """
        expected_return = np.dot(weights, self.mean_returns)
        variance = np.dot(weights.T, np.dot(self.cov_matrix, weights))
        return expected_return, variance

    def min_variance_portfolio(self):
        """
        Minimiert das Risiko für eine gegebene Rendite, hier Beispiel: Minimierung der Varianz ohne Vorgaben
        Benutzt numpy.linalg für einfache Lösung mit Gewichtssumme=1
        """
        ones = np.ones(self.num_assets)
        inv_cov = np.linalg.inv(self.cov_matrix)
        weights = np.dot(inv_cov, ones) / np.dot(ones.T, np.dot(inv_cov, ones))
        return weights

    @staticmethod
    def from_csv(file_path):
        """
        Lädt historische Renditen aus CSV-Datei
        """
        df = pd.read_csv(file_path, index_col=0)
        return ModernPortfolioTheory(df)

    @staticmethod
    def from_json(file_path):
        """
        Lädt historische Renditen aus JSON-Datei
        """
        df = pd.read_json(file_path)
        return ModernPortfolioTheory(df)


# Beispiel-Nutzung
mpt = ModernPortfolioTheory.from_csv('historical_returns.csv')
weights = mpt.min_variance_portfolio()
expected_return, variance = mpt.portfolio_performance(weights)

print("Optimale Portfolio-Gewichte:", weights)
print("Erwartete Rendite:", expected_return)
print("Portfolio-Varianz:", variance)
