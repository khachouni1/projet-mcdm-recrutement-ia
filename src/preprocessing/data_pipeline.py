"""
Pipeline de préparation des données et prévention des fuites (Data Leakage).
Directement aligné sur le Cours 1 de l'ECC (slides 84-86) :
Three-Way Holdout method (Train / Validation / Test) avec séparation stricte AVANT tout prétraitement.
"""
import pandas as pd
from sklearn.model_selection import train_test_split


class DataPreparationPipeline:
    def __init__(self, test_size: float = 0.20, val_size: float = 0.15, random_state: int = 42):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

    def split_data_three_way(self, df: pd.DataFrame, target_col: str = None) -> tuple:
        """
        Découpage Three-Way Holdout (Train / Validation / Test).
        Garantit qu'aucune information du Test/Val ne fuite dans le Train.
        """
        # 1. Séparation Test vs (Train + Val)
        stratify_val = df[target_col] if (target_col and target_col in df.columns) else None
        train_val_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_val
        )

        # 2. Séparation Train vs Val
        val_relative_ratio = self.val_size / (1.0 - self.test_size)
        stratify_tv = train_val_df[target_col] if (target_col and target_col in train_val_df.columns) else None
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_relative_ratio,
            random_state=self.random_state,
            stratify=stratify_tv
        )

        return train_df, val_df, test_df
