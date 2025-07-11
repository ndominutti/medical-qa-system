import random

import pandas as pd


class TripletsMiner:
    def __init__(self, random_seed):
        random.seed(random_seed)
        self.TYPE_TO_FUNC_MAP = {"soft": self._soft_mine, "hard": self._hard_mine}
        self.random_seed = random_seed

    def _soft_mine(self, question: str, dataframe: pd.DataFrame, n: int) -> list:
        negatives = (
            dataframe[dataframe["question"] != question]
            .sample(n=n, random_state=self.random_seed)["answer"]
            .tolist()
        )
        return [random.choice(neg) for neg in negatives]

    def _group_chunks(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        grouped_df = dataframe.groupby("question")["answer"].apply(list).reset_index()
        return grouped_df

    def _hard_mine(
        self, question: str, dataframe: pd.DataFrame, n: int
    ) -> pd.DataFrame:
        # Placeholder for hard mining logic
        raise NotImplementedError("Hard mining is not implemented yet.")

    def get_negatives(
        self, dataframe: pd.DataFrame, mine_type: str, negatives_n: int
    ) -> pd.Series:
        df = dataframe.copy()
        df = self._group_chunks(df)
        return df["question"].apply(
            lambda q: self.TYPE_TO_FUNC_MAP[mine_type](q, df, negatives_n)
        )
