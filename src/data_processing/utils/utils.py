import random
from typing import List

import pandas as pd


class TripletsMiner:
    """
    Generates negative samples (triplets) for training tasks by mining from a DataFrame.
    """
    def __init__(self, random_seed:int):
        random.seed(random_seed)
        self.TYPE_TO_FUNC_MAP = {"soft": self._soft_mine, "hard": self._hard_mine}
        self.random_seed = random_seed

    def _soft_mine(self, question: str, dataframe: pd.DataFrame, n: int) -> List[str]:
        """
        Samples 'soft' negative answers from the DataFrame that do not correspond to the given question.

        Args:
            question (str): The anchor question to find negatives for.
            dataframe (pd.DataFrame): DataFrame containing 'question' and 'answer' columns.
            n (int): Number of negative samples to retrieve.

        Returns:
            list: List of randomly chosen negative answers.
        """
        negatives = (
            dataframe[dataframe["question"] != question]
            .sample(n=n, random_state=self.random_seed)["answer"]
            .tolist()
        )
        return [random.choice(neg) for neg in negatives]

    def _group_chunks(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Groups the DataFrame by 'question', aggregating answers into lists.

        Args:
            dataframe (pd.DataFrame): DataFrame with 'question' and 'answer' columns.

        Returns:
            pd.DataFrame: Grouped DataFrame with unique questions and list of answers.
        """
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
        """
        Retrieves negative samples for each question in the DataFrame using the specified mining method.

        Args:
            dataframe (pd.DataFrame): DataFrame containing 'question' and 'answer' columns.
            mine_type (str): Type of mining to perform; must be one of ['soft', 'hard'].
            negatives_n (int): Number of negatives to retrieve per question.

        Returns:
            pd.Series: Series of lists containing negative samples for each question.
        """
        df = dataframe.copy()
        df = self._group_chunks(df)
        return df["question"].apply(
            lambda q: self.TYPE_TO_FUNC_MAP[mine_type](q, df, negatives_n)
        )
