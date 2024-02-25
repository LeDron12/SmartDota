from sklearn.preprocessing import MultiLabelBinarizer

import pandas as pd
import numpy as np

from typing import List


""" Making one hot encoding for radiant and dire heroes """
def make_hero_onehot(df: pd.DataFrame, 
                     radiant_col_name: List[int]='radiant_hero_ids', 
                     dire_col_name: List[int]='dire_hero_ids'
                     ) -> pd.DataFrame:

    mlb = MultiLabelBinarizer(sparse_output=True)

    df1 = df.join(
            pd.DataFrame.sparse.from_spmatrix(
                mlb.fit_transform(df.pop(radiant_col_name)),
                index=df.index,
                columns=mlb.classes_
            )
        )
    df2 = df1.join(
            pd.DataFrame.sparse.from_spmatrix(
                mlb.fit_transform(df1.pop(dire_col_name)),
                index=df1.index,
                columns=mlb.classes_
            ),
            lsuffix='_radiant', 
            rsuffix='_dire'
        )
       
    assert df2.shape[1] - df.shape[1] == 2 * (126 - 1)

    return df2
