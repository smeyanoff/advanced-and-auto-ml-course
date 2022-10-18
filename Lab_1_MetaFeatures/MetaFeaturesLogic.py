from cgi import print_environ
from enum import unique

import numpy as np
import pandas as pd
from scipy.stats.mstats import kurtosis, skew, variation
from sklearn.pipeline import make_pipeline


def dummies(data: pd.Series) -> pd.Series:

    """
    Script requeres replace categorial features with numbers
    """

    uniques = data.unique()
    if len(uniques) == len(data) and data.dtype != object:
        data = data.fillna(data.mean())
        return data

    for unqiue_, replace in zip(uniques, range(len(uniques))):
        data = data.replace(unqiue_, replace)

    # fillna values
    data_ = data.fillna(len(uniques)).copy()

    return data_


class MetaFeatures:
    def __init__(self, data: pd.DataFrame):

        self.data = data.iloc[:, :-1]
        # if column has only one category I drop it
        for col in self.data:
            if len(self.data[col].unique()) == 1:
                self.data = self.data.drop(col, axis=1)
        # if df ufter first step has no columns or less then 20 rows ValueError raised
        if self.data.shape[1] == 0:
            raise ValueError
        if self.data.shape[0] < 20:
            raise ValueError
        self.target = data.iloc[:, -1]
        self.discret_columns = self.data.select_dtypes([int, float]).columns
        self.categor_columns = self.data.select_dtypes([object]).columns

    def get_base_meta_features(self) -> dict:

        """
        Get Base Meta Features
        return: dict
        """

        base_meta_features = {}

        base_meta_features["base_obj_Count"] = self.data.shape[0]
        base_meta_features["base_features_Count"] = self.data.shape[1]
        # compute percent of cat features
        base_meta_features["base_persent_Categorial_features"] = round(
            (self.categor_columns.shape[0]) / self.data.shape[1], 2
        )

        # count classes
        base_meta_features["base_class_Count"] = len(self.target.unique())
        # percent of nans in each column 
        base_meta_features["base_all_percent_of_nans"] = sum(self.data.isna().sum()) / (
            self.data.shape[0] * self.data.shape[1]
        )
        
        # save information about target type
        if self.target.unique().shape[0] == 2:
            base_meta_features["base_target_type"] = -1000  # for binary target
        elif self.target.unique().shape[0] == self.target.shape[0] / 2:
            base_meta_features["base_target_type"] = 1000  # for discret target
        else:
            base_meta_features["base_target_type"] = 0  # for multiclass

        return base_meta_features

    def get_categorial_meta_features(self) -> dict:

        """
        Get Statistical Meta Features for categorical features
        return: dict
        """

        # добавить IV или GINI
        # добавить количество пропусков как отдельную колонку

        statical_categ_meta_features = {}
        data = self.data[self.categor_columns]
        if data.shape[1] == 0:
            return {}
        
        # get fummies of all categorial features
        # add nans
        data = pd.get_dummies(data, prefix_sep='%$#', dummy_na=True)

        stat_for_categ_vars = pd.DataFrame(
            columns=[
                "cat_meta_categCount",
                "cat_meta_entropy",
                "cat_meta_percent_nans",
            ],
            index=self.categor_columns,
        )
        entropy = {}
        columns = []

        for col in data.columns:
            
            # on this step I get information about column name in high_level_col_name
            # and information about each unique category in this column in low_level_column_name
            if len(col.split("%$#")) > 1:
                high_level_col_name = [x for x in col.split("%$#")][0]
                low_level_column_name = [x for x in col.split("%$#")][1]
            else:
                high_level_col_name = col
                low_level_column_name = ""

            if not high_level_col_name in columns:

                columns.append(high_level_col_name)
                entropy[high_level_col_name] = []

            try:
                categCount = sum(data[col])
            except:
                categCount = len(data[col].unique())

            # count percent of nans
            if low_level_column_name == "nan":
                stat_for_categ_vars.loc[
                    high_level_col_name, "cat_meta_percent_nans"
                ] = categCount / len(data[col])

            # write count
            # может правильнее считать в процентах от общей длины 
            stat_for_categ_vars.loc[col, "cat_meta_categCount"] = categCount

            entropy[high_level_col_name].append(categCount / len(data[col]))

        for high_level_col_name in columns:

            # compute entropy
            pi = entropy[high_level_col_name]
            stat_for_categ_vars.loc[high_level_col_name, "cat_meta_entropy"] = -np.sum(
                [np.where(x != 0, x * np.log(x), 0) for x in pi]
            )

        # ogrigate data
        for col in stat_for_categ_vars.columns:
            statical_categ_meta_features[col + "_min"] = round(
                stat_for_categ_vars[col].min(), 2
            )
            statical_categ_meta_features[col + "_max"] = round(
                stat_for_categ_vars[col].max(), 2
            )
            statical_categ_meta_features[col + "_mean"] = round(
                stat_for_categ_vars[col].mean(), 2
            )

        return statical_categ_meta_features

    def get_discret_meta_features(self) -> dict:

        """
        Get Statistical Meta Features for discrete features
        return: dict
        """

        statical_discr_meta_features = {}
        stat_for_discr_vars = pd.DataFrame(
            columns=[
                "disc_meta_min",
                "disc_meta_max",
                "disc_meta_mean",
                "disc_meta_variation",
                "disc_meta_skew",
                "disc_meta_kurtosis",
                "disc_meta_percent_of_nans",
            ],
            index=self.discret_columns,
            dtype=np.float32,
        )

        for col in self.discret_columns:
            stat_for_discr_vars.loc[col, "disc_meta_min"] = self.data[col].min()
            stat_for_discr_vars.loc[col, "disc_meta_max"] = self.data[col].max()
            stat_for_discr_vars.loc[col, "disc_meta_mean"] = self.data[col].mean()
            stat_for_discr_vars.loc[col, "disc_meta_percent_of_nans"] = (
                self.data[col].isna().sum()
            )
            stat_for_discr_vars.loc[col, "disc_meta_variation"] = variation(
                self.data[col]
            )
            stat_for_discr_vars.loc[col, "disc_meta_skew"] = skew(self.data[col])
            stat_for_discr_vars.loc[col, "disc_meta_kurtosis"] = kurtosis(
                self.data[col]
            )

        for col in stat_for_discr_vars.columns:
            statical_discr_meta_features[col + "_min"] = round(
                stat_for_discr_vars[col].min(), 2
            )
            statical_discr_meta_features[col + "_max"] = round(
                stat_for_discr_vars[col].max(), 2
            )
            statical_discr_meta_features[col + "_mean"] = round(
                stat_for_discr_vars[col].mean(), 2
            )
            # statical_discr_meta_features[col+'_variation'] = round(variation(stat_for_discr_vars[col]),2)

        return statical_discr_meta_features

    def get_structure_meta_features(self, model, scaler) -> dict:

        """
        Get Structure Meta Features

        model: Sklearn Linear Model
        scaler: Sklearn Scaler to make Pipeline
        return: dict
        """

        if len(self.categor_columns) == 0:
            df = self.data[self.discret_columns].astype(np.float32)
        elif len(self.discret_columns) == 0:
            categorial_df = self.data[self.categor_columns]
            df = pd.get_dummies(categorial_df, dummy_na=True, prefix_sep="%$#")
        else:
            categorial_df = self.data[self.categor_columns]
            categorial_df = pd.get_dummies(
                categorial_df, dummy_na=True, prefix_sep="%$#"
            )

            discret_df = self.data[self.discret_columns].astype(np.float32)

            df = categorial_df.merge(discret_df, left_index=True, right_index=True)

        df = df.fillna(-100000)
        df = df.replace([np.inf, -np.inf], -100000)

        self.dummied_df = df

        self.dummied_target = dummies(self.target)

        pipeline = make_pipeline(scaler, model)

        pipeline.fit(df, self.dummied_target)

        structure_meta_features = {}
        coef = model.coef_
        col = "model"
        structure_meta_features[col + "_min"] = round(coef.min(), 2)
        structure_meta_features[col + "_max"] = round(coef.max(), 2)
        structure_meta_features[col + "_kurtosis"] = round(kurtosis(coef), 2)
        structure_meta_features[col + "_variation"] = round(variation(coef), 2)

        return structure_meta_features
