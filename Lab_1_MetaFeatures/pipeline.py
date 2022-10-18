import os
import warnings

import numpy as np
import pandas as pd
from MetaFeaturesLogic import MetaFeatures
from scipy.io import arff
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

data_path = os.path.normpath(
    r"C:\Users\mirad\OneDrive\Рабочий стол\Обучение\Advanced and Auto ML\data"
)

list_dir = os.listdir(data_path)

model = SGDRegressor()
scaler = StandardScaler()


all_meta = {}
i = 1

for file in list_dir:

    print(len(list_dir) - i, file)
    accuracy = {}

    try:
        raw_data = arff.loadarff(os.path.join(data_path, file))
    except:
        pass
    data = pd.DataFrame(raw_data[0])

    try:
        meta_features = MetaFeatures(data)
    except ValueError:
        print('No_Unique_Type_Error', file)
        pass

    # get meta features

    base_mf = meta_features.get_base_meta_features()

    st_cat_mf = meta_features.get_categorial_meta_features()

    st_discr_mf = meta_features.get_discret_meta_features()

    structure_mf = meta_features.get_structure_meta_features(model, scaler)

    if structure_mf is None:

        structure_mf = {}


    df = meta_features.dummied_df
    target = meta_features.dummied_target

    df = df.round(3)

    x_train, x_test, y_train, y_test = train_test_split(df, target)
    
    mod = RandomForestRegressor()
    mod.fit(x_train, y_train)
    y_pred = np.round(mod.predict(x_test))
    try:
        accuracy["RF"] = round(accuracy_score(y_test, y_pred), 2)
    except ValueError:
        accuracy["RF"] = round(r2_score(y_test, y_pred), 2)


    mod1 = SGDRegressor()
    sclr = StandardScaler()
    sclr.fit(df)
    x_train_scl = sclr.transform(x_train)
    x_test_scl = pd.DataFrame(
        sclr.transform(x_test, copy=True), index=x_test.index, columns=x_test.columns
    )
    mod1.fit(x_train_scl, y_train)
    y_pred = x_test_scl.apply(lambda x: int(sum(x * mod1.coef_)), axis=1)
    try:
        accuracy["SGD"] = round(accuracy_score(y_test, y_pred), 2)
    except ValueError:
        accuracy["SGD"] = round(r2_score(y_test, y_pred), 2)


    mod2 = KNeighborsRegressor(algorithm="kd_tree")
    mod2.fit(x_train, y_train)
    y_pred = np.round(mod2.predict(x_test))
    try:
        accuracy["kNN"] = round(accuracy_score(y_test, y_pred), 2)
    except ValueError:
        accuracy["kNN"] = round(r2_score(y_test, y_pred), 2)


    # нормализую качество алгоритмов
    Qi = accuracy.values()
    accuracy_weighted = {
        y: np.where(
            max(Qi) == 0, 0, 
            np.where(
                max(Qi) == min(Qi),
                1,
                (x - min(Qi))/(max(Qi)-min(Qi))
            )
            ) 
        for x,y in zip(
            Qi, accuracy.keys()
        )
    }

    all_meta_features = {
        **base_mf,
        **st_cat_mf,
        **st_discr_mf,
        **structure_mf,
        **accuracy_weighted,
    }
    all_meta[file.replace(".arff", "")] = all_meta_features

    i+=1

all_meta_df = pd.DataFrame(all_meta)
all_meta_df = all_meta_df.fillna(-100000)
all_meta_df.T.to_csv("./Lab_1_MetaFeatures/all_meta.csv", index=False)
print('done')