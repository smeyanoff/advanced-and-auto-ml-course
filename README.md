# Advanced-and-Auto-ML-Course

This repo was created as my homeworks on ITMO University (SPB) Advanced-and-Auto cource.
The repo exists to provide information about Advanced-and-Auto approaches in ML.

#### DataFrames for researches lay into "Data" folder

# Meta Features model studing system (not Facebook)

In simple words Meta Studing helps ML Engineer choose best alghoritm. In this task Meta Features are used to be the data and Alghoritm's quality is used to be a target.
Meta Features describe the data as is. They don't change when row or column order changes.
You shoud agregate columns with some function (for example max for discret column) and then agregate them one more time. Then you'll get one value (max of maxes) from all columns.

There are three main types of Meta Features

#### Base
for example: count of objectives, percent of nans etc
#### Statistical
for example: variation, skew
#### Structure
for example: coefficients of linear model

The most difficult thing is to get enaught datasets because in meta studing one row is a whole datset.

## Using 

In Lab_1_MetaFeatures folder 
script 
### pipeline.py 
to add new datasets.
script
### MetaFeaturesLogic.py
contains logic of agregations etc.

In main folder
jupyter notebook 
### Lab1_MetaFeaturesRealisation.ipynb
you may see the results
##### DataFrame with Meta Features and using examples.
