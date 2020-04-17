
# Interactions - Lab

## Introduction

In this lab, you'll explore interactions in the Ames Housing dataset.

## Objectives

You will be able to:
- Implement interaction terms in Python using the `sklearn` and `statsmodels` packages 
- Interpret interaction variables in the context of a real-world problem 

## Build a baseline model 

You'll use a couple of built-in functions, which we imported for you below: 


```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

If you still want to build a model in the end, you can do that, but this lab will just focus on finding meaningful insights in interactions and how they can improve $R^2$ values.


```python
regression = LinearRegression()
```

Create a baseline model which includes all the variables we selected from the Ames housing data set to predict the house prices. Then use 10-fold cross-validation and report the mean $R^2$ value as the baseline $R^2$.


```python
ames = pd.read_csv('ames.csv')

continuous = ['LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']
categoricals = ['BldgType', 'KitchenQual', 'SaleType', 'MSZoning', 'Street', 'Neighborhood']

ames_cont = ames[continuous]

# log features
log_names = [f'{column}_log' for column in ames_cont.columns]

ames_log = np.log(ames_cont)
ames_log.columns = log_names

# normalize (subract mean and divide by std)

def normalize(feature):
    return (feature - feature.mean()) / feature.std()

ames_log_norm = ames_log.apply(normalize)

# one hot encode categoricals
ames_ohe = pd.get_dummies(ames[categoricals], prefix=categoricals)

preprocessed = pd.concat([ames_cont, ames_ohe], axis=1)

X = preprocessed.drop('SalePrice', axis=1)
y = preprocessed['SalePrice']

crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
baseline = np.mean(cross_val_score(regression, X, y, scoring='r2', cv=crossvalidation))

baseline
```




    0.777381712804318



## See how interactions improve your baseline

Next, create all possible combinations of interactions, loop over them and add them to the baseline model one by one to see how they affect the $R^2$. We'll look at the 3 interactions which have the biggest effect on our $R^2$, so print out the top 3 combinations.

You will create a `for` loop to loop through all the combinations of 2 predictors. You can use `combinations` from itertools to create a list of all the pairwise combinations. To find more info on how this is done, have a look [here](https://docs.python.org/2/library/itertools.html).

Since there are so many different neighbourhoods we will exclude


```python
from itertools import combinations

interactions = []

feat_combinations = combinations(X.columns, 2)

data = X.copy()
for i, (a, b) in enumerate(feat_combinations):
    data['interaction'] = data[a] * data[b]
    score = np.mean(cross_val_score(regression, data, y, scoring='r2', cv=crossvalidation))
    if score > baseline:
        interactions.append((a, b, round(score,3)))
    
    if i % 50 == 0:
        print(i)
            
print('Top 3 interactions: %s' %sorted(interactions, key=lambda inter: inter[2], reverse=True)[:3])
```

    0
    50
    100
    150
    200
    250
    300
    350
    400
    450
    500
    550
    600
    650
    700
    750
    800
    850
    900
    950
    1000
    1050
    1100
    1150
    1200
    1250
    1300
    1350
    Top 3 interactions: [('LotArea', 'Neighborhood_Edwards', 0.809), ('GrLivArea', 'Neighborhood_Edwards', 0.808), ('1stFlrSF', 'Neighborhood_Edwards', 0.803)]


It looks like the top interactions involve the Neighborhood_Edwards feature so lets add the interaction between LotArea and Edwards to our model.

We can interpret this feature as the relationship between LotArea and SalePrice when the house is in Edwards or not.

## Visualize the Interaction

Separate all houses that are located in Edwards and those that are not. Run a linear regression on each population against `SalePrice`. Visualize the regression line and data points with price on the y axis and LotArea on the x axis.


```python

fig, ax = plt.subplots(figsize=(13, 10))

col = 'LotArea'

is_in = preprocessed.loc[preprocessed['Neighborhood_Edwards'] == 1, [col, 'SalePrice']]

linreg = LinearRegression()
linreg.fit(np.log(is_in[[col]]), np.log(is_in['SalePrice']))

preds = linreg.predict(np.log(is_in[[col]]))

ax.scatter(np.log(is_in[[col]]), np.log(is_in['SalePrice']), alpha=.3, label=None)

x = np.linspace(6, 12)
ax.plot(x, linreg.predict(x.reshape(-1, 1)), label=f'In Edwards:   {linreg.coef_[0]:.2f}')

not_in = preprocessed.loc[preprocessed['Neighborhood_Edwards'] == 0, [col, 'SalePrice']]

linreg = LinearRegression()
linreg.fit(np.log(not_in[[col]]), np.log(not_in['SalePrice']))

preds = linreg.predict(np.log(not_in[[col]]))

ax.scatter(np.log(not_in[[col]]), np.log(not_in['SalePrice']), alpha=.1, label=None)

x = np.linspace(6, 12)
ax.plot(x, linreg.predict(x.reshape(-1, 1)), label=f'Outside of Edwards:   {linreg.coef_[0]:.2f}')

ax.legend()
```




    <matplotlib.legend.Legend at 0x1a259ad4a8>




![png](index_files/index_17_1.png)


## Build a final model with interactions

Use 10-fold cross-validation to build a model using the above interaction. 


```python
regression = LinearRegression()
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
final = X.copy()

final['Neighborhood_Edwards*LotArea'] = final['Neighborhood_Edwards'] * final['LotArea']

final_model = np.mean(cross_val_score(regression, final, y, scoring='r2', cv=crossvalidation))

final_model
```




    0.8093363403933006



Our $R^2$ has increased considerably! Let's have a look in `statsmodels` to see if this interactions are significant.


```python
import statsmodels.api as sm
df_inter_sm = sm.add_constant(final)
model = sm.OLS(y,final)
results = model.fit()

results.summary()
```

    /Users/lore.dirick/anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)





<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>SalePrice</td>    <th>  R-squared:         </th> <td>   0.835</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.829</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   148.6</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 16 Apr 2020</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>14:27:18</td>     <th>  Log-Likelihood:    </th> <td> -17229.</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>  1460</td>      <th>  AIC:               </th> <td>3.456e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  1411</td>      <th>  BIC:               </th> <td>3.482e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    48</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
                <td></td>                  <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>LotArea</th>                      <td>    0.6108</td> <td>    0.103</td> <td>    5.916</td> <td> 0.000</td> <td>    0.408</td> <td>    0.813</td>
</tr>
<tr>
  <th>1stFlrSF</th>                     <td>   35.0664</td> <td>    3.288</td> <td>   10.664</td> <td> 0.000</td> <td>   28.616</td> <td>   41.517</td>
</tr>
<tr>
  <th>GrLivArea</th>                    <td>   58.1426</td> <td>    2.405</td> <td>   24.171</td> <td> 0.000</td> <td>   53.424</td> <td>   62.861</td>
</tr>
<tr>
  <th>BldgType_1Fam</th>                <td> 2.922e+04</td> <td> 2902.954</td> <td>   10.066</td> <td> 0.000</td> <td> 2.35e+04</td> <td> 3.49e+04</td>
</tr>
<tr>
  <th>BldgType_2fmCon</th>              <td> 1.242e+04</td> <td> 5897.809</td> <td>    2.105</td> <td> 0.035</td> <td>  847.853</td> <td>  2.4e+04</td>
</tr>
<tr>
  <th>BldgType_Duplex</th>              <td>-3641.5299</td> <td> 4775.013</td> <td>   -0.763</td> <td> 0.446</td> <td> -1.3e+04</td> <td> 5725.358</td>
</tr>
<tr>
  <th>BldgType_Twnhs</th>               <td>-7057.2521</td> <td> 5689.576</td> <td>   -1.240</td> <td> 0.215</td> <td>-1.82e+04</td> <td> 4103.687</td>
</tr>
<tr>
  <th>BldgType_TwnhsE</th>              <td> 5879.6997</td> <td> 3832.141</td> <td>    1.534</td> <td> 0.125</td> <td>-1637.606</td> <td> 1.34e+04</td>
</tr>
<tr>
  <th>KitchenQual_Ex</th>               <td> 6.041e+04</td> <td> 4203.276</td> <td>   14.371</td> <td> 0.000</td> <td> 5.22e+04</td> <td> 6.87e+04</td>
</tr>
<tr>
  <th>KitchenQual_Fa</th>               <td>-2.033e+04</td> <td> 4811.017</td> <td>   -4.225</td> <td> 0.000</td> <td>-2.98e+04</td> <td>-1.09e+04</td>
</tr>
<tr>
  <th>KitchenQual_Gd</th>               <td> 6309.0317</td> <td> 2725.293</td> <td>    2.315</td> <td> 0.021</td> <td>  962.969</td> <td> 1.17e+04</td>
</tr>
<tr>
  <th>KitchenQual_TA</th>               <td>-9568.2577</td> <td> 2512.797</td> <td>   -3.808</td> <td> 0.000</td> <td>-1.45e+04</td> <td>-4639.037</td>
</tr>
<tr>
  <th>SaleType_COD</th>                 <td>-1.616e+04</td> <td> 6414.150</td> <td>   -2.519</td> <td> 0.012</td> <td>-2.87e+04</td> <td>-3575.093</td>
</tr>
<tr>
  <th>SaleType_CWD</th>                 <td> 5095.8845</td> <td> 1.54e+04</td> <td>    0.330</td> <td> 0.741</td> <td>-2.52e+04</td> <td> 3.54e+04</td>
</tr>
<tr>
  <th>SaleType_Con</th>                 <td> 4.676e+04</td> <td> 2.18e+04</td> <td>    2.149</td> <td> 0.032</td> <td> 4071.965</td> <td> 8.95e+04</td>
</tr>
<tr>
  <th>SaleType_ConLD</th>               <td>  267.8450</td> <td> 1.11e+04</td> <td>    0.024</td> <td> 0.981</td> <td>-2.14e+04</td> <td> 2.19e+04</td>
</tr>
<tr>
  <th>SaleType_ConLI</th>               <td>  888.5698</td> <td> 1.39e+04</td> <td>    0.064</td> <td> 0.949</td> <td>-2.64e+04</td> <td> 2.82e+04</td>
</tr>
<tr>
  <th>SaleType_ConLw</th>               <td>-5236.1620</td> <td>  1.4e+04</td> <td>   -0.375</td> <td> 0.708</td> <td>-3.26e+04</td> <td> 2.22e+04</td>
</tr>
<tr>
  <th>SaleType_New</th>                 <td> 1.611e+04</td> <td> 5396.453</td> <td>    2.986</td> <td> 0.003</td> <td> 5528.209</td> <td> 2.67e+04</td>
</tr>
<tr>
  <th>SaleType_Oth</th>                 <td>-7445.4890</td> <td> 1.76e+04</td> <td>   -0.423</td> <td> 0.673</td> <td> -4.2e+04</td> <td> 2.71e+04</td>
</tr>
<tr>
  <th>SaleType_WD</th>                  <td>-3470.6182</td> <td> 4492.513</td> <td>   -0.773</td> <td> 0.440</td> <td>-1.23e+04</td> <td> 5342.104</td>
</tr>
<tr>
  <th>MSZoning_C (all)</th>             <td>-1.649e+04</td> <td> 1.06e+04</td> <td>   -1.553</td> <td> 0.121</td> <td>-3.73e+04</td> <td> 4333.248</td>
</tr>
<tr>
  <th>MSZoning_FV</th>                  <td>  2.15e+04</td> <td> 7819.714</td> <td>    2.749</td> <td> 0.006</td> <td> 6156.896</td> <td> 3.68e+04</td>
</tr>
<tr>
  <th>MSZoning_RH</th>                  <td> 1565.0621</td> <td> 8028.716</td> <td>    0.195</td> <td> 0.845</td> <td>-1.42e+04</td> <td> 1.73e+04</td>
</tr>
<tr>
  <th>MSZoning_RL</th>                  <td> 1.276e+04</td> <td> 3975.156</td> <td>    3.210</td> <td> 0.001</td> <td> 4963.351</td> <td> 2.06e+04</td>
</tr>
<tr>
  <th>MSZoning_RM</th>                  <td> 1.748e+04</td> <td> 4540.010</td> <td>    3.851</td> <td> 0.000</td> <td> 8578.857</td> <td> 2.64e+04</td>
</tr>
<tr>
  <th>Street_Grvl</th>                  <td> 2.003e+04</td> <td> 1.02e+04</td> <td>    1.961</td> <td> 0.050</td> <td>   -1.589</td> <td> 4.01e+04</td>
</tr>
<tr>
  <th>Street_Pave</th>                  <td> 1.679e+04</td> <td> 5572.202</td> <td>    3.012</td> <td> 0.003</td> <td> 5855.136</td> <td> 2.77e+04</td>
</tr>
<tr>
  <th>Neighborhood_Blmngtn</th>         <td> 1.084e+04</td> <td> 8819.610</td> <td>    1.229</td> <td> 0.219</td> <td>-6460.879</td> <td> 2.81e+04</td>
</tr>
<tr>
  <th>Neighborhood_Blueste</th>         <td> 1.885e+04</td> <td> 2.29e+04</td> <td>    0.822</td> <td> 0.411</td> <td>-2.62e+04</td> <td> 6.39e+04</td>
</tr>
<tr>
  <th>Neighborhood_BrDale</th>          <td> -363.5975</td> <td> 9693.132</td> <td>   -0.038</td> <td> 0.970</td> <td>-1.94e+04</td> <td> 1.87e+04</td>
</tr>
<tr>
  <th>Neighborhood_BrkSide</th>         <td>-3.052e+04</td> <td> 5047.172</td> <td>   -6.046</td> <td> 0.000</td> <td>-4.04e+04</td> <td>-2.06e+04</td>
</tr>
<tr>
  <th>Neighborhood_ClearCr</th>         <td>-1.273e+04</td> <td> 6635.232</td> <td>   -1.918</td> <td> 0.055</td> <td>-2.57e+04</td> <td>  290.949</td>
</tr>
<tr>
  <th>Neighborhood_CollgCr</th>         <td> 2881.7921</td> <td> 3296.652</td> <td>    0.874</td> <td> 0.382</td> <td>-3585.075</td> <td> 9348.659</td>
</tr>
<tr>
  <th>Neighborhood_Crawfor</th>         <td> 2742.1253</td> <td> 4857.565</td> <td>    0.565</td> <td> 0.572</td> <td>-6786.701</td> <td> 1.23e+04</td>
</tr>
<tr>
  <th>Neighborhood_Edwards</th>         <td> 3.292e+04</td> <td> 6255.485</td> <td>    5.263</td> <td> 0.000</td> <td> 2.07e+04</td> <td> 4.52e+04</td>
</tr>
<tr>
  <th>Neighborhood_Gilbert</th>         <td> -851.6815</td> <td> 4190.447</td> <td>   -0.203</td> <td> 0.839</td> <td>-9071.859</td> <td> 7368.496</td>
</tr>
<tr>
  <th>Neighborhood_IDOTRR</th>          <td> -4.11e+04</td> <td> 7466.396</td> <td>   -5.505</td> <td> 0.000</td> <td>-5.58e+04</td> <td>-2.65e+04</td>
</tr>
<tr>
  <th>Neighborhood_MeadowV</th>         <td>-1.664e+04</td> <td> 8866.487</td> <td>   -1.877</td> <td> 0.061</td> <td> -3.4e+04</td> <td>  749.566</td>
</tr>
<tr>
  <th>Neighborhood_Mitchel</th>         <td>-9326.6382</td> <td> 4965.415</td> <td>   -1.878</td> <td> 0.061</td> <td>-1.91e+04</td> <td>  413.752</td>
</tr>
<tr>
  <th>Neighborhood_NAmes</th>           <td>-2.378e+04</td> <td> 3027.800</td> <td>   -7.853</td> <td> 0.000</td> <td>-2.97e+04</td> <td>-1.78e+04</td>
</tr>
<tr>
  <th>Neighborhood_NPkVill</th>         <td> 1.716e+04</td> <td> 1.13e+04</td> <td>    1.517</td> <td> 0.130</td> <td>-5035.043</td> <td> 3.94e+04</td>
</tr>
<tr>
  <th>Neighborhood_NWAmes</th>          <td>-1.572e+04</td> <td> 4297.809</td> <td>   -3.658</td> <td> 0.000</td> <td>-2.42e+04</td> <td>-7292.309</td>
</tr>
<tr>
  <th>Neighborhood_NoRidge</th>         <td> 6.116e+04</td> <td> 5802.386</td> <td>   10.540</td> <td> 0.000</td> <td> 4.98e+04</td> <td> 7.25e+04</td>
</tr>
<tr>
  <th>Neighborhood_NridgHt</th>         <td> 5.343e+04</td> <td> 4730.312</td> <td>   11.295</td> <td> 0.000</td> <td> 4.42e+04</td> <td> 6.27e+04</td>
</tr>
<tr>
  <th>Neighborhood_OldTown</th>         <td>-4.916e+04</td> <td> 4938.221</td> <td>   -9.954</td> <td> 0.000</td> <td>-5.88e+04</td> <td>-3.95e+04</td>
</tr>
<tr>
  <th>Neighborhood_SWISU</th>           <td>-4.748e+04</td> <td> 6953.924</td> <td>   -6.828</td> <td> 0.000</td> <td>-6.11e+04</td> <td>-3.38e+04</td>
</tr>
<tr>
  <th>Neighborhood_Sawyer</th>          <td>-2.453e+04</td> <td> 4299.054</td> <td>   -5.705</td> <td> 0.000</td> <td> -3.3e+04</td> <td>-1.61e+04</td>
</tr>
<tr>
  <th>Neighborhood_SawyerW</th>         <td>-4573.2529</td> <td> 4664.797</td> <td>   -0.980</td> <td> 0.327</td> <td>-1.37e+04</td> <td> 4577.430</td>
</tr>
<tr>
  <th>Neighborhood_Somerst</th>         <td>  1.49e+04</td> <td> 7438.164</td> <td>    2.003</td> <td> 0.045</td> <td>  305.753</td> <td> 2.95e+04</td>
</tr>
<tr>
  <th>Neighborhood_StoneBr</th>         <td> 6.479e+04</td> <td> 7077.402</td> <td>    9.155</td> <td> 0.000</td> <td> 5.09e+04</td> <td> 7.87e+04</td>
</tr>
<tr>
  <th>Neighborhood_Timber</th>          <td> 7540.1022</td> <td> 5710.630</td> <td>    1.320</td> <td> 0.187</td> <td>-3662.136</td> <td> 1.87e+04</td>
</tr>
<tr>
  <th>Neighborhood_Veenker</th>         <td> 2.636e+04</td> <td> 9993.454</td> <td>    2.638</td> <td> 0.008</td> <td> 6761.011</td> <td>  4.6e+04</td>
</tr>
<tr>
  <th>Neighborhood_Edwards*LotArea</th> <td>   -7.1552</td> <td>    0.513</td> <td>  -13.959</td> <td> 0.000</td> <td>   -8.161</td> <td>   -6.150</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>381.039</td> <th>  Durbin-Watson:     </th> <td>   1.945</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>3465.080</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.947</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td>10.306</td>  <th>  Cond. No.          </th> <td>1.01e+16</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The smallest eigenvalue is 3.06e-21. This might indicate that there are<br/>strong multicollinearity problems or that the design matrix is singular.



What is your conclusion here?

## Summary

You should now understand how to include interaction effects in your model! As you can see, interactions can have a strong impact on linear regression models, and they should always be considered when you are constructing your models.
