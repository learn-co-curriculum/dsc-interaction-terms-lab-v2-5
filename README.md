
# Interactions - Lab

## Introduction

In this lab, you'll explore interactions in the Boston Housing data set.

## Objectives

You will be able to:
- Understand what interactions are
- Understand how to accommodate for interactions in regression

## Build a baseline model 

You'll use a couple of built-in functions, which we imported for you below.


```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

Import the Boston data set using `load_boston()`. We won't bother to preprocess the data in this lab. If you still want to build a model in the end, you can do that, but this lab will just focus on finding meaningful insights in interactions and how they can improve $R^2$ values.


```python
regression = LinearRegression()
boston = load_boston()
```

Create a baseline model which includes all the variables in the Boston housing data set to predict the house prices. Then use 10-fold cross-validation and report the mean $R^2$ value as the baseline $R^2$.


```python
y = pd.DataFrame(boston.target,columns = ["target"])
df = pd.DataFrame(boston.data, columns = boston.feature_names)
all_data = pd.concat([y,df], axis = 1)

crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
baseline = np.mean(cross_val_score(regression, df, y, scoring="r2", cv=crossvalidation))
```


```python
baseline
```




    0.7189415144723069



## See how interactions improve your baseline

Next, create all possible combinations of interactions, loop over them and add them to the baseline model one by one to see how they affect the $R^2$. We'll look at the 3 interactions which have the biggest effect on our $R^2$, so print out the top 3 combinations.

You will create a for loop to loop through all the combinations of 2 predictors. You can use `combinations` from itertools to create a list of all the pairwise combinations. To find more info on how this is done, have a look [here](https://docs.python.org/2/library/itertools.html).


```python
from itertools import combinations
combinations = list(combinations(boston.feature_names, 2))
```


```python
interactions = []
data = df.copy()
for comb in combinations:
    data["interaction"] = data[comb[0]] * data[comb[1]]
    score = np.mean(cross_val_score(regression, data, y, scoring="r2", cv=crossvalidation))
    if score > baseline: interactions.append((comb[0], comb[1], round(score,3)))
            
print("Top 3 interactions: %s" %sorted(interactions, key=lambda inter: inter[2], reverse=True)[:5])
```

    Top 3 interactions: [('RM', 'LSTAT', 0.786), ('RM', 'TAX', 0.775), ('RM', 'RAD', 0.768), ('RM', 'PTRATIO', 0.763), ('INDUS', 'RM', 0.758)]


## Look at the top 3 interactions: "RM" as a confounding factor

The top three interactions seem to involve "RM", the number of rooms as a confounding variable for all of them. Let's have a look at interaction plots for all three of them. This exercise will involve:

- splitting the data up in 3 groups: one for houses with a few rooms, one for houses with a "medium" amount of rooms, one for a high amount of rooms.
- Create a function `build_interaction_rm`. This function takes an argument `varname` (which can be set equal to the column name as a string) and a column `description` (which describes the variable or varname, to be included on the x-axis of the plot). The function outputs a plot that uses "RM" as a confounding factor. Each plot should have three regression lines, one for each level of "RM." 

The data has been split into high, medium and low number of rooms for you.


```python
rm = np.asarray(df[["RM"]]).reshape(len(df[["RM"]]))
```


```python
high_rm = all_data[rm > np.percentile(rm, 67)]
med_rm = all_data[(rm > np.percentile(rm, 33)) & (rm <= np.percentile(rm, 67))]
low_rm = all_data[rm <= np.percentile(rm, 33)]
```

Create `build_interaction_rm`.


```python
def build_interaction_rm(varname, description):
    regression_h = LinearRegression()
    regression_m = LinearRegression()
    regression_l = LinearRegression()
    regression_h.fit(high_rm[varname].values.reshape(-1, 1), high_rm["target"])
    regression_m.fit(med_rm[varname].values.reshape(-1, 1), med_rm["target"])
    regression_l.fit(low_rm[varname].values.reshape(-1, 1), low_rm["target"])

    # Make predictions using the testing set
    pred_high = regression_h.predict(high_rm[varname].values.reshape(-1, 1))
    pred_med = regression_m.predict(med_rm[varname].values.reshape(-1, 1))
    pred_low = regression_l.predict(low_rm[varname].values.reshape(-1, 1))

    # The coefficients
    print(regression_h.coef_)
    print(regression_m.coef_)
    print(regression_l.coef_)

    # Plot outputs
    plt.figure(figsize=(12,7))
    plt.scatter(high_rm[varname], high_rm["target"],  color='blue', alpha = 0.3, label = "more rooms")
    plt.scatter(med_rm[varname], med_rm["target"],  color='red', alpha = 0.3, label = "medium rooms")
    plt.scatter(low_rm[varname], low_rm["target"],  color='orange', alpha = 0.3, label = "low amount of rooms")

    plt.plot(low_rm[varname], pred_low,  color='orange', linewidth=2)
    plt.plot(med_rm[varname], pred_med,  color='red', linewidth=2)
    plt.plot(high_rm[varname], pred_high,  color='blue', linewidth=2)
    plt.ylabel("house value")
    plt.xlabel(description)
    plt.legend()
```

Next, use build_interaction_rm with the three variables that came out with the highest effect on $R^2$


```python
build_interaction_rm("LSTAT", "% of lower status")
```

    [-1.46614438]
    [-0.67588205]
    [-0.51981339]



![png](index_files/index_25_1.png)



```python
build_interaction_rm("RAD","highway accessibility index")
```

    [-0.66803793]
    [-0.24276834]
    [-0.17393132]



![png](index_files/index_26_1.png)



```python
build_interaction_rm("TAX", "average tax rate")
```

    [-0.03708037]
    [-0.01431143]
    [-0.01166035]



![png](index_files/index_27_1.png)


## Build a final model including all three interactions at once

Use 10-fold cross validation.


```python
regression = LinearRegression()
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)

df_inter = df.copy()
df_inter["RM_LSTAT"] = df["RM"] * df["LSTAT"]
df_inter["RM_TAX"] = df["RM"] * df["TAX"]
df_inter["RM_RAD"] = df["RM"] * df["RAD"]

final_model = np.mean(cross_val_score(regression, df_inter, y, scoring="r2", cv=crossvalidation))
```


```python
final_model
```




    0.7853084169016563



Our $R^2$ has increased considerably! Let's have a look in statsmodels to see if all these interactions are significant.


```python
import statsmodels.api as sm
df_inter_sm = sm.add_constant(df_inter)
model = sm.OLS(y,df_inter_sm)
results = model.fit()

results.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>target</td>      <th>  R-squared:         </th> <td>   0.815</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.809</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   134.4</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 18 Apr 2019</td> <th>  Prob (F-statistic):</th> <td>2.83e-167</td>
</tr>
<tr>
  <th>Time:</th>                 <td>09:49:00</td>     <th>  Log-Likelihood:    </th> <td> -1413.8</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   506</td>      <th>  AIC:               </th> <td>   2862.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   489</td>      <th>  BIC:               </th> <td>   2933.</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    16</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>        <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>    <td>  -14.8281</td> <td>    7.426</td> <td>   -1.997</td> <td> 0.046</td> <td>  -29.419</td> <td>   -0.237</td>
</tr>
<tr>
  <th>CRIM</th>     <td>   -0.1628</td> <td>    0.028</td> <td>   -5.782</td> <td> 0.000</td> <td>   -0.218</td> <td>   -0.107</td>
</tr>
<tr>
  <th>ZN</th>       <td>    0.0174</td> <td>    0.012</td> <td>    1.462</td> <td> 0.144</td> <td>   -0.006</td> <td>    0.041</td>
</tr>
<tr>
  <th>INDUS</th>    <td>    0.0904</td> <td>    0.053</td> <td>    1.715</td> <td> 0.087</td> <td>   -0.013</td> <td>    0.194</td>
</tr>
<tr>
  <th>CHAS</th>     <td>    2.5987</td> <td>    0.740</td> <td>    3.512</td> <td> 0.000</td> <td>    1.145</td> <td>    4.053</td>
</tr>
<tr>
  <th>NOX</th>      <td>  -13.5145</td> <td>    3.276</td> <td>   -4.125</td> <td> 0.000</td> <td>  -19.952</td> <td>   -7.077</td>
</tr>
<tr>
  <th>RM</th>       <td>   10.8229</td> <td>    0.986</td> <td>   10.977</td> <td> 0.000</td> <td>    8.886</td> <td>   12.760</td>
</tr>
<tr>
  <th>AGE</th>      <td>    0.0054</td> <td>    0.011</td> <td>    0.471</td> <td> 0.638</td> <td>   -0.017</td> <td>    0.028</td>
</tr>
<tr>
  <th>DIS</th>      <td>   -0.9554</td> <td>    0.175</td> <td>   -5.474</td> <td> 0.000</td> <td>   -1.298</td> <td>   -0.612</td>
</tr>
<tr>
  <th>RAD</th>      <td>    0.7080</td> <td>    0.476</td> <td>    1.487</td> <td> 0.138</td> <td>   -0.228</td> <td>    1.643</td>
</tr>
<tr>
  <th>TAX</th>      <td>    0.0333</td> <td>    0.025</td> <td>    1.356</td> <td> 0.176</td> <td>   -0.015</td> <td>    0.082</td>
</tr>
<tr>
  <th>PTRATIO</th>  <td>   -0.6858</td> <td>    0.113</td> <td>   -6.077</td> <td> 0.000</td> <td>   -0.907</td> <td>   -0.464</td>
</tr>
<tr>
  <th>B</th>        <td>    0.0049</td> <td>    0.002</td> <td>    2.115</td> <td> 0.035</td> <td>    0.000</td> <td>    0.009</td>
</tr>
<tr>
  <th>LSTAT</th>    <td>    1.1562</td> <td>    0.232</td> <td>    4.988</td> <td> 0.000</td> <td>    0.701</td> <td>    1.612</td>
</tr>
<tr>
  <th>RM_LSTAT</th> <td>   -0.2924</td> <td>    0.041</td> <td>   -7.188</td> <td> 0.000</td> <td>   -0.372</td> <td>   -0.212</td>
</tr>
<tr>
  <th>RM_TAX</th>   <td>   -0.0072</td> <td>    0.004</td> <td>   -1.829</td> <td> 0.068</td> <td>   -0.015</td> <td>    0.001</td>
</tr>
<tr>
  <th>RM_RAD</th>   <td>   -0.0697</td> <td>    0.078</td> <td>   -0.893</td> <td> 0.372</td> <td>   -0.223</td> <td>    0.084</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>255.249</td> <th>  Durbin-Watson:     </th> <td>   1.087</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>2563.776</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 1.963</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td>13.305</td>  <th>  Cond. No.          </th> <td>1.18e+05</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 1.18e+05. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



What is your conclusion here?


```python
"""Even though each individual interaction had a considerable effect
on the $R^2$, but adding all three of them in our final model led to
insignificant results for some of them. It might be worth checking 
how the $R^2$ changes again when just including 2 interactions in 
the final model."""
```




    'Even though each individual interaction had a considerable effect\non the $R^2$, but adding all three of them in our final model led to\ninsignificant results for some of them. It might be worth checking \nhow the $R^2$ changes again when just including 2 interactions in \nthe final model.'



## Summary

You should now understand how to include interaction effects in your model! As you can see, interactions can have a strong impact on linear regression models, and they should always be considered when you are constructing your models.
