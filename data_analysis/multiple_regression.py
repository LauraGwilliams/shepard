
import numpy as np
import statsmodels.api as sm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.formula.api import ols

# VIF
# https://etav.github.io/python/vif_factor_python.html

explanatory_vars = ['Active_Engagement', 'Emotions', 'General_Sophistication',
                    'Musical_Training', 'Perceptual_Abilities',
                    'Singing_Abilities', 'same_diff','up_down','PLV']

# take out PLV
explanatory_vars.remove('PLV')

# gather features
features = "+".join(explanatory_vars)

# https://www.statsmodels.org/devel/gettingstarted.html
# get y and X dataframes based on this regression (X is features, y is exogenous var)
y,X = dmatrices('PLV ~' + features,df,return_type='dataframe')

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns

vif.round(1)

# greater than 5 = highly correlated --> get rid of general sophistication

#__________________________________________________________
# multiple regression
explanatory_vars = ['Active_Engagement', 'Emotions',
                    'Musical_Training', 'Perceptual_Abilities',
                    'Singing_Abilities', 'same_diff','up_down']
pred_var = ['High_Low']


# extract data
X = df[explanatory_vars]
y = df[pred_var]

# fit the multiple regression
regr = make_pipeline(StandardScaler(),
                     LinearRegression())
regr.fit(X, y)

# get beta coeficients of the regression
betas = regr.steps[1][1].coef_

#__________________________________________________________
# multiple correlation
# https://www.statsmodels.org/devel/gettingstarted.html
features = "+".join(explanatory_vars)

model = ols('High_Low ~' + features, data=df).fit()
print model.params
print model.summary()
