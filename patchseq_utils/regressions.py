import traceback
import pandas as pd
from statsmodels.stats.weightstats import ttest_ind
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.anova import anova_lm

def mixedlm(data, feature, md, groups, **kwargs):
    res = smf.mixedlm(f"{feature}~{md}", data=data, groups=groups).fit()
    model = f"{md}+{groups}" if groups else f"{md}"
    formula = f"{feature}~{model}"
    return {
        'feature':feature,
        'model':model,
        'pval': res.pvalues[md],
        'coeff': res.fe_params[md],
        'coeff_se': res.bse_fe[md],
        'group_re': res.cov_re.iloc[0,0],
    }

def ols(data, feature, md, groups=None, categorical=False, cov_type='HC3'):
    model = f"{md}+{groups}" if groups else f"{md}"
    formula = f"{feature}~{model}"
        
    res = smf.ols(formula, data=data).fit(cov_type=cov_type)
    out = {
        'feature':feature,
        'model':model,
        'nobs': res.nobs,
        'rsquared': res.rsquared,
    }
    if categorical: 
        anova = anova_lm(res, typ=2)
        out['pval'] = anova.loc[md,"PR(>F)"]
    else:
        out['pval'] = res.pvalues[md]
        out['coeff'] = res.params[md]
        
    return out

def run_regressions(function, data, features, md_features, groups=None, cov_type='HC3',
                    categorical=False, drop_small=False):
    results = []
    for feature in features:
        for md in md_features:
            try:
                data1 = data.dropna(subset=[feature, md])
                if drop_small:
                    small_types = data1[md].value_counts().loc[lambda x: x < 10].index
                    data1 = data1[~data1[md].isin(small_types)]
                results.append(function(data1, feature, md, groups, categorical=categorical, cov_type=cov_type))
            except Exception as e:
                print(f"{feature}, {md}")
                traceback.print_exc()
                continue

    df = pd.DataFrame.from_records(results).dropna(subset=['pval'])
    fdr_method = 'fdr_bh'
    df['pval'] = (df['pval'].pipe(pd.DataFrame)#in case this is Series
        .apply(lambda col: multipletests(col, method=fdr_method)[1]).astype(float)
    )
    return df.sort_values('pval')