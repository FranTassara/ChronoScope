"""
LimoRhyde pair comparison of CosinorPy, vendored.

compare_pair_df_extended is copied verbatim from CosinorPy 3.1 (cosinor.py;
https://github.com/mmoskon/CosinorPy, MIT licence, (c) Miha Moskon) with three
fixes, all marked "ChronoScope fix":
  * pandas >= 3: the coefficients of the condition terms were taken with
    results.params[idx_params], a positional lookup on a labelled Series that
    pandas 3 rejects (KeyError), so LimoRhyde always failed;
  * analysis 'bootstrap2': the CI and p-value of each difference were stored
    under keys the caller does not read (KeyError 'CI(d_amplitude)');
  * 'p params' (smallest p-value of the condition x rhythm terms) was taken
    over a set of coefficients shifted by one, which included the last
    coefficient of the common rhythm: with one component it was ~0 for any
    clearly rhythmic pair, even with identical conditions.
Otherwise the statistical method is unchanged (the extra-sum-of-squares
F-test keeps CosinorPy's 2N+1 dropped parameters). Every helper it calls comes from the
installed CosinorPy. The loop over pairs (compare_pairs_limo) is in
core.cosinor_analysis.CosinorAnalyzer._compare_pairs_limo.
"""
import numpy as np
import matplotlib.pyplot as plt
import statsmodels
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std

from CosinorPy.cosinor import (
    fit_me, compare_models, generate_independents_compare, evaluate_rhythm_params, project_acr,
    compare_pair_CI, compare_pair_bootstrap, eval_params_diff_CI, eval_params_diff_bootstrap,
)


# additional analysis - options (from best to worst)
# - bootstrap1: independent bootstrap analysis
# - CI1: independent analysis of confidence intervals of two models
# - bootstrap2: bootstrap analysis of a merged model
# - CI2: analysis of confidence intervals of a merged model
def compare_pair_df_extended(df, test1, test2, n_components = 3, period = 24, n_components2 = None, period2 = None, lin_comp = False, model_type = 'lin', alpha = 0, save_to = '', non_rhythmic = False, plot=True, plot_measurements=True, plot_residuals=False, plot_margins=True, x_label = '', y_label = '', additional_analysis = "", bootstrap_type="std", bootstrap_size=1000, samples_per_param_CI=5, max_samples_CI = 1000, sampling_type="LHS", parameters_to_analyse = ['amplitude', 'acrophase', 'mesor'], parameters_angular = ['acrophase']):
       
    n_components1 = n_components
    period1 = period
    if not n_components2:
        n_components2 = n_components1
    if not period2:
        period2 = period1
        
    
    df_pair = df[(df.test == test1) | (df.test == test2)].copy()
    df_pair['h_i'] = 0
    df_pair.loc[df_pair.test == test2, 'h_i'] = 1
    
    
    X = df_pair.x
    Y = df_pair.y
    H_i = df_pair.h_i
    
    """
    ###
    # prepare the independent variables
    ###
    """
    X_i = H_i * X 

    for i in np.arange(n_components1):
        n = i+1

        A = np.sin((X/(period1/n))*np.pi*2)        
        B = np.cos((X/(period1/n))*np.pi*2) 
        if not i:
            X_fit = np.column_stack((A, B))        
        else:
            X_fit = np.column_stack((X_fit, np.column_stack((A, B))))
        
    if non_rhythmic:
        X_fit = np.column_stack((X_fit, H_i))
        idx_params = np.array([-1])
    else:
        for i in np.arange(n_components2):
            n = i+1

            A_i = H_i * np.sin((X/(period2/n))*np.pi*2)        
            B_i = H_i * np.cos((X/(period2/n))*np.pi*2) 
        
               
            X_fit = np.column_stack((X_fit, np.column_stack((A_i, B_i))))
        
        X_fit = np.column_stack((X_fit, H_i))
        
        # idx_params = [3, 4] # n = 1
        # idx_params = [5, 6, 7, 8] # n = 2
        # idx_params = [7, 8, 9, 10, 11, 12] # n = 3
        # idx_params = [9, 10, 11, 12, 13, 14, 15, 16] # n = 4
        
        #strt = 2*n_components + 1
        #stp = strt + 2*n_components + 1

        strt = -2
        stp = strt - 2*n_components2 - 1
        idx_params = np.arange(strt, stp, -1)


           
        
    if lin_comp:
        X_fit = np.column_stack((X_i, X_fit))
        X_fit = np.column_stack((X, X_fit))
        idx_params = np.array(idx_params) + 2                                
    
    X_fit = sm.add_constant(X_fit, has_constant='add')

    """
    ###
    # fit
    ###
    """       
    if model_type == 'lin':
        model = sm.OLS(Y, X_fit)
        results = model.fit()
    elif model_type == 'poisson':
        #model = sm.GLM(Y, X_fit, family=sm.families.Poisson())
        model = statsmodels.discrete.discrete_model.Poisson(Y, X_fit)
        results = model.fit(disp=0)
    elif model_type =='gen_poisson':
        #model = statsmodels.discrete.discrete_model.GeneralizedPoisson(Y, X_fit)
        model = statsmodels.discrete.discrete_model.GeneralizedPoisson(Y, X_fit, p=1)
        results = model.fit(disp=0)
    elif model_type == 'nb':
        
        # if not alpha:
        #     train_model = sm.GLM(Y, X_fit, family=sm.families.Poisson())
        #     train_results = train_model.fit()

        #     df_train = pd.DataFrame()
        #     df_train['Y'] = Y
        #     df_train['mu'] = train_results.mu
        #     df_train['AUX_OLS_DEP'] = df_train.apply(lambda x: ((x['Y'] - x['mu'])**2 - x['Y']) / x['mu'], axis=1)
        #     ols_expr = """AUX_OLS_DEP ~ mu - 1"""
        #     aux_olsr_results = smf.ols(ols_expr, df_train).fit()

        #     alpha=aux_olsr_results.params[0]
        

        # model = sm.GLM(Y, X_fit, family=sm.families.NegativeBinomial(alpha=alpha))
        
        model = statsmodels.discrete.discrete_model.NegativeBinomialP(Y, X_fit, p=1)
        results = model.fit(disp=0)
    else:
        print("Invalid option")
        return    
   
    """
    ###
    # plot
    ###
    """
    
    
    ###
    if plot and plot_measurements:
        plt.plot(df_pair[df_pair.test == test1].x.values, df_pair[df_pair.test == test1].y.values, 'ko', markersize=1, alpha = 0.75)
        plt.plot(df_pair[df_pair.test == test2].x.values, df_pair[df_pair.test == test2].y.values, 'ro', markersize=1, alpha = 0.75)
    #plt.plot(X, results.fittedvalues, label = 'fit')
    
    if model_type =='lin':
        Y_fit = results.fittedvalues        
        p_overall = results.f_pvalue
    else:
        Y_fit = results.predict(X_fit)
        p_overall = results.llr_pvalue
     
    
    X1 = X[H_i == 0]
    #Y_fit1 = Y_fit[H_i == 0]
    #L1 = list(zip(X1,Y_fit1))
    #L1.sort()
    #X1,Y_fit1 = list(zip(*L1))  
    X2 = X[H_i == 1]
    #Y_fit2 = Y_fit[H_i == 1]
    #L2 = list(zip(X2,Y_fit2))
    #L2.sort()
    #X2,Y_fit2 = list(zip(*L2))  
    
    
    
    #plt.plot(X1, Y_fit1, 'k', label = 'fit '+test1)    
    #plt.plot(X2, Y_fit2, 'r', label = 'fit '+test2)    

    ### F-test
    # for nested models
    # using extra-sum-of-squares F test
    # in a similar way as described in CYCLOPS
    # https://www.pnas.org/content/114/20/5312#sec-8
    # https://www.pnas.org/content/pnas/suppl/2017/04/20/1619320114.DCSupplemental/pnas.201619320SI.pdf?targetid=nameddest%3DSTXT

    n_params_full = len(results.params)
    n_params_small = n_params_full - len(idx_params) 
    N = len(Y)

    r_small = fit_me(X, Y, n_components, period, lin_comp=lin_comp, model_type=model_type, alpha=alpha, plot=False, x_label = x_label, y_label = y_label)
    RSS_small = r_small[1]['RSS']
    RSS_full = sum((Y - Y_fit)**2)

    DoF_small = N-n_params_small
    DoF_full = N-n_params_full

    """
    print('RSS_small: ', RSS_small)
    print('RSS_full: ', RSS_full)
    print('n_small, dof: ', n_params_small, DoF_small)
    print('n_full, dof: ', n_params_full, DoF_full)
    """
    p_f = compare_models(RSS_small, RSS_full, DoF_small, DoF_full)

    
    if plot:
         ### plot with higher density   
        n_points = 1000
        max_P = max(period1, period2)
        X_full = np.linspace(min(min(X1),min(X2)), max(max_P, max(max(X1), max(X2))), n_points)
        
        X_fit_full = generate_independents_compare(X_full, X_full, n_components1 = n_components1, period1 = period1, n_components2 = n_components2, period2 = period2, lin_comp= lin_comp)
        
        H_i = X_fit_full[:,-1]
        locs = H_i == 0

        #Y_fit_full = results.predict(X_fit_full)
        #plt.plot(X_full, Y_fit_full[0:n_points], 'k', label = test1)    
        #plt.plot(X_full, Y_fit_full[n_points:], 'r', label = test2)    
        
        Y_fit_full1 = results.predict(X_fit_full[locs])
        Y_fit_full2 = results.predict(X_fit_full[~locs])

        
        plt.plot(X_full, Y_fit_full1, 'k', label = test1)    
        plt.plot(X_full, Y_fit_full2, 'r', label = test2)    
        
        if model_type == 'lin' and plot_margins:
            _, lower, upper = wls_prediction_std(results, exog=X_fit_full[locs], alpha=0.05)
            plt.fill_between(X_full, lower, upper, color='black', alpha=0.1)   
            _, lower, upper = wls_prediction_std(results, exog=X_fit_full[~locs], alpha=0.05)
            plt.fill_between(X_full, lower, upper, color='red', alpha=0.1)

    
        ### end of plot with higher density
        
        
        #p = min(results.pvalues[idx_params])
        #plt.title(test1 + ' vs. ' + test2 + ', p-value=' + "{0:.5f}".format(p))
        plt.title(test1 + ' vs. ' + test2 + ', p-value=' + "{0:.5f}".format(p_f))
        plt.xlabel('time [h]')
        plt.ylabel('measurements')
        plt.legend()
        
        #fig = plt.gcf()
        #fig.set_size_inches(11,8)
        
        if save_to:
            plt.savefig(save_to+'.png')
            plt.savefig(save_to+'.pdf')
            plt.close()
        else:
            plt.show()
        
        if plot_residuals:
            
            resid = results.resid
            sm.qqplot(resid)
            plt.title(test1 + ' vs. ' + test2)
            save_to_resid = save_to.split(".")[0] + '_resid' + save_to.split(".")[1]
            if save_to:
                plt.savefig(save_to_resid)
                plt.close()
            else:
                plt.show()
        

    #p_values = list(results.pvalues[idx_params]) + [p_f]
    pvalues = (results.pvalues)
    
    if type(pvalues) != np.ndarray:
        pvalues = pvalues.values
    # ChronoScope fix: idx_params (sized 2N+1 for the F-test above) runs one position too far and
    # includes the last coefficient of the common rhythm; p params is taken over the 2N
    # condition x rhythm terms listed in the comments above (idx_params = [3, 4] for n = 1, ...).
    idx_p = idx_params if non_rhythmic else idx_params[:-1]
    p_params = np.nanmin(pvalues[idx_p.astype(int)])

    # eval rhythm parameters
    n_points = 1000
    max_P = max(2*period1, 2*period2)
    X_full = np.linspace(0, max_P, n_points)
    X_fit_full = generate_independents_compare(X_full, X_full, n_components1 = n_components1, period1 = period1, n_components2 = n_components2, period2 = period2, lin_comp= lin_comp, remove_lin_comp=True)
    H_i = X_fit_full[:,-1]
    locs = H_i == 0

    Y_fit_full1 = results.predict(X_fit_full[locs])
    Y_fit_full2 = results.predict(X_fit_full[~locs])

    # rhythm_params
    rhythm_params1 = evaluate_rhythm_params(X_full, Y_fit_full1, period=period1)
    rhythm_params2 = evaluate_rhythm_params(X_full, Y_fit_full2, period=period2)
    

    rhythm_params = {'amplitude1': rhythm_params1['amplitude'],
                     'amplitude2': rhythm_params2['amplitude'],
                     'd_amplitude': rhythm_params2['amplitude']-rhythm_params1['amplitude'],
                     'acrophase1': rhythm_params1['acrophase'],
                     'acrophase2': rhythm_params2['acrophase'],
                     'd_acrophase': project_acr(rhythm_params2['acrophase']-rhythm_params1['acrophase']),
                     'mesor1': rhythm_params1['mesor'],
                     'mesor2': rhythm_params2['mesor'],
                     'd_mesor': rhythm_params2['mesor']-rhythm_params1['mesor']}

    
    if additional_analysis == "CI1":
        compare_pair_CI(df, test1, test2, n_components = n_components1, period = period1, n_components2 = n_components2, period2 = period2, samples_per_param_CI = samples_per_param_CI, max_samples_CI = max_samples_CI, sampling_type=sampling_type, rhythm_params=rhythm_params, parameters_to_analyse = parameters_to_analyse, parameters_angular = parameters_angular)
    elif additional_analysis == "bootstrap1":
        compare_pair_bootstrap(df, test1, test2, n_components = n_components1, period = period1, n_components2 = n_components2, period2=period2, rhythm_params=rhythm_params, bootstrap_size=bootstrap_size, bootstrap_type=bootstrap_type, parameters_to_analyse = parameters_to_analyse, parameters_angular = parameters_angular)
    elif additional_analysis == "CI2":
        eval_params_diff_CI(X_full, X_fit_full, locs, results, rhythm_params = rhythm_params, samples_per_param = samples_per_param_CI, max_samples = max_samples_CI, k = len(X), sampling_type=sampling_type, parameters_to_analyse = parameters_to_analyse, parameters_angular = parameters_angular, period1=period1, period2=period2)
    elif additional_analysis == "bootstrap2":
        eval_params_diff_bootstrap(X, X_fit, X_full, X_fit_full, Y, model_type, locs, rhythm_params = rhythm_params, bootstrap_size = bootstrap_size, bootstrap_type = bootstrap_type, parameters_to_analyse = parameters_to_analyse, parameters_angular = parameters_angular, period1=period1, period2=period2)    
        # ChronoScope fix: eval_params_diff_bootstrap stores 'CI(param)'/'p(param)', but the callers
        # (compare_pairs_limo) read 'CI(d_param)'/'p(d_param)', as written by the other analyses.
        for param in parameters_to_analyse:
            rhythm_params[f'CI(d_{param})'] = rhythm_params.pop(f'CI({param})')
            rhythm_params[f'p(d_{param})'] = rhythm_params.pop(f'p({param})')
    elif additional_analysis == "":
        pass
    else:
        print("Invalid option")          

    if additional_analysis:  
        for param in parameters_to_analyse:
            d_param = rhythm_params2[param] - rhythm_params1[param]
            if param in parameters_angular:
                d_param = project_acr(d_param)
                rhythm_params[f'd_{param}'] = d_param    
        
  
    # ChronoScope fix (pandas >= 3): results.params is a labelled Series, so it is indexed by
    # position explicitly (pandas 3 removed the positional fallback of Series[int array]).
    return (p_overall, p_params, p_f, np.asarray(results.params)[idx_params.astype(int)], results, rhythm_params)
