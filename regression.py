import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy.stats import mode
from sklearn import linear_model
from scipy.stats import poisson
from scipy.misc import factorial
from scipy import optimize
from pandas.stats.api import ols

raw_ints = pd.read_csv('C:\Users\Sheridan\Desktop\Dropbox\Insight\My Files\intersectionsbb.csv')
traf = pd.read_csv('C:/Users/Sheridan/Desktop/Dropbox/Insight/My Files/trafficvols.csv')
inj = pd.read_csv('C:\Users\Sheridan\Desktop\Dropbox\Insight\My Files\injuries.csv')
streets = pd.read_csv('C:\Users\Sheridan\Desktop\Dropbox\Insight\My Files\streets.csv')

# Ceremonial class excludes most of Market street, whose complicated intersection/crosswalk patterns cannot be modeled with available data
ints = raw_ints[(raw_ints.intrsctn_type=='INTERSECTIONS') & (raw_ints.BSP_class!='Alley') & (raw_ints.BSP_class!='Ceremonial') & (raw_ints.BSP_class!='Paseo') & (raw_ints.pvtraf > 0) ]
ints['constant'] = 1 # For use in regression
# Some test neighborhoods and intersection types to look at for convenience
# w_nhoods is a regular grid of mostly residential neighborhoods with a few commercial avenues and one major surface road highway
# e_nhoods is the busiest region of the city in terms of pedestrian and vehicle traffic; mostly gridlike, Market street roughly bisects it
w_nhoods=['Parkside','Inner Parkside','Outer Parkside', 'Inner Richmond', 'Central Richmond','Outer Richmond', 'Central Sunset','Inner Sunset', 'Outer Sunset']
e_nhoods=['Financial District North','Financial District South','Downtown - Tenderloin']

# sign indicates traffic signal; fstop: all-way stop; lstop: limited stop; uncont: uncontrolled
ints_hood_w = pd.DataFrame()
for w_nhood in w_nhoods:
	ints_hood_w = ints_hood_w.append(ints[ints.NHOOD == w_nhood])
ints_sig_w = ints_hood_w[ints_hood_w.signal_yn == 1]
ints_fstop_w = ints_hood_w[ints_hood_w.sign_stop_yn == 1]
ints_lstop_w = ints_hood_w[ints_hood_w.limited_stop_yn == 1]
ints_uncont_w = ints_hood_w[(ints_hood_w.limited_stop_yn == 0) & (ints_hood_w.sign_stop_yn == 0) & (ints_hood_w.signal_yn == 0)]

ints_hood_e = pd.DataFrame()
for e_nhood in e_nhoods:
    ints_hood_e = ints_hood_e.append(ints[ints.NHOOD == e_nhood])
ints_sig_e = ints_hood_e[ints_hood_e.signal_yn == 1]
ints_fstop_e = ints_hood_e[ints_hood_e.sign_stop_yn == 1]
ints_lstop_e = ints_hood_e[ints_hood_e.limited_stop_yn == 1]
ints_uncont_e = ints_hood_e[(ints_hood_e.limited_stop_yn == 0) & (ints_hood_e.sign_stop_yn == 0) & (ints_hood_e.signal_yn == 0)]

def holdout(data): # create set of training and testing data via holdout of half intersections randomly
    ints_shuf = data.reindex(np.random.permutation(data.index))
    train = ints_shuf[:len(ints_shuf)/2]
    test = ints_shuf[len(ints_shuf)/2:]
    return train, test

def bilin_regr(data,xvars,plot=True):
    regr=linear_model.LinearRegression()
    X = shape_xvars(data[xvars])
    y = np.asarray(data.pedinj_count)
    regr.fit(X,y)
    print "Coeffs, intercept, score = "
    print regr.coef_, regr.intercept_, regr.score(X,y)
    if plot==True:
        plt.plot(data['pvtraf'],regr.predict(X))
        plt.scatter(data['pvtraf'],data['pedinj_count'])
        plt.show()
    return regr

def shape_xvars(data_xvars): # Refactoring/reshaping columns/variables into format that LinearRegression likes
    X = np.asarray(data_xvars)
    if 'max_sl' in data_xvars.columns:
        for i in range(len(X)):
            X[i] = [a*b for a,b in zip(X[i],[1e-8,1])]
    else:
        X = X * 1e-8
    return X

def regr_test(data):
	train_data = holdout(data)[0]
	test_data = holdout(data)[1]
	regr = bilin_regr(train_data,plot=False)
	plt.plot(test_data['pvtraf'],regr.predict(test_data['pvtraf'].reshape(-1,1)/1e8))
	plt.scatter(test_data['pvtraf'],test_data['pedinj_count'])
	plt.show()

def lrtest(data,xvars): # Likelihood ratio test between input model and null "traffic only" model
    pedinj_count = data['pedinj_count']
    xvars_0 = shape_xvars(data[['pvtraf']])
    xvars_1 = shape_xvars(data[xvars])
    inj_pred_0 = bilin_regr(data,['pvtraf'],plot=False).predict(xvars_0)
    inj_pred_1 = bilin_regr(data,xvars,plot=False).predict(xvars_1)
    LL_0 = np.log(poisson.pmf(pedinj_count,inj_pred_0)).sum()
    LL_1 = np.log(poisson.pmf(pedinj_count,inj_pred_1)).sum()
    TS = -2 * (LL_0 - LL_1)
    print "TS, LL_1, LL_0 = ", TS, LL_1, LL_0

def sq_err(data,xvars):
    actual = data['pedinj_count']
    regr = bilin_regr(data,xvars,plot=False)
    X = shape_xvars(data[xvars])
    predicted = regr.predict(X)
    sq_err = (actual - predicted)**2
    return sq_err

def rss(data,xvars):
    sq_error = sq_err(data,xvars)
    rss = sq_error.sum()
    return rss

def meanse(data,xvars):
    sq_error = sq_err(data,xvars)
    meanse = (sq_error / (len(data) - len(xvars))).sum()
    return meanse

def nmse(data,xvars):
    m = meanse(data,xvars)
    nmse = m / data['pedinj_count'].mean()
    return nmse

def ftest(data,xvars):
    rss1 = rss(data,['pvtraf'])
    rss2 = rss(data,xvars)
    p1 = 1
    p2 = len(xvars)
    n = len(data)
    F = ((rss1 - rss2)/(p2 - p1))/(rss2/(n - p2))
    return F

def sim_data(data,xvars): # simulate Poisson data from model 
    regr = bilin_regr(data,xvars,plot=False)
    X = shape_xvars(data[xvars])
    predicted = regr.predict(X)
    rsim = poisson.rvs(predicted)
    plt.plot(data['pvtraf'],regr.predict(shape_xvars(data[xvars])))
    plt.scatter(data['pvtraf'],rsim)
    plt.show()

def pd_regr(data,xvars,convs): # regression test using pandas
    y = data['pedinj_count']
    X = dict()
    for var,conv in zip(xvars,convs):
        X[var] = data[var]*conv
    regr = ols(y = y, x = X)
    return regr

# Calculating the log-likelihood of a Poisson regression model
# Log(Y) = b0 + b1x1 + b2x2 + ... ---> Y = e^(b0 + b1x1 + b2x2 + ...)
# LL = Sum(y*beta*x - e**[beta*x] - log (y!))

def ll_poisson(betas,data=ints,xvars=['pvtraf']): 
    if 'constant' not in xvars:
        xvars.append('constant')
    X = data[xvars]
    y = data['pedinj_count']
    conv=np.concatenate(([1e-8],np.ones(len(xvars)-1)))
    xpred = np.dot(X*conv,betas)
    negL = -((y*xpred).sum() - np.sum(np.e**(xpred))) # We will minimize this function for betas, so don't need to include log(y) term
    return negL

def ll_min(data,xvars,ftol=1e-3):
    conv=np.concatenate(([1e-8],np.ones(len(xvars)-1)))
    ols_regr = pd_regr(data,xvars,conv)
    b_guess = ols_regr._results['beta']
    ll_opt = optimize.fmin(ll_poisson,b_guess,args=(data,xvars),ftol=ftol)
    return ll_opt