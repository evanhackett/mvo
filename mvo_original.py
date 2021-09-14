# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.stats
from matplotlib.ticker import StrMethodFormatter


def asset_def(name, ter=0.0,
              mkt=0.0, smb=0.0, hml=0.0, rmw=0.0, cma=0.0, mom=0.0,
              itt=0.0, ltt=0.0):
    return name, {
        'TER': ter,
        'MKT': mkt,

        'SmB': smb,
        'HmL': hml,
        'RmW': rmw,
        'CmA': cma,

        'MOM': mom,

        'ITT': itt,
        'LTT': ltt,
    }


# return assumptions
# monthly arithmetic return, monthly std
factors = np.array([
    ['TER', -1 / 12.0, 0],
    ['MKT', .53, 4.44],

    # US factor data, 1963 - july 2020
    ['SmB', .21 / 2, 3.02],  # estimate only half the historical factor premia
    ['HmL', .25 / 2, 2.87],  # estimate only half the historical factor premia
    ['RmW', .26 / 2, 2.15],  # estimate only half the historical factor premia
    ['CmA', .26 / 2, 1.99],  # estimate only half the historical factor premia
    ['MOM',     .66 / 4, 4.7],  # estimate only a quarter the historical factor premia

    ['ITT', 1.66 / 12, 5.67 / 12**.5],
    ['LTT', 2.01 / 12, 9.79 / 12**.5],
])

factor_labels = list(factors[:, 0])
factor_mean = factors[:, 1].astype(np.float)
factor_std = factors[:, 2].astype(np.float)

factor_mean = np.array(factor_mean) / 100
factor_std = np.array(factor_std) / 100

print('factor mean & std (monthly)')
print(np.vstack((factor_labels, factor_mean*100, factor_std*100)).transpose())


cash_asset = asset_def('cash')
mkt_asset = asset_def('total stock market', mkt=1)
sp500_asset = asset_def('S&P 500', mkt=1, smb=-.16)

iusv_ff5_asset = asset_def('IUSV (value)', ter=0.04, mkt=.97,  smb=.03, hml=.30, rmw=.15, cma=.21, mom=-0.02)
ijs_ff5_asset = asset_def('IJS (scv)', ter=.25,   mkt=1.04, smb=.86, hml=.21, rmw=.19, cma=.09, mom=-0.05)
dfsvx_ff5_asset = asset_def('DFSVX (scv)', ter=.4,   mkt=1.03, smb=.87, hml=.40, rmw=.14, cma=.08, mom=-0.08)
vfmf_ff5_asset = asset_def('VFMF', ter=.18, mkt=.98, smb=.47, hml=.36, rmw=.16, cma=.12, mom=.19)

itt_asset = asset_def('ITT', ter=0, itt=1)
ltt_asset = asset_def('LTT', ter=0, ltt=1)
tmf_asset = asset_def('TMF', ter=1.1, ltt=3)
upro_asset = asset_def('UPRO', ter=2.0, mkt=3, smb=-.16 * 3)

BOUNDS = (0, 1)
CASH_BOUNDS = (0, 1)
SHORTING_COST = .5 / 100
X_AXIS = (0, .50)   # annualized standard deviation
Y_AXIS = (0, .12)   # annualized return

chosen_assets = [
    cash_asset,
    mkt_asset,
    itt_asset,
    iusv_ff5_asset,
    ijs_ff5_asset,
]

def asset_to_weight_array(asset):
    a = np.zeros(len(factor_labels))
    for k, v in asset[1].items():
        if k in factor_labels:
            a[list(factor_labels).index(k)] = v

    return a


asset_labels = [a[0] for a in chosen_assets]
assets = np.array([
    asset_to_weight_array(asset) for asset in chosen_assets
])


correlations = {
    # collected from FF, A Five-Factor Asset Pricing Model (2014)
    ('MKT', 'SmB'): .28,
    ('MKT', 'HmL'): -.30,
    ('MKT', 'RmW'): -.21,
    ('MKT', 'CmA'): -.39,

    ('SmB', 'HmL'): -.11,
    ('SmB', 'RmW'): -.36,
    ('SmB', 'CmA'): -.11,

    ('HmL', 'RmW'): .08,
    ('HmL', 'CmA'): .70,

    ('RmW', 'CmA'): -.11,

    ('ITT', 'LTT'): .8,
}

correlation_coef = np.identity(len(factor_labels))
for k, v in correlations.items():
    a, b = k
    try:
        a = factor_labels.index(a)
        b = factor_labels.index(b)
        correlation_coef[a, b] = v
        correlation_coef[b, a] = v
    except ValueError:
        pass


class Model:
    def __init__(self):
        self.init_weights = np.zeros(len(asset_labels), dtype=np.float)
        self.init_weights[0] = 1
        self.init_weights[:len(asset_labels)] = 1.0/len(asset_labels)

    def extract_weights(self, weights):
        return weights[:len(weights) / 2]

    def bounds(self):
        b = [BOUNDS for i in range(len(self.init_weights))]
        b[0] = CASH_BOUNDS
        return b

    def extract_weights(self, weights):
        return weights

    def get_mean(self, weights):
        factor_loadings = np.dot(weights, assets)
        mean = np.dot(factor_loadings, factor_mean)
        return mean

    def mean_and_std(self, weights):
        factor_loadings = np.dot(weights, assets)
        mean = np.dot(factor_loadings, factor_mean)
        std = np.array(factor_loadings * factor_std, ndmin=2)
        covariance_matrix = np.matmul(std.transpose(), std) * correlation_coef
        variance = np.sum(covariance_matrix)
        std = np.sqrt(variance)
        return mean, std

model = Model()
def find_weights(expected_std):

    w_bound = model.bounds()
    constraints = [
        ({'type': 'eq', 'fun': lambda w: sum(model.extract_weights(w)) - 1.}),      # weights must sum to 1
        ({'type': 'eq', 'fun': lambda w: model.mean_and_std(w)[1] - expected_std})  # std must be equal to expected_std
    ]

    w = scipy.optimize.minimize(
        lambda w: -model.get_mean(w) - np.sum(np.clip(w, -np.inf, 0)) * SHORTING_COST,
        model.init_weights,
        method='SLSQP',
        bounds=w_bound,
        constraints=constraints,
        tol=0.00000000005
    )

    return w.x

X = []
Y = []
asset_ratios = []

for expected_std in np.linspace(X_AXIS[0], X_AXIS[1], 200):
    vars = find_weights(expected_std / 12 **.5)
    weights = model.extract_weights(vars)

    if not (0.999 <= sum(weights) <= 1.001):
        print('fail')
        continue

    asset_ratios.append(weights)
    factor_loadings = np.matmul(weights, assets)
    mean, std = model.mean_and_std(vars)
    # print(expected_std, mean, std, expected_std - std, weights, factor_loadings)
    print("mean: {:.2%}, std: {:.2%}".format(mean * 12, std * 12 ** .5))
    Y.append(mean * 12)
    X.append(std * 12 ** .5)

asset_ratios = np.array(asset_ratios)


def isoeastic_utility(X, Y, gamma):
    return np.array(Y) - .5 * np.power(X, 2) * gamma

def plot_utility_function(X, Y, gamma, label, color):
    U = isoeastic_utility(X, Y, gamma)
    plt.plot(X, U, label=label, color=color)
    max_u = np.max(U)
    mask = max_u == U
    # draw a colored "X" at the maximum
    plt.scatter(np.array(X)[mask], U[mask], color=color, marker='x')



plt.subplots(2, 1, figsize=(8, 7))

plt.subplot('211')
plt.title('efficient frontier, for various coefficients of relative risk aversion')
plt.ylabel('expected utility')
plt.xlabel('risk (annualized stddev)')
plt.grid()
plt.xlim(X_AXIS)
plt.ylim(Y_AXIS)
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0%}'))
plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0%}'))

plot_utility_function(X, Y, 0, u'γ = 0', 'C1')
plot_utility_function(X, Y, 0.5, u'γ = 0.5', 'C2')
plot_utility_function(X, Y, 1, u'γ = 1', 'C3')
plot_utility_function(X, Y, 2, u'γ = 2', 'C4')
plot_utility_function(X, Y, 3, u'γ = 3', 'C5')
plot_utility_function(X, Y, 5, u'γ = 5', 'C6')

plt.text(.30, .005, 'γ = 0 is the efficient frontier\nγ = 1 maximizes geometric growth,\nalso known as the kelly criterion',
    bbox=dict(boxstyle="square",
            ec=(.6, .6, .6),
            fc=(1, 1, 1),
            )
)
plt.legend(loc='upper left')


plt.subplot('212')
plt.title('asset weights')
plt.ylabel('asset weight')
plt.xlabel('risk (annualized stddev)')
plt.grid()
plt.xlim(X_AXIS)
plt.ylim((0, 1))
plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0%}'))

for i in range(len(assets)):
    plt.plot(X, asset_ratios[:, i], label=asset_labels[i])
plt.legend()

plt.subplots_adjust(hspace=.3)
plt.subplots_adjust(left=.1)
plt.subplots_adjust(right=.9)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.08)
plt.show()
