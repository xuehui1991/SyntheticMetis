import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts

from scipy.special import comb
from IPython.display import Markdown, display

np.random.seed(7)

def gen_distribution(dist_type, mu, std, n, pmin=None, pmax=None, show_plot=False):
    if dist_type not in ['normal', 'gamma']:
        raise ValueError('dist_type must be in ["normal", "gamma"]')
    elif dist_type == 'normal':
        x = np.random.normal(mu, std, n)
        pmin = pmin or min(x)
        pmax = pmax or max(x)
        x_tick = np.linspace(pmin, pmax, n)
        x_pdf = sts.norm.pdf(x_tick, mu, std)
    else:
        shape = (mu/std)**2
        scale = mu/shape
        x = np.random.gamma(shape, scale, n)
        pmin = pmin or min(x)
        pmax = pmax or max(x)
        x_tick = np.linspace(pmin, pmax, n)
        x_pdf = sts.gamma.pdf(x_tick, a=shape, scale=scale)
    reverse = np.random.randint(0, 2)
    if reverse==1:
        x = pmax - x + pmin
        x_pdf = x_pdf[::-1]
    if show_plot:
        plt.hist(x, bins=25, density=True, alpha=0.6, color='darkcyan')
        plt.plot(x_tick, x_pdf, 'black', linewidth=2)
        plt.xlim((pmin, pmax))
        plt.show()
    return x, x_tick, x_pdf

def generate_monomial(param_list, dist, n_per_context, ci_dist):
    for j, m in enumerate(param_list):
        m_rnd = np.random.choice(range(len(dist[m]['tick'])), n_per_context, replace=True)
        m_tick = np.array([dist[m]['tick'][x] for x in m_rnd])
        m_v = [dist[m]['pdf'][x] for x in m_rnd]
        m_adj = np.random.choice(ci_dist, n_per_context, replace=True)
        m_v = m_v + m_adj
        if j==0:
            reward_terms = m_v.reshape(len(m_v), 1)
            par_values = m_tick.reshape(len(m_tick), 1)
        else:
            reward_terms = np.append(reward_terms, m_v.reshape(len(m_v), 1), 1)
            par_values = np.append(par_values, m_tick.reshape(len(m_tick), 1), 1)
    return reward_terms, par_values

def add_interactions(reward_terms):
    for t in inter_terms:
        t_v = np.multiply(reward_terms[:,t[0]], reward_terms[:,t[1]])
        reward_terms = np.append(reward_terms, t_v.reshape(len(t_v), 1), 1)
    return reward_terms

def rescale_reward(s, reward_scale):
    reward_rescale = (s-s.min())/(s.max()-s.min()) * (reward_scale[1]-reward_scale[0]) + reward_scale[0]
    return reward_rescale
    
def combine_elements(reward_terms, coefficients, c, par_values, reward_range):
    reward_terms = np.multiply(reward_terms, coefficients)
    reward_sum = np.sum(reward_terms, 1)
    reward_sum_rescale = rescale_reward(reward_sum, reward_range)
    con_values = [c]*len(reward_sum)
    num_values = np.concatenate([par_values, 
                             reward_sum.reshape(len(reward_sum), 1), 
                             reward_sum_rescale.reshape(len(reward_sum_rescale), 1)], axis=1).tolist()
    c_data = [x+y for x, y in zip(con_values, num_values)]
    return c_data, num_values

def plot_1d_param_reward(param_dist):
    plt.figure(figsize=(15,2))
    pn = len(param_dist)
    i = 1
    for p, pv in param_dist.items():
        subpn = int('1{0}{1}'.format(pn, i))
        ax = plt.subplot(subpn)
        ax.plot(list(pv['tick']), list(pv['pdf']), 'black', linewidth=2)
        ax.fill_between(list(pv['tick']), 0, list(pv['pdf']), facecolor='grey', alpha=0.3)
        ax.set_title(p)
        ax.set_xlabel('parameter value')
        ax.set_ylabel('reward')
        i = i +1
    plt.show()
    
def plot_2d_paris(num_values, param_list, inter_terms, round_to=0.1, cmap='viridis_r'):
    plt.figure(figsize=(15,3))
    plot_data = pd.DataFrame(num_values, columns=param_list+['reward', 'reward_rescale'])
    for p in param_list:
        plot_data[p] = plot_data[p]//round_to*round_to
    for i, t in enumerate(inter_terms):
        df_grid = plot_data.groupby([param_list[t[0]], param_list[t[1]]]).agg({'reward_rescale': 'mean'}).unstack(0)
        df_grid.columns = df_grid.columns.droplevel(0)
        df_grid.fillna(method='ffill', inplace=True)
        subpn = int('1{0}{1}'.format(len(inter_terms), i+1))
        ax = plt.subplot(subpn)    
        ax.pcolor(df_grid.columns, df_grid.index, df_grid, cmap=cmap)
        ax.set_xlabel(df_grid.columns.name)
        ax.set_ylabel(df_grid.index.name)
        ax.set_title('Reward on {0} vs {1}'.format(df_grid.columns.name, df_grid.index.name))
    plt.show()

# ========================
# Preparation
# ========================

# Contexts 
contexts = {'platform': ['Mac', 'Windows'], 
            'network': ['wifi', 'wired'], 
            'country': ['US', 'CA']}
unique_contexts = [list(x) for x in itertools.product(*contexts.values())]

# Parameter statistics
params = {'x': {'mean': 1, 'min': 0, 'max': 4, 'std_range': [0.1, 1.1]},
          'y': {'mean': 1, 'min': 0, 'max': 3, 'std_range': [0.1, 1.1]}, 
          'z': {'mean': 1, 'min': 0, 'max': 2, 'std_range': [0.1, 1.1]}
         }

# Initialization
data = []
dist = {}
reward_range = [0.05, 0.35]
coefficient_range = [0.1, 1]
interaction2 = True
known_n = False

# Confidence Interval statistics
ci_mean = 0
ci_std = 0.1
ci_dist = gen_distribution('normal', ci_mean, ci_std, 5000)[0]

# N for each unique context
if known_n:
    n_per_context = 20000
else:
    ci_diff = 0.001
    ci_mult = 1.96
    n_per_context = int((ci_mult*ci_std/ci_diff)**2//500*500)

# ========================
# Generate data
# ========================
param_list = list(params.keys())
df_cols = list(contexts.keys()) + param_list + ['reward', 'reward_rescale']
plot_pairs = [x for x in itertools.combinations(range(len(param_list)), 2)]
for i, c in enumerate(unique_contexts):
    display(Markdown('**[{0}/{1}] Generating {2} data points for context {3} ...**'.format(i+1, len(unique_contexts), n_per_context, c)))
    # [1] Generate Distributions
    n_dist = 5000
    dist[i] = {}
    for p, pv in params.items():
        pmu, pmin, pmax = pv['mean'], pv['min'], pv['max']
        pstd = np.random.uniform(pv['std_range'][0], pv['std_range'][1])
        dist[i][p] = {}
        dist[i][p]['raw'], dist[i][p]['tick'], dist[i][p]['pdf'] = gen_distribution('gamma', pmu, pstd, n_dist, pmin=pmin, pmax=pmax)
    for p in params.keys():
        dist[i][p]['pdf'] = rescale_reward(dist[i][p]['pdf'], [0, 1])
        dist[i][p]['pdf'] = 1 - dist[i][p]['pdf']
        dist[i][p]['pdf'] = rescale_reward(dist[i][p]['pdf'], reward_range)
    #plot_1d_param_reward(dist[i])
    
    # [2] Coefficients
    if interaction2:
        inter_terms = [x for x in itertools.combinations(range(len(param_list)), 2)]
    else:
        inter_terms = []
    n_coef = len(param_list) + len(inter_terms)
    coefficients = np.random.uniform(coefficient_range[0], coefficient_range[1], n_coef)
    # [3] Generate random data
    reward_formula = param_list + ['{0}{1}'.format(param_list[x[0]], param_list[x[1]]) for x in inter_terms]
    reward_terms, par_values = generate_monomial(param_list, dist[i], n_per_context, ci_dist)
    reward_terms = add_interactions(reward_terms)
    c_data, num_values = combine_elements(reward_terms, coefficients, c, par_values, reward_range)
    data = data + c_data
    reward_equation = 'reward = {0}'.format(
        ' + '.join(['{0}{1}'.format(round(coefficients[i], 4), reward_formula[i]) for i in range(len(coefficients))]))
    display(Markdown('* {0}'.format(reward_equation)))
    
    # [4] Plot 2D
    #plot_2d_paris(num_values, param_list, inter_terms, round_to=0.05)

df = pd.DataFrame(data, columns = df_cols)

# Save file
df = df.sample(frac=1)
display(df.head())
df.to_csv(r'/data/data0/xuehui/workspace/synthetic_metis/simulation_data.csv')