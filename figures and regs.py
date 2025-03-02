import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
from patsy import dmatrices
import numpy as np
from linearmodels.panel import PanelOLS
from scipy.stats.mstats import winsorize
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.utils import shuffle
from tabulate import tabulate
import seaborn as sns
import pyhdfe

# load data
fund = pd.read_excel('fund_clean.xlsx')
company = pd.read_excel('company_clean.xlsx')

# winsorize
def winsorize_series(series, limits=(0.05, 0.05)):  
    return pd.Series(winsorize(series, limits=limits, nan_policy='omit'))  

cols_to_winsorize = ['Beta','SiteVisit',"Duration", "qReturn", "ValueGrowth", "TNA", "YrsExp"]
for col in cols_to_winsorize:
    fund[col] = winsorize_series(fund[col])
company['NAV'] = winsorize_series(company['NAV'])
company['VisitCount'] = winsorize_series(company['VisitCount'])
company['NumFunds'] = winsorize_series(company['NumFunds'])

# MgmtFee
def clean_mgmt_fee(x):
    x = str(x).strip()  
    if "%" in x:
        try:
            return float(x.replace("%", ""))  
        except ValueError:
            return np.nan  
    return x  

fund["MgmtFee"] = fund["MgmtFee"].apply(clean_mgmt_fee)

# Parallel Trend Graphs
## Beta and BetaRnk
treatment_group = fund[fund['Treat'] == 1]
control_group = fund[fund['Treat'] == 0]
average_beta_treatment = treatment_group.groupby('Time')['Beta'].mean().reset_index()
average_beta_control = control_group.groupby('Time')['Beta'].mean().reset_index()
fig, ax = plt.subplots(figsize=(8, 5))  
ax.plot(pd.to_datetime(average_beta_treatment['Time']), average_beta_treatment['Beta'],
        label='Treatment Group', color='#1f77b4', linewidth=2, linestyle='-')

ax.plot(pd.to_datetime(average_beta_control['Time']), average_beta_control['Beta'],
        label='Control Group', color='#ff7f0e', linewidth=2, linestyle='--')

policy_date = pd.Timestamp('2022-09-30')
ax.axvline(policy_date, color='red', linestyle=':', lw=2, label='Policy Change')
ax.text(policy_date, ax.get_ylim()[1] * 0.95, ' Policy Change ', 
        color='red', fontsize=10, verticalalignment='top', horizontalalignment='left',
        bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.3'))

plt.xlabel('Quarter', fontsize=12, fontweight='bold')
plt.ylabel('Average Beta', fontsize=12, fontweight='bold')
plt.title('Parallel Trend: Control vs. Treatment Groups', fontsize=14, fontweight='bold')

ax.xaxis.set_major_locator(mdates.YearLocator())  # 每年一个主刻度
ax.xaxis.set_minor_locator(mdates.MonthLocator((3, 6, 9, 12)))  # 每季度一个次刻度
ax.xaxis.set_major_formatter(mdates.DateFormatter("\n%Y"))  # 主刻度格式
ax.xaxis.set_minor_formatter(mdates.DateFormatter("%b"))  # 次刻度格式（月份）

plt.setp(ax.get_xticklabels(), rotation=0, ha='center')  # 调整 X 轴刻度对齐
plt.legend(fontsize=10, loc='best', frameon=True)  # 美化图例
plt.grid(visible=True, linestyle='--', linewidth=0.5, alpha=0.7)  # 透明度提升可读性

plt.savefig("figures and tables\parallel_trend_beta.png", dpi=300, bbox_inches='tight')

average_beta_treatment = treatment_group.groupby('Time')['BetaRnk'].mean().reset_index()
average_beta_control = control_group.groupby('Time')['BetaRnk'].mean().reset_index()
fig, ax = plt.subplots(figsize=(8, 5))  
ax.plot(pd.to_datetime(average_beta_treatment['Time']), average_beta_treatment['BetaRnk'],
        label='Treatment Group', color='#1f77b4', linewidth=2, linestyle='-')

ax.plot(pd.to_datetime(average_beta_control['Time']), average_beta_control['BetaRnk'],
        label='Control Group', color='#ff7f0e', linewidth=2, linestyle='--')

policy_date = pd.Timestamp('2022-09-30')
ax.axvline(policy_date, color='red', linestyle=':', lw=2, label='Policy Change')
ax.text(policy_date, ax.get_ylim()[1] * 0.95, ' Policy Change ', 
        color='red', fontsize=10, verticalalignment='top', horizontalalignment='left',
        bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.3'))

plt.xlabel('Quarter', fontsize=12, fontweight='bold')
plt.ylabel('Average BetaRnk', fontsize=12, fontweight='bold')
plt.title('Parallel Trend: Control vs. Treatment Groups', fontsize=14, fontweight='bold')

ax.xaxis.set_major_locator(mdates.YearLocator())  # 每年一个主刻度
ax.xaxis.set_minor_locator(mdates.MonthLocator((3, 6, 9, 12)))  # 每季度一个次刻度
ax.xaxis.set_major_formatter(mdates.DateFormatter("\n%Y"))  # 主刻度格式
ax.xaxis.set_minor_formatter(mdates.DateFormatter("%b"))  # 次刻度格式（月份）

plt.setp(ax.get_xticklabels(), rotation=0, ha='center')  # 调整 X 轴刻度对齐
plt.legend(fontsize=10, loc='best', frameon=True)  # 美化图例
plt.grid(visible=True, linestyle='--', linewidth=0.5, alpha=0.7)  # 透明度提升可读性

plt.savefig("figures and tables\parallel_trend_betarnk.png", dpi=300, bbox_inches='tight')


## Turnover
average_beta_treatment = treatment_group.groupby('Time')['Turnover'].mean().reset_index()
average_beta_control = control_group.groupby('Time')['Turnover'].mean().reset_index()
fig, ax = plt.subplots(figsize=(8, 5))  
ax.plot(pd.to_datetime(average_beta_treatment['Time']), average_beta_treatment['Turnover'],
        label='Treatment Group', color='#1f77b4', linewidth=2, linestyle='-')

ax.plot(pd.to_datetime(average_beta_control['Time']), average_beta_control['Turnover'],
        label='Control Group', color='#ff7f0e', linewidth=2, linestyle='--')

policy_date = pd.Timestamp('2022-09-30')
ax.axvline(policy_date, color='red', linestyle=':', lw=2, label='Policy Change')
ax.text(policy_date, ax.get_ylim()[1] * 0.95, ' Policy Change ', 
        color='red', fontsize=10, verticalalignment='top', horizontalalignment='left',
        bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.3'))

plt.xlabel('Quarter', fontsize=12, fontweight='bold')
plt.ylabel('Average Beta', fontsize=12, fontweight='bold')
plt.title('Parallel Trend: Control vs. Treatment Groups', fontsize=14, fontweight='bold')

ax.xaxis.set_major_locator(mdates.YearLocator())  # 每年一个主刻度
ax.xaxis.set_minor_locator(mdates.MonthLocator((3, 6, 9, 12)))  # 每季度一个次刻度
ax.xaxis.set_major_formatter(mdates.DateFormatter("\n%Y"))  # 主刻度格式
ax.xaxis.set_minor_formatter(mdates.DateFormatter("%b"))  # 次刻度格式（月份）

plt.setp(ax.get_xticklabels(), rotation=0, ha='center')  # 调整 X 轴刻度对齐
plt.legend(fontsize=10, loc='best', frameon=True)  # 美化图例
plt.grid(visible=True, linestyle='--', linewidth=0.5, alpha=0.7)  # 透明度提升可读性

plt.savefig("figures and tables\parallel_trend_turnover.png", dpi=300, bbox_inches='tight')

## SiteVisit
treatment_group = company[company['Treat'] == 1]
control_group = company[company['Treat'] == 0]
average_beta_treatment = treatment_group.groupby('Time')['VisitCount'].mean().reset_index()
average_beta_control = control_group.groupby('Time')['VisitCount'].mean().reset_index()
fig, ax = plt.subplots(figsize=(8, 5))  
ax.plot(pd.to_datetime(average_beta_treatment['Time']), average_beta_treatment['VisitCount'],
        label='Treatment Group', color='#1f77b4', linewidth=2, linestyle='-')

ax.plot(pd.to_datetime(average_beta_control['Time']), average_beta_control['VisitCount'],
        label='Control Group', color='#ff7f0e', linewidth=2, linestyle='--')

policy_date = pd.Timestamp('2022-09-30')
ax.axvline(policy_date, color='red', linestyle=':', lw=2, label='Policy Change')
ax.text(policy_date, ax.get_ylim()[1] * 0.95, ' Policy Change ', 
        color='red', fontsize=10, verticalalignment='top', horizontalalignment='left',
        bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.3'))

plt.xlabel('Quarter', fontsize=12, fontweight='bold')
plt.ylabel('Average Beta', fontsize=12, fontweight='bold')
plt.title('Parallel Trend: Control vs. Treatment Groups', fontsize=14, fontweight='bold')

ax.xaxis.set_major_locator(mdates.YearLocator())  # 每年一个主刻度
ax.xaxis.set_minor_locator(mdates.MonthLocator((3, 6, 9, 12)))  # 每季度一个次刻度
ax.xaxis.set_major_formatter(mdates.DateFormatter("\n%Y"))  # 主刻度格式
ax.xaxis.set_minor_formatter(mdates.DateFormatter("%b"))  # 次刻度格式（月份）

plt.setp(ax.get_xticklabels(), rotation=0, ha='center')  # 调整 X 轴刻度对齐
plt.legend(fontsize=10, loc='best', frameon=True)  # 美化图例
plt.grid(visible=True, linestyle='--', linewidth=0.5, alpha=0.7)  # 透明度提升可读性

plt.savefig("figures and tables\parallel_trend_VisitCount.png", dpi=300, bbox_inches='tight')

# coefficient distribution in the placebo test.
## Beta and BeteRnk
fund.set_index(["MasterFundCode", "Time"], inplace=True)
fund['Beta'] = pd.to_numeric(fund['Beta'], errors='coerce')
fund['Treat'] = pd.to_numeric(fund['Treat'], errors='coerce')

coefficients = []
num_trials = 1000  # 设定随机试验次数

for _ in range(num_trials):
    fund['Shuffled_Treat'] = shuffle(fund['Treat']).values  # 随机打乱 Treatment
    fund['Shuffled_Treat'] = fund['Shuffled_Treat'].astype(float)  
    model = PanelOLS(fund[['Beta']], fund[['Shuffled_Treat']], entity_effects=True)
    result = model.fit()
    if 'Shuffled_Treat' in result.params:
        coefficients.append(result.params['Shuffled_Treat'])
    else:
        coefficients.append(np.nan)
coefficients = [c for c in coefficients if not np.isnan(c)]
mean_coef = np.mean(coefficients)
std_coef = np.std(coefficients)
plt.figure(figsize=(6, 5))
sns.kdeplot(coefficients, color='dodgerblue', linewidth=2)
plt.axvline(mean_coef, color='red', linestyle='dashed', linewidth=1.5)  # 标记均值
plt.axvline(mean_coef + std_coef, color='green', linestyle='dashed', label=f'+1 Std Dev')
plt.axvline(mean_coef - std_coef, color='green', linestyle='dashed', label=f'-1 Std Dev')
plt.text(mean_coef, max(plt.ylim())*0.9, f"Mean: {mean_coef:.4f}", color='red')
plt.title("Distribution of coefficients from placebo tests")
plt.xlabel("Coefficient")
plt.ylabel("Density")
plt.grid(True)
plt.savefig("figures and tables\placebo_Beta.png", dpi=300, bbox_inches='tight')

for _ in range(num_trials):
    fund['Shuffled_Treat'] = shuffle(fund['Treat']).values  # 随机打乱 Treatment
    fund['Shuffled_Treat'] = fund['Shuffled_Treat'].astype(float)  
    model = PanelOLS(fund[['BetaRnk']], fund[['Shuffled_Treat']], entity_effects=True)
    result = model.fit()
    if 'Shuffled_Treat' in result.params:
        coefficients.append(result.params['Shuffled_Treat'])
    else:
        coefficients.append(np.nan)
coefficients = [c for c in coefficients if not np.isnan(c)]
mean_coef = np.mean(coefficients)
std_coef = np.std(coefficients)
plt.figure(figsize=(6, 5))
sns.kdeplot(coefficients, color='dodgerblue', linewidth=2)
plt.axvline(mean_coef, color='red', linestyle='dashed', linewidth=1.5)  # 标记均值
plt.axvline(mean_coef + std_coef, color='green', linestyle='dashed', label=f'+1 Std Dev')
plt.axvline(mean_coef - std_coef, color='green', linestyle='dashed', label=f'-1 Std Dev')
plt.text(mean_coef, max(plt.ylim())*0.9, f"Mean: {mean_coef:.4f}", color='red')
plt.title("Distribution of coefficients from placebo tests")
plt.xlabel("Coefficient")
plt.ylabel("Density")
plt.grid(True)
plt.savefig("figures and tables\placebo_BetaRnk.png", dpi=300, bbox_inches='tight')

## Turnover
for _ in range(num_trials):
    fund['Shuffled_Treat'] = shuffle(fund['Treat']).values  # 随机打乱 Treatment
    fund['Shuffled_Treat'] = fund['Shuffled_Treat'].astype(float)  
    model = PanelOLS(fund[['Turnover']], fund[['Shuffled_Treat']], entity_effects=True)
    result = model.fit()
    if 'Shuffled_Treat' in result.params:
        coefficients.append(result.params['Shuffled_Treat'])
    else:
        coefficients.append(np.nan)
coefficients = [c for c in coefficients if not np.isnan(c)]
mean_coef = np.mean(coefficients)
std_coef = np.std(coefficients)
plt.figure(figsize=(6, 5))
sns.kdeplot(coefficients, color='dodgerblue', linewidth=2)
plt.axvline(mean_coef, color='red', linestyle='dashed', linewidth=1.5)  # 标记均值
plt.axvline(mean_coef + std_coef, color='green', linestyle='dashed', label=f'+1 Std Dev')
plt.axvline(mean_coef - std_coef, color='green', linestyle='dashed', label=f'-1 Std Dev')
plt.text(mean_coef, max(plt.ylim())*0.9, f"Mean: {mean_coef:.4f}", color='red')
plt.title("Distribution of coefficients from placebo tests")
plt.xlabel("Coefficient")
plt.ylabel("Density")
plt.grid(True)
plt.savefig("figures and tables\placebo_Turnover.png", dpi=300, bbox_inches='tight')

## SiteVisit
company.set_index(["FundCompanyName", "Time"], inplace=True)
company['VisitCount'] = pd.to_numeric(company['VisitCount'], errors='coerce')
company['Treat'] = pd.to_numeric(company['Treat'], errors='coerce')

coefficients = []
num_trials = 1000  # 设定随机试验次数

for _ in range(num_trials):
    company['Shuffled_Treat'] = shuffle(company['Treat']).values  # 随机打乱 Treatment
    company['Shuffled_Treat'] = company['Shuffled_Treat'].astype(float)  
    model = PanelOLS(company[['VisitCount']], company[['Shuffled_Treat']], entity_effects=True)
    result = model.fit()
    if 'Shuffled_Treat' in result.params:
        coefficients.append(result.params['Shuffled_Treat'])
    else:
        coefficients.append(np.nan)
coefficients = [c for c in coefficients if not np.isnan(c)]
mean_coef = np.mean(coefficients)
std_coef = np.std(coefficients)
plt.figure(figsize=(6, 5))
sns.kdeplot(coefficients, color='dodgerblue', linewidth=2)
plt.axvline(mean_coef, color='red', linestyle='dashed', linewidth=1.5)  # 标记均值
plt.axvline(mean_coef + std_coef, color='green', linestyle='dashed', label=f'+1 Std Dev')
plt.axvline(mean_coef - std_coef, color='green', linestyle='dashed', label=f'-1 Std Dev')
plt.text(mean_coef, max(plt.ylim())*0.9, f"Mean: {mean_coef:.4f}", color='red')
plt.title("Distribution of coefficients from placebo tests")
plt.xlabel("Coefficient")
plt.ylabel("Density")
plt.grid(True)
plt.savefig("figures and tables\placebo_VisitCount.png", dpi=300, bbox_inches='tight')

# Summary Statistics
fund_variables = ["Beta",'BetaRnk', "Turnover", "Duration", "TNA", "qReturn", 
                  "MgmtFee", "Gender", "YrsExp", "ValueGrowth"]
fund_summary = fund[fund_variables].describe().T  

company_variables = ["VisitCount", "Post", "Treat", "NAV", "NumFunds"]
company_summary = company[company_variables].describe().T

fund_summary = fund_summary[["count", "mean", "std", "min", "max"]]
company_summary = company_summary[["count", "mean", "std", "min", "max"]]

fund_summary.index.name = "Fund-level data"
company_summary.index.name = "Management Company Level Data"

summary_table = pd.concat([fund_summary, company_summary])
summary_table.columns = ["Obs.", "Mean", "Sd.", "Min", "Max"]

table_str = tabulate(summary_table, headers="keys", tablefmt="grid", floatfmt=".3f")

with open("figures and tables/summary_statistics.txt", "w", encoding="utf-8") as f:
    f.write("""Panel B: Summary Statistics\n\n""")
    f.write(table_str)

# Regs
fund_fundpanel = fund.set_index(['MasterFundCode', 'Time'])
fund_companypanel = fund.set_index(['FundCompanyName', 'Time'])
## BetaRnk and Beta
Y = fund_companypanel['Beta']
X = fund_companypanel[[ 'Post_Treat', 'Duration','qReturn','ValueGrowth', 'TNA', 'MgmtFee', 'Gender', 'YrsExp']]
X = sm.add_constant(X)
model1 = PanelOLS(Y, X, entity_effects=True, time_effects=True)
reg1 = model1.fit(cov_type='clustered', cluster_entity=True)

Y = fund_fundpanel['Beta']
X = fund_fundpanel[['Post_Treat', 'Duration','qReturn','ValueGrowth', 'TNA', 'MgmtFee', 'Gender', 'YrsExp']]
X = sm.add_constant(X)
model2 = PanelOLS(Y, X, entity_effects=True, time_effects=True)
reg2 = model2.fit(cov_type='clustered', cluster_entity=True)

with open("figures and tables/regression_Beta.txt", "w") as f:
    f.write(reg1.summary.as_text())  
    f.write("\n\n")
    f.write(reg2.summary.as_text())  

## Turnover
Y = fund_fundpanel['Turnover']
X = fund_fundpanel[[ 'Post_Treat', 'Duration','qReturn','ValueGrowth', 'TNA', 'MgmtFee', 'Gender', 'YrsExp']]
X = sm.add_constant(X)
model1 = PanelOLS(Y, X, entity_effects=True, time_effects=True)
reg1 = model1.fit(cov_type='clustered', cluster_entity=True)

fund_fundpanel['Post_Treat_YrsExp'] = fund_fundpanel['Post_Treat']*fund_fundpanel['YrsExp']
Y = fund_fundpanel['Turnover']
X = fund_fundpanel[[ 'Post_Treat', 'Post_Treat_YrsExp','Duration','qReturn','ValueGrowth', 'TNA', 'MgmtFee', 'Gender', 'YrsExp']]
X = sm.add_constant(X)
model2 = PanelOLS(Y, X, entity_effects=True, time_effects=True)
reg2 = model2.fit(cov_type='clustered', cluster_entity=True)

fund_fundpanel['Post_Treat_Topperformer'] = fund_fundpanel['Post_Treat']*fund_fundpanel['Topperformer']
Y = fund_fundpanel['Turnover']
X = fund_fundpanel[[ 'Post_Treat', 'Post_Treat_Topperformer','Duration','qReturn','ValueGrowth', 'TNA', 'MgmtFee', 'Gender', 'YrsExp']]
X = sm.add_constant(X)
model3 = PanelOLS(Y, X, entity_effects=True, time_effects=True)
reg3 = model3.fit(cov_type='clustered', cluster_entity=True)

with open("figures and tables/regression_Turnover.txt", "w") as f:
    f.write(reg1.summary.as_text())  
    f.write("\n\n")
    f.write(reg2.summary.as_text())
    f.write("\n\n")
    f.write(reg3.summary.as_text())  

## SiteVisit
formula = 'VisitCount ~ Treat_Post + NAV + NumFunds + C(Time)'
model_poisson = smf.poisson(formula, data=company)
result_poisson = model_poisson.fit(cov_type='HC3')
reg1 = result_poisson.summary()

with open("figures and tables/regression_SiteVisit.txt", "w") as f:
    f.write(reg1.as_text())  