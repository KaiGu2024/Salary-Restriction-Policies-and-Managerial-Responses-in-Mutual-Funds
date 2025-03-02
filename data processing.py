import pandas as pd
import numpy as np

# Load data
Fund=pd.read_excel("fund_main.xlsx", engine='openpyxl')

# Convert data type
Fund['MasterFundCode'] = Fund['MasterFundCode'].astype(str).str.zfill(6)
Fund['IsETF'] = Fund['IsETF'].astype(str)
Fund['IsQDII'] = Fund['IsQDII'].astype(str)
Fund['IsIndexFund'] = Fund['IsIndexFund'].astype(str)
Fund['IsActiveOrPassive'] = Fund['IsActiveOrPassive'].astype(str)
Fund['InceptionDate'] = pd.to_datetime(Fund['InceptionDate'], errors='coerce')

# Select fund before 2018-01-01, and select fund type is '契约型开放式', category is '混合型基金' or '股票型基金', is ETF, is QDII, is index fund, is active fund
Fund=Fund[(Fund['InceptionDate']<='2018-01-01') & (Fund['FundType'] == '契约型开放式') & 
          (Fund['Category'].isin(['混合型基金','股票型基金'])) & (Fund['IsETF'] == '2') &
          (Fund['IsQDII'] == '2') & (Fund['IsIndexFund'] == '2') & (Fund['IsActiveOrPassive'] == '1')
          ]
Fund = Fund[['MasterFundCode','FundCompanyName','InceptionDate','FundCompanyID']]

# convert to panel data
time_range = pd.date_range(start="2019-03-31", end="2024-12-31", freq='Q')
time_df = pd.DataFrame({'Time': time_range})
Fund['key'] = 1
time_df['key'] = 1
Fund_panel = Fund.merge(time_df, on='key').drop(columns=['key'])
Fund_panel['Time'] = pd.to_datetime(Fund_panel['Time'])
Fund_panel = Fund_panel.sort_values(by=['MasterFundCode', 'Time']).reset_index(drop=True)

# label post-treat
treat = pd.read_excel("treat.xlsx", engine='openpyxl')
Fund_label = Fund_panel.merge(treat, on="FundCompanyName", how="left")
Fund_label['Post'] = Fund_label['Time'].apply(lambda x: 1 if x >= pd.to_datetime('2022-9-30') else 0)
Fund_label['Post_Treat'] = Fund_label['Post'] * Fund_label['Treat']
Fund_label = Fund_label.drop_duplicates().reset_index(drop=True)

# calculate Fund log Age
Fund_label['Age'] = np.log((Fund_label['Time'] - Fund_label['InceptionDate']).dt.days/365.0) 

# calcaulate Fund log TNA
TNA_19 = pd.read_excel("TNA_19-23.xlsx", engine='openpyxl') 
TNA_24 = pd.read_excel("TNA_23-24.xlsx") 
TNA_24.rename(columns={'净资产值_NetAss': '期末基金资产净值_NetAssV'}, inplace=True)

TNA = pd.concat([TNA_19, TNA_24]).drop_duplicates()

# 重命名列
TNA.rename(columns={'基金代码_FdCd': 'MasterFundCode',
                    '截止日期_EndDt': 'Time',
                    '期末基金资产净值_NetAssV': 'TNA'}, inplace=True)
TNA['Time'] = pd.to_datetime(TNA['Time'], errors='coerce')
TNA['MasterFundCode'] = TNA['MasterFundCode'].astype(str).str.zfill(6)
Fund_merge = Fund_label.merge(TNA, on=['MasterFundCode', 'Time'], how='left')
Fund_merge['TNA'] = np.log(Fund_merge['TNA'])

# calculate qreturn
qreturn_19 = pd.read_excel("qreturn_19-23.xlsx", engine='openpyxl') 
qreturn_24 = pd.read_excel("qreturn_23-24.xls") 

qreturn = pd.concat([qreturn_19, qreturn_24]).drop_duplicates()
qreturn = qreturn[['基金代码_Fdcd', '截止日期_EndDt', '季收益_按红利再投资日调整_QtrRet_ReInvDt']]
qreturn.rename(columns={'基金代码_Fdcd': 'MasterFundCode',
                    '截止日期_EndDt': 'Time',
                    '季收益_按红利再投资日调整_QtrRet_ReInvDt': 'qReturn'}, inplace=True)
qreturn['Time'] = pd.to_datetime(qreturn['Time'], errors='coerce')
qreturn['MasterFundCode'] = qreturn['MasterFundCode'].astype(str).str.zfill(6)
qreturn = qreturn.drop_duplicates(subset=['MasterFundCode', 'Time'])
Fund_merge1 = Fund_merge.merge(qreturn, on=['MasterFundCode', 'Time'], how='left')

# mgmt fee
fund_fee = pd.read_excel("fund_fee_change.xlsx", engine='openpyxl')
fund_fee = fund_fee[['基金代码_FdCd', '执行日期_ExecDt', '费率描述_ChgRtDec']].rename(
    columns={'基金代码_FdCd': 'MasterFundCode', '执行日期_ExecDt': 'Time', '费率描述_ChgRtDec': 'MgmtFee'}
)
fund_fee['MasterFundCode'] = fund_fee['MasterFundCode'].astype(str).str.zfill(6)
fund_fee['Time'] = pd.to_datetime(fund_fee['Time'], errors='coerce')
fund_fee['Time'] = fund_fee['Time'].dt.to_period("Q").dt.to_timestamp(how="end")
fund_fee['Time'] = fund_fee['Time'].dt.floor('D')
fund_fee['MgmtFee'] = fund_fee['MgmtFee'].astype(str).str.replace('%', '').astype(float)
unique_dates = Fund_merge1[['MasterFundCode', 'Time']].drop_duplicates()
fund_fee_full = unique_dates.merge(fund_fee, on=['MasterFundCode', 'Time'], how='left')
fund_fee_full = fund_fee_full.sort_values(by=['MasterFundCode', 'Time'])
# 进行填充：
# - `ffill()` 向前填充，确保早期数据继承较高费率
# - `bfill()` 向后填充，确保后期数据继承较低费率
fund_fee_full['MgmtFee'] = fund_fee_full.groupby('MasterFundCode')['MgmtFee'].ffill().bfill()
fund_fee_full = fund_fee_full.drop_duplicates(subset=['MasterFundCode', 'Time'])
Fund_merge2 = Fund_merge1.merge(fund_fee_full, on=['MasterFundCode', 'Time'], how='left') 

# merge beta, Reg_1
file_list = ["fund_beta_2019.xlsx", "fund_beta_2020.xlsx", "fund_beta_2021.xlsx",
             "fund_beta_2022.xlsx", "fund_beta_2023.xlsx", "fund_beta_2024.xlsx"]

df_list = []
for file in file_list:
    df = pd.read_excel(file, engine='openpyxl')[['TradingDate', 'Symbol', 'Beta', 'BetaRnk']]
    df_list.append(df)

fund_beta = pd.concat(df_list, ignore_index=True)
fund_beta.rename(columns={'TradingDate': 'Time', 'Symbol': 'MasterFundCode'}, inplace=True)
fund_beta['MasterFundCode'] = fund_beta['MasterFundCode'].astype(str).str.zfill(6)
fund_beta['Time'] = pd.to_datetime(fund_beta['Time'], errors='coerce')
fund_beta['Quarter'] = fund_beta['Time'].dt.to_period('Q')  # 获取季度信息
fund_beta = fund_beta.loc[fund_beta.groupby(['MasterFundCode', 'Quarter'])['Time'].idxmax()]  # 取该季度最大日期
fund_beta['Time'] = fund_beta['Quarter'].dt.to_timestamp(how='end')
fund_beta['Time'] = fund_beta['Time'].dt.floor('D')
fund_beta = fund_beta[['MasterFundCode', 'Time', 'Beta', 'BetaRnk']]
fund_beta  = fund_beta .drop_duplicates(subset=['MasterFundCode', 'Time'])
Fund_merge3 = Fund_merge2.merge(fund_beta, on=['MasterFundCode', 'Time'], how='left')
Fund_merge3.rename(columns={'Age':'Duration'},inplace = True) # 修改列名
Fund_merge3.to_excel('fund_reg_1_clean.xlsx')

# fund manager
fund_manager = pd.read_excel("fund_manager.xlsx", engine='openpyxl')

fund_manager.rename(columns={
    '基金代码_FdCd': 'MasterFundCode',
    '职位名称_PostNm': 'Position',
    '姓名_Nm': 'Name',
    '性别()_Gender': 'Gender',
    '证券从业经历(年)_ExpcTime': '证券业年限',
    '证券从业日期_PrctcDt': '证券从业日期',
    '到任日期_AccssDt': 'AccssDate',
    '离职日期_OffDt': '离职日期',
    '任职天数()_NumDayWork': '任职天数',
    '任职期间基金净值增长率_Prfmc': 'ValueGrowth'
}, inplace=True)

fund_manager['MasterFundCode'] = fund_manager['MasterFundCode'].astype(str).str.zfill(6)

date_cols = ['证券从业日期', 'AccssDate', '离职日期']
for col in date_cols:
    fund_manager[col] = pd.to_datetime(fund_manager[col], errors='coerce')

time_range = pd.date_range(start="2019-01-01", end="2024-12-31", freq='Q').to_frame(index=False, name='Time')
time_range['Time'] = time_range['Time'].dt.floor('D')  

unique_funds = fund_manager[['MasterFundCode']].drop_duplicates()
full_time_df = unique_funds.merge(time_range, how='cross')
fund_manager_full = full_time_df.merge(fund_manager, on='MasterFundCode', how='left')

fund_manager_full = fund_manager_full[
    (fund_manager_full['Time'] >= fund_manager_full['AccssDate'].fillna(pd.Timestamp('2000-01-01'))) & 
    (fund_manager_full['Time'] < fund_manager_full['离职日期'].fillna(pd.Timestamp('2025-12-31')))
]  # 筛选当时在任管理者
## 选择资深管理者作为主管理者
fund_manager_full = fund_manager_full.sort_values(by=['MasterFundCode', 'Time', '证券业年限'], ascending=[True, True, False])
fund_manager_full = fund_manager_full.drop_duplicates(subset=['MasterFundCode', 'Time'], keep='first') 
fund_manager_full['Turnover'] = np.where(
    (fund_manager_full['Time'] == fund_manager_full['离职日期'].dt.to_period('Q').dt.to_timestamp(how="end").dt.floor('D')),
    1, 0
)
fund_manager_full['YrsExp'] = (fund_manager_full['Time'] - fund_manager_full['证券从业日期']).dt.days / 365.00
top_25_threshold = fund_manager_full['ValueGrowth'].quantile(0.75)
fund_manager_full['Topperformer'] = np.where(fund_manager_full['ValueGrowth'] >= top_25_threshold, 1, 0)

fund_manager_final = fund_manager_full[['MasterFundCode', 'Time','Name', 'Gender', 'ValueGrowth', 'Turnover', 'YrsExp', 'Topperformer']]
fund_manager_final = fund_manager_final.copy()  # 确保是副本
fund_manager_final.loc[:, 'Gender'] = fund_manager_final['Gender'].replace({1: 0, 2: 1})
fund_cleaned = Fund_merge3.merge(fund_manager_final, on=['MasterFundCode', 'Time'], how='left')
fund_cleaned = pd.read_excel('fund_cleaned.xlsx')
fund_cleaned = fund_cleaned.sort_values(by=["MasterFundCode", "Time"]).copy()

fund_cleaned["Turnover"] = (fund_cleaned.groupby("MasterFundCode")["Name"]
                            .transform(lambda x: x.ne(x.shift(1)).astype(int))) # ensure Turnover labelled right
first_rows = fund_cleaned.groupby("MasterFundCode").apply(lambda x: x.index.min())
fund_cleaned.loc[first_rows, "Turnover"] = 0
fund_cleaned["Turnover"] = fund_cleaned["Turnover"].astype(int)

fund_cleaned.to_excel('fund_cleaned.xlsx',index=False)

# fund company
fund_company = pd.read_excel('fund_company.xlsx', header=[0, 1])

fund_company = fund_company[[("序号", "公司名称"), ("序号", "截止日期"), 
                  ("基金资产净值合计(亿元)", "全部"), 
                  ("基金数量合计(只)", "全部")]]

fund_company.columns = ["FundCompanyName", "Time", "NAV", "NumFunds"]
fund_company['NAV'] = np.log(fund_company['NAV'])

def convert_quarter_to_date(df, col):
    extracted = df[col].str.extract(r'(\d{4})年第(\d)季')  # 提取年份和季度
    df[col] = pd.to_datetime(extracted[0] + 'Q' + extracted[1]) + pd.offsets.QuarterEnd(0)
    return df

fund_company = convert_quarter_to_date(fund_company, "Time")

treat = pd.read_excel('treat.xlsx')
time_range = pd.date_range(start="2019-03-31", end="2024-12-31", freq='Q')
time_df = pd.DataFrame({'Time': time_range})
treat['key'] = 1
time_df['key'] = 1
treat_panel = treat.merge(time_df, on='key').drop(columns=['key'])
treat_panel['Time'] = pd.to_datetime(treat_panel['Time'])
treat_panel = treat_panel.sort_values(by=['FundCompanyName', 'Time']).reset_index(drop=True)

fund_company_panel = treat_panel.merge(fund_company, on = ['FundCompanyName','Time'],how = 'left')
cutoff_date = pd.to_datetime("2022-09-30")
fund_company_panel["Post"] = (fund_company_panel["Time"] >= cutoff_date).astype(int)
fund_company_panel["Treat_Post"] = fund_company_panel["Treat"] * fund_company_panel["Post"]

# sitevisit
sitivisit_1 = pd.read_excel('sitevisit_1.xlsx')
sitivisit_2 = pd.read_excel('sitevisit_2 .xlsx')
sitivisit_3 = pd.read_excel('sitevisit_3.xlsx')

dfs = [sitivisit_1, sitivisit_2, sitivisit_3]
site_visits = pd.concat([
    df[df["InstitutionType"] == "基金公司"][["ReportDate", "InstitutionName"]].rename(
        columns={"ReportDate": "Time", "InstitutionName": "FundCompanyName"}
    )
    for df in dfs
], ignore_index=True)

## clean FundCompanyName
from thefuzz import process
standard_names = fund_company_panel["FundCompanyName"].unique()
def fuzzy_match(name):
    match, score = process.extractOne(name, standard_names)  # 取最相近的匹配
    return match

site_visits["FundCompanyName"] = site_visits["FundCompanyName"].apply(fuzzy_match)
'万家基金管理有限公司' in standard_names
site_visits["Time"] = pd.to_datetime(site_visits["Time"])
site_visits["Quarter"] = site_visits["Time"].dt.to_period("Q")
visit_counts = site_visits.groupby(["FundCompanyName", "Quarter"]).size().reset_index(name="VisitCount")
all_quarters = pd.period_range("2019Q1", "2024Q4", freq="Q")
all_funds = site_visits["FundCompanyName"].unique()
full_index = pd.MultiIndex.from_product([all_funds, all_quarters], names=["FundCompanyName", "Quarter"])
visit_counts = visit_counts.set_index(["FundCompanyName", "Quarter"]).reindex(full_index, fill_value=0).reset_index()
visit_counts = visit_counts.sort_values(by=["FundCompanyName", "Quarter"])

## merge fund_company and visit
visit_counts["Time"] = visit_counts["Quarter"].dt.to_timestamp(how="end").dt.floor('D')
visit_counts = visit_counts.drop(columns=["Quarter"])
fund_company_panel["Time"] = pd.to_datetime(fund_company_panel["Time"])
merged_panel = fund_company_panel.merge(visit_counts, on=["FundCompanyName", "Time"], how="left")
merged_panel.to_excel('company_clean.xlsx')
