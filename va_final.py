# import pandas as pd
# import statsmodels.api as sm
# from statsmodels.formula.api import ols
#
# va_final = pd.read_csv("VA_FINAL.csv",
#                        usecols=["SNAME", "TEST", "NOICM", "NOICSD", "L", "Q", "PROM", "EXP", "EXPSQR", "EDU", "TOP",
#                                 "RK", "SEX", "CG", "ZA"])
#
# va_df = pd.DataFrame(va_final)
#
# print(va_df)


import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
from linearmodels.panel.data import PanelData
from statsmodels.api import add_constant

# 假设你有一个名为df的pandas DataFrame，其中包含以下列：
# 'entity' - 个体标识
# 'time' - 时间标识
# 'y' - 因变量
# 'x1', 'x2' - 自变量

# 示例数据（这里用字典和DataFrame构造器代替实际数据）
data = {
    'entity': ['A', 'A', 'B', 'B', 'A', 'A', 'B', 'B'],
    'time': [1, 2, 1, 2, 3, 4, 3, 4],
    'y': [1, 2, 3, 4, 5, 6, 7, 8],
    'x1': [2, 3, 4, 5, 6, 7, 8, 9],
    'x2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
}
df = pd.DataFrame(data)

# 转换数据为PanelData格式
# panel_data = PanelData(df, entity_id='entity', time_id='time')

# 为模型指定自变量，并添加常数项
exog = add_constant(df[['x1', 'x2']])

# 拟合双向固定效应模型
mod = PanelOLS(panel_data.dep_var, exog, entity_effects=True, time_effects=True)
res = mod.fit()

# 输出结果
print(res)
