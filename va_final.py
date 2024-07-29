import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

va_final = pd.read_csv("VA_FINAL.csv",
                       usecols=["SNAME",  # 学生名称
                                "TEST",  # 考试序号
                                "NOICM",  # 全班排除自己的平均成绩
                                "NOICSD",  # 全班排除自己的成绩标准差
                                "L",  # 所在班级低分人
                                "Q",  # 所在班的区分度
                                "PROM",  # 老师上次评职称到现在的间隔年限
                                "EXP",  # 老师工作年限
                                "EXPSQR",  # 老师工作年限平方项
                                "EDU",  # 老师教育年限
                                "TOP",  # 老师毕业学校性质 985、211(1)
                                "RK",  # 老师是否有职称
                                "SEX",  # 老师性别 女(0)
                                "CG",  # 高一高二换过老师 换过(1）
                                "ZA"  # 每次考试归一化成绩
                                ])

va_df = pd.DataFrame(va_final)

unique_sname = va_df['SNAME'].unique()

print(unique_sname)

# import pandas as pd
# import numpy as np
# from linearmodels.panel import PanelOLS
# from linearmodels.panel.data import PanelData
# from statsmodels.api import add_constant
#
# # 假设你有一个名为df的pandas DataFrame，其中包含以下列：
# # 'entity' - 个体标识
# # 'time' - 时间标识
# # 'y' - 因变量
# # 'x1', 'x2' - 自变量
#
# # 示例数据（这里用字典和DataFrame构造器代替实际数据）
# data = {
#     'entity': ['A', 'A', 'B', 'B', 'A', 'A', 'B', 'B'],
#     'time': [1, 2, 1, 2, 3, 4, 3, 4],
#     'y': [1, 2, 3, 4, 5, 6, 7, 8],
#     'x1': [2, 3, 4, 5, 6, 7, 8, 9],
#     'x2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
# }
# df = pd.DataFrame(data)
#
# # 转换数据为PanelData格式
# # panel_data = PanelData(df, entity_id='entity', time_id='time')
#
# # 为模型指定自变量，并添加常数项
# exog = add_constant(df[['x1', 'x2']])
#
# # 拟合双向固定效应模型
# mod = PanelOLS(panel_data.dep_var, exog, entity_effects=True, time_effects=True)
# res = mod.fit()
#
# # 输出结果
# print(res)
