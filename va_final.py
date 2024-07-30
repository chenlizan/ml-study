import pandas as pd
import statsmodels.api as sm

from linearmodels.panel import PanelOLS
from linearmodels.panel.data import PanelData

va_final = pd.read_csv("FULL_3_LONG.csv",
                       usecols=["SID",  # 学生ID
                                "TEST",  # 考试序号
                                "ZA",  # 每次考试归一化成绩
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
                                "SIZE",  # 班上的学生数
                                "KEY",  # 重点班
                                "SCIE",  # 理科班 （1）
                                "CHOSE",  # 科目选择 1物化生  2地物化 3 政物生  4 史地生 5 政史地
                                "SBJ",  # 1 语文 2 数学 3 外语 4 政治 5 历史  6 地理  7 物理  8 化学  9 生物
                                "LANG",  # 外语语种： 1.英语；2.日语；3.俄语
                                "TEST",  # 考试次数，0是高一的分班考试，1-5是高二的月考，其中2 3 5是区统考，其他的是校统考
                                "GRADE",  # 年级，1.一年级 2.二年级
                                "CLASS",  # 班级，高二重新分班，与高一的班级不一样，用GRADE CLASS两个变量才能识别
                                "PID",  # SID-TEST 二维面板识别码
                                "CID",  # GRADE-CLASS二维组合
                                "DF",  # 以第一次考试为基线的难度系数
                                "A",  # 考试成绩的原始分，经过难度系数调整后的值
                                "Q",  # 考卷的区分度
                                "UNIV",  # 教师的毕业院校
                                "M",  # 每科每次考试的年级均值
                                "SD",  # 每科每次考试的年级标准差，包括自己在内
                                ])

va_df = pd.DataFrame(va_final)

va_df = va_df.dropna()  # 删除所有含有缺失值的行

va_df.set_index(['SID', 'TEST'])

print(va_df.head())



dependent = va_df.ZA
exog = sm.add_constant(va_df[["NOICM",  # 全班排除自己的平均成绩
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
                              "SIZE",  # 班上的学生数
                              "KEY",  # 重点班
                              "SCIE",  # 理科班 （1）
                              "CHOSE",  # 科目选择 1物化生  2地物化 3 政物生  4 史地生 5 政史地
                              "SBJ",  # 1 语文 2 数学 3 外语 4 政治 5 历史  6 地理  7 物理  8 化学  9 生物
                              "LANG",  # 外语语种： 1.英语；2.日语；3.俄语
                              "TEST",  # 考试次数，0是高一的分班考试，1-5是高二的月考，其中2 3 5是区统考，其他的是校统考
                              "GRADE",  # 年级，1.一年级 2.二年级
                              "CLASS",  # 班级，高二重新分班，与高一的班级不一样，用GRADE CLASS两个变量才能识别
                              "PID",  # SID-TEST 二维面板识别码
                              "CID",  # GRADE-CLASS二维组合
                              "DF",  # 以第一次考试为基线的难度系数
                              "A",  # 考试成绩的原始分，经过难度系数调整后的值
                              "Q",  # 考卷的区分度
                              "UNIV",  # 教师的毕业院校
                              "M",  # 每科每次考试的年级均值
                              "SD",  # 每科每次考试的年级标准差，包括自己在内
                              ]])

mod = PanelOLS(va_df, exog, entity_effects=True, time_effects=True)
res = mod.fit()

print(res)
