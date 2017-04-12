# coding:utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''Series：一维数组，与Numpy中的一维array类似。
二者与Python基本的数据结构List也很相近，
其区别是：List中的元素可以是不同的数据类型，
而Array和Series中则只允许存
储相同的数据类型，
这样可以更有效的使用内存，
提高运算效率'''
s = pd.Series([1, 3, 5, np.nan, 6, 8])
print s

'''通过传递一个numpy array,时间索引以及列标签来创建一个DataFrame'''
dates = pd.date_range('20130101', periods=6)
print dates

df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
print df

'''通过传递一个能够被转换成类似序列结构的字典对象来创建一个DataFrame：'''
df2 = pd.DataFrame({'A': range(1, 5, 1),
                    'B': pd.Timestamp('20130102'),
                    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                    'D': np.array(range(4), dtype='int32'),
                    'E': pd.Categorical(['test', 'train', 'test', 'train']),
                    'F': 'foo'})

print df2
print df2.dtypes
'''取图表某列'''
print df2.B
'''按照E列分类求和'''
print df2.groupby('E').sum().D

'''取头尾'''
print df2.head(2)
print df2.tail(3)

'''查看索引'''
print df2.index

'''查看列'''
print df2.columns

'''查看数据'''
print df2.values

'''describe()函数对于数据快速统计汇总'''
print df2.describe()

'''对数据转置'''
print df2.T

'''按轴进行排序1,横，0，竖'''
print df2.sort_index(axis=1, ascending=False)
print df2.sort_index(axis=0, ascending=False)

'''按值进行排序'''
print df2.sort(columns='E')

'''选择E列'''
print df2['E']

'''选择前3行'''
print df2[0:3]

'''使用标签来选择一个交叉的区域'''
print df
print df.loc[dates[0]]
print df.loc[:, ['A', 'B']]

'''标签切片'''
print df2.loc[1:3, ['A', 'B']]

'''对于返回对象进行维度缩减'''
print df2.loc[1, ['A', 'B']]
print df2['A']

'''取一个标量'''
print df.loc[dates[0], 'A']
print df.at[dates[0], 'A']

'''选择行'''
print df2.iloc[3]

'''通过数值进行切片，与numpy/python中的情况类似'''
print df.iloc[3:5, 0:2]
print df2.iloc[1:3, 0:2]

'''通过指定一个位置的列表，与numpy/python中的情况类似'''
print df.iloc[[1, 2, 4], [0, 2]]

'''对行进行切片'''
print df.iloc[1:3, :]
print df2.iloc[1:3, :]

'''对列进行切片'''
print df.iloc[:, 1:3]
print df2.iloc[:, 1:3]

'''获取特定的值'''
print df.iloc[1, 1]
print df2.iloc[1, 1]

print df.iat[1, 1]

'''使用一个单列的值来选择数据'''
print df[df.A > 0]

print df[df > 0]

'''使用isin()方法来过滤'''
df3 = df.copy()
df3['E'] = ['one', 'one', 'two', 'three', 'four', 'three']
print df3, '\n', df3['E']

print df3[df3['E'].isin(['two', 'four'])]

'''设置一个新的列'''
s1 = pd.Series(range(1, 7), index=pd.date_range('20130102', periods=6))
print s1

df['F'] = s1
print df

# '通过标签设置新的值'
df.at[dates[0], 'A'] = 0
print df
df2.at[1, 'A'] = 0
print df2

'''通过位置设置新的值'''
df.iat[0, 1] = 0
print df

'''通过一个numpy数组设置一组新值'''
df.loc[:, 'D'] = np.array([5]*len(df))
print df
print df.loc[:, 'D']

'''通过where操作来设置新的值'''
df3 = df.copy()
df3[df3 > 0] = -df3
print df3

'''缺失值处理'''
df1 = df.reindex(index=dates[0:4], columns=list(df.columns)+['E'])
print df1
df1.loc[dates[0]:dates[1], 'E'] = 1
print df1

'''去掉包含缺失值的行'''
print df1.dropna(how='any')

'''对缺失值进行填充'''
print df1.fillna(value=5)

'''对数据进行布尔填充'''
print pd.isnull(df1)

print df
print df.mean()
print df.mean(1)
print dates

'''shift从第几个配和'''
s = pd.Series([1, 3, 5, np.nan, 6, 8], index=dates).shift(2)
print s

print df
print df.sub(s, axis='index')

'''对数据应用函数'''

'''按列累加求和'''
print df.apply(np.cumsum)

'''列最大值减去列最小值'''
print df.apply(lambda x: x.max() - x.min())


'''直方图'''
s = pd.Series(np.random.randint(0, 7, size=10))
print s
print s.value_counts()

s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'gog', 'cat'])
print s
print s.str.lower()

'''合并'''
df = pd.DataFrame(np.random.randn(10, 4))
print df

pieces = [df[:3], df[3:7], df[7:]]
print pieces[1]

print pd.concat(pieces)

''' Join 类似于SQL类型的合并'''
left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['fool', 'foo'], 'rval': [4, 5]})
print left
print right

print pd.merge(left, right, on='key')
print pd.merge(left, right, how='outer', on='key')

'''分组'''
df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
                   'B': ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
                   'C': np.random.randn(8),
                   'D': np.random.randn(8)})
print df

print df.groupby('A').sum()

print df.groupby(['A', 'B']).sum()
print df.groupby(['A', 'B']).sum().D

'''Reshaping'''
tuples = list(zip(*[['bar', 'bar', 'baz', 'baz',
                     'foo', 'foo', 'qux', 'qux'],
                    ['one', 'two', 'one', 'two',
                     'one', 'two', 'one', 'two']]))
print tuples

index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
print index

df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])
print df
df2 = df[: 4]
print df2

stacked = df2.stack()
print stacked

print stacked.unstack()
print stacked.unstack(1)
print stacked.unstack(0)

'''数据透视表'''

import datetime

df = pd.DataFrame({'A': ['one', 'one', 'two', 'three'] * 6,
                   'B': ['A', 'B', 'C'] * 8,
                   'C': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 4,
                   'D': np.random.randn(24),
                   'E': np.random.randn(24),
                   'F': [datetime.datetime(2013, i, 1) for i in range(1, 13)] +
                   [datetime.datetime(2013, i, 15) for i in range(1, 13)]})

print df


print pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'])

'''时间序列'''
'''如将按秒采样的数据转换为按5分钟为单位进行采样的数据'''
rng = pd.date_range('1/1/2012', periods=100, freq='S')
print rng
ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
print ts
print ts.resample('5Min').sum()

rng = pd.date_range('3/6/2012 00:00', periods=5, freq='D')
print rng

ts = pd.Series(np.random.randn(len(rng)), rng)
print ts

ts_utc = ts.tz_localize('UTC')
print ts_utc


'''时区转换'''
print ts_utc.tz_convert('US/Eastern')


'''时间跨度转换'''
rng = pd.date_range('1/1/2012', periods=5, freq='M')
print rng

ts = pd.Series(np.random.randn(len(rng)), index=rng)
print ts

ps = ts.to_period()
print ps

print ps.to_timestamp()

'''时期和时间戳之间的转换使得可以使用一些方便的算术函数'''

prng = pd.period_range('1990Q1', '2000Q4', freq='Q-NOV')
print prng

ts = pd.Series(np.random.randn(len(prng)), prng)
print ts

ts.index = (prng.asfreq('M', 'e')+1).asfreq('H', 's')+ 9
print ts.index
print ts

print ts.head()
print ts.tail()


df = pd.DataFrame({'id':[1, 2, 3, 4, 5, 6], 'raw_grade':['a', 'b', 'b',
                                                         'a', 'a', 'e']})
print df

df['grade'] = df['raw_grade'].astype('category')
print df

print df['grade']

'''将Categorical类型数据重命名为更有意义的名称：'''
df['grade'].cat.categories = ['very good', 'good', 'very bad']
print df

'''将Categorical类型数据重命名为更有意义的名称：'''

df['grade'] = df['grade'].cat.set_categories(['very bad', 'bad', 'medium', 'good', 'very good'])

print df['grade']


'''画图'''
import matplotlib.pyplot as plt

ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()
ts.plot()

df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index, columns=['A', 'B', 'C', 'D'])

df = df.cumsum()

plt.figure(); df.plot(); plt.legend(loc='best');
plt.show()
