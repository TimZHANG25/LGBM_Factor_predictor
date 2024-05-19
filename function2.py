import pandas as pd
from pandas.tseries.offsets import DateOffset
import numpy as np
import math
from datetime import datetime, date, timedelta
from tqdm import tqdm
from WindPy import w
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LinearRegression as LR
import re
from cvxopt import matrix, solvers

# ------数据部分------
# 加载文件


def is_legal(filename, start_date, end_date):
    '''
    函数名称：is_legal
    函数功能：根据数据名称判断数据是否在规定区间内
    输入参数：filename：数据名称列表;
            start_date, end_date: 设置过滤数据的区间,只包含开始日;
    输出参数：True/False
    '''
    date = pd.to_datetime(filename[50:-4])
    return (date >= pd.to_datetime(start_date) and date < pd.to_datetime(end_date))


def Load_Calender(file_names, model_date, n, time_unit, data_interval_unit, date_begin=True):
    '''
    函数名称：Load_Calender
    函数功能：读取日历
    输入参数：filename：数据名称列表；
            model_date：日历开始日期；
            n：日历整体长度；
            time_unit：日历整体长度的单位；
            data_interval_unit：日历中抽取日期的频率
    输出参数：calender
    '''
    if date_begin:
        calender_list = list(pd.to_datetime(x[50:-4]) for x in file_names if is_legal(
            x, model_date, str(pd.to_datetime(model_date) + time_interval(n, time_unit))))
        start_date = pd.to_datetime(min(calender_list))
        if data_interval_unit == '周':
            result = list((x.strftime("%Y%m%d"))
                          for x in calender_list if x.dayofweek == start_date.dayofweek)
            # result = [date.strftime("%Y%m%d") for date in calender_list if date == min(
            #     [d for d in calender_list if (d.week == date.week) and (d.month == date.month) and (d.year == date.year)])]
        elif data_interval_unit == '月':
            # result = list((x.strftime("%Y%m%d"))
            #               for x in calender_list if x.day == start_date.day)
            result = [date.strftime("%Y%m%d") for date in calender_list if date == min(
                [d for d in calender_list if (d.month == date.month) and (d.year == date.year)])]
        elif data_interval_unit == '年':
            # result = list((x.strftime("%Y%m%d"))
            #               for x in calender_list if x.dayofyear == start_date.dayofyear)
            result = [date.strftime("%Y%m%d") for date in calender_list if date == min(
                [d for d in calender_list if d.year == date.year])]
        else:
            result = list((x.strftime("%Y%m%d")) for x in calender_list)
        calender = pd.DataFrame()
        calender['TRADE_DT'] = result

    else:
        calender_list = list(pd.to_datetime(x[50:-4]) for x in file_names if is_legal(
            x, str(pd.to_datetime(model_date) - time_interval(n, time_unit)), model_date))
        start_date = pd.to_datetime(min(calender_list))
        if data_interval_unit == '周':
            result = list((x.strftime("%Y%m%d"))
                          for x in calender_list if x.dayofweek == start_date.dayofweek)
            # result = [date.strftime("%Y%m%d") for date in calender_list if date == min(
            #     [d for d in calender_list if (d.week == date.week) and (d.month == date.month) and (d.year == date.year)])]
        elif data_interval_unit == '月':
            # result = list((x.strftime("%Y%m%d"))
            #               for x in calender_list if x.day == start_date.day)
            result = [date.strftime("%Y%m%d") for date in calender_list if date == min(
                [d for d in calender_list if (d.month == date.month) and (d.year == date.year)])]
        elif data_interval_unit == '年':
            # result = list((x.strftime("%Y%m%d"))
            #               for x in calender_list if x.dayofyear == start_date.dayofyear)
            result = [date.strftime("%Y%m%d") for date in calender_list if date == min(
                [d for d in calender_list if d.year == date.year])]
        else:
            result = list((x.strftime("%Y%m%d")) for x in calender_list)
        calender = pd.DataFrame()
        calender['TRADE_DT'] = result
    return calender


def filter_file_name(input_file_names, Calender):
    '''
    函数名称：filter_file_name
    函数功能：根据日历过滤数据名称
    输入参数：input_file_names：数据名称列表；
            Calender：日历；
    输出参数：data
    '''
    data = list(
        x for x in input_file_names if x[50:-4].replace('.', '') in list(Calender['TRADE_DT']))
    return data


def read_data(input_file_names, response_type, unit, num_type=10, if_split=True):
    '''
    函数名称：read_data
    函数功能：读取数据并返回特征和因变量
    输入参数：input_file_names：数据名称列表；
            response_type：应变量选择；
            unit：收益频率；
            num_type：分类数目；
            if_split：是否返回分开的特征和标签；
    输出参数：features，response
    '''
    units = {
        '周': 5,
        '月': 20
    }
    response_types = {
        '夏普率': 'sharp_momentum_{}p'.format(units[unit]),
        '收益率': 'zdf{}'.format(units[unit]),
        '夏普率分类': 'sharp_momentum_type_{}p'.format(units[unit]),
        '收益率分类': 'zdf_type{}'.format(units[unit])
    }
    if '分类' not in response_type:
        result = pd.DataFrame()
        for files in tqdm(input_file_names, desc='读取数据...'):
            data = pd.read_csv(files, sep=',', encoding='gb2312')


            data.replace(['-0w', '0w'], np.NaN, inplace=True)
            data.fillna(data.median(), inplace=True)
            data.fillna(0, inplace=True)
            for columns in ['Amount_20D_AVG', 'ep_ts_score120', 'grossprofit_qfa', 'net_profit_excl_min_int_inc_qfa', 'oper_rev_qfa_tb', 'delta_EPFY1_20d', 'np_fy1_tb_120d', 'net_profit_after_ded_nr_lp_qfa', 'net_profit_after_ded_nr_lp_qfa_tb', 'Amt2RealizedVolatility_5D', 'RNOAMRQ', 'delta_conSP_20d', 'ROA_excl_qfa_chg', 'ROE_excl_qfa_chg', 'oper_profit_qfa', 'con_ROE2_fy3']:
                data[columns] = data[columns].astype(float)

# 'con_ROE2' why invalaid?   'sharp_momentum_5p', 'sharp_momentum_20p', 'zdf5', 'zdf20' not found

            result = result.append(data, ignore_index=True)
            #print("result")
            #print(result)
            #result = pd.concat([result, pd.DataFrame([data])], ignore_index=True)
        result.rename(columns={'trade_dt': 'TRADE_DT',
                      's_info_windcode': 'S_INFO_WINDCODE'}, inplace=True)




       # result['TRADE_DT'] = list(x.replace('-', '')
        #                          for x in result['TRADE_DT'])
        result['TRADE_DT'] = list(re.sub(r'\s+|-', '', x)
                                  for x in result['TRADE_DT'])
        features = result.drop(
            ['zdf5', 'zdf20'], axis=1)
        #    'sharp_momentum_5p', 'sharp_momentum_20p','zdf5', 'zdf20'
    else:

        result = pd.DataFrame()
        for files in tqdm(input_file_names, desc='读取数据...'):
            data = pd.read_csv(files, sep=',', encoding='gb2312')

            data.replace(['-0w', '0w'], np.NaN, inplace=True)
            data.fillna(data.median(), inplace=True)
            data.fillna(0, inplace=True)
            for columns in ['Amount_20D_AVG', 'ep_ts_score120', 'grossprofit_qfa', 'net_profit_excl_min_int_inc_qfa', 'oper_rev_qfa_tb', 'delta_EPFY1_20d', 'np_fy1_tb_120d', 'net_profit_after_ded_nr_lp_qfa', 'net_profit_after_ded_nr_lp_qfa_tb', 'Amt2RealizedVolatility_5D', 'RNOAMRQ', 'delta_conSP_20d', 'ROA_excl_qfa_chg', 'ROE_excl_qfa_chg', 'oper_profit_qfa', 'con_ROE2_fy3']:
                data[columns] = data[columns].astype(float)

            data['zdf_type{}'.format(units[unit])] = pd.cut(
                data['zdf{}'.format(units[unit])], bins=num_type, labels=False, right=False)
            # data['sharp_momentum_type_{}p'.format(units[unit])] = pd.cut(
            #     data['sharp_momentum_{}p'.format(units[unit])], bins=num_type, labels=False, right=False)

            result = result.append(data, ignore_index=True)
        result.rename(columns={'trade_dt': 'TRADE_DT',
                      's_info_windcode': 'S_INFO_WINDCODE'}, inplace=True)
        print(result['TRADE_DT'])
        result['TRADE_DT'] = list(x.replace('-', '')
                                  for x in result['TRADE_DT'])
        #result['TRADE_DT'] = list(re.sub(r'\s+|-', '', x)
        #                          for x in result['TRADE_DT'])
        features = result.drop(['sharp_momentum_5p', 'sharp_momentum_20p', 'zdf5', 'zdf20', 'zdf_type{}'.format(
            units[unit]), 'sharp_momentum_type_{}p'.format(units[unit])], axis=1)

    response = result[response_types[response_type]]
    result = pd.concat([features, response], axis=1)
    if if_split == True:
        return features, response
    else:
        return result


def time_interval(n, unit):
    time_intervals = {'日': DateOffset(days=n),
                      '周': DateOffset(weeks=n),
                      '月': DateOffset(months=n),
                      '年': DateOffset(years=n)}
    return time_intervals[unit]


def valid_modle_date(file_names, model_time, train_interval, test_interval, evaluate_interval, unit):
    '''
    函数名称：valid_modle_date
    函数功能：检查是否有足够的数据训练模型
    输入参数：file_names：训练模型所需的所有文件名;
            model_time：模型的训练时间；
            train_interval：模型训练数据长度
            test_interval：模型测试数据长度
            evaluate_interval：回测数据长度
            unit：模型使用数据长度单位;
    输出参数：True/False
    '''
    # all_time = list(pd.to_datetime(x[53:-4]) for x in file_names)
    all_time = list(pd.to_datetime(
        re.search(r'\d{4}\.\d{2}\.\d{2}', x).group()) for x in file_names)
    start_time = pd.to_datetime(
        model_time) - time_interval(train_interval+test_interval, unit)
    end_time = pd.to_datetime(model_time) + \
        time_interval(evaluate_interval, unit)
    return (end_time <= max(all_time)) and (start_time >= min(all_time))


def Cal_Stratify(data, GroupNum):
    '''
    函数名称：Cal_Stratify
    函数功能：获取因子值分组
    输入参数：data：因子值；GroupNum：组数
    输出参数：Result：因子分组
    '''
    # 按照ep分成10组，分别计算收益率
    data_Rank = data.groupby('TRADE_DT').apply(Get_Rank).reset_index(drop=True)
    length = data_Rank.groupby('TRADE_DT')[['RANK']].count()  # 计算每月共有多少只股票
    # bins记录分组的边界，是分组的依据（左闭右开）
    bins = pd.DataFrame(length['RANK'].apply(
        lambda _: [math.ceil(_ / GroupNum * i) for i in range(GroupNum + 1)]))
    GroupLabels = [str(i) for i in range(1, GroupNum + 1)]  # 为每组标号
    data_Rank = data_Rank.groupby('TRADE_DT').apply(
        Get_Group, bins, GroupLabels).reset_index(drop=True)
    Result = data_Rank[['TRADE_DT', 'S_INFO_WINDCODE', 'GROUP']]
    return Result


def Get_Rank(group):
    '''
    函数名称：Get_Rank
    函数功能：排序，groupby内部函数
    输入参数：group：待排序值
    输出参数：group：添加了一列序号
    '''
    group['RANK'] = group[['FACTOR']].rank(
        ascending=False, method='first', axis=0, na_option='keep')
    return group


def Get_Group(group, bins, GroupLabels):
    '''
    函数名称：Get_Group
    函数功能：分组，groupby内部函数
    输入参数：group：待排序值；bins分组边界；GroupLabels：各组标签
    输出参数：group：添加了一列组标签
    '''
    group['GROUP'] = pd.cut(
        group.RANK, bins.loc[group.TRADE_DT.values[0], 'RANK'], labels=GroupLabels)
    return group


def Get_FACTOR_Neu(input_data, model, if_regression=True):
    '''
    函数名称:Get_FACTOR_Neu
    函数功能:计算测试集预测标签
    输入参数:input_data:待预测值
             model:预测模型
             if_regression:判断是分类还是回归问题
    输出参数:result:训练集特征和标签
    '''
    result = input_data
    if if_regression:
        result['predicted_y'] = model.predict(
            input_data.drop(['TRADE_DT', 'S_INFO_WINDCODE'], axis=1))
    else:
        weights = model.predict(input_data.drop(
            ['TRADE_DT', 'S_INFO_WINDCODE'], axis=1))
        labels = np.array([i+1 for i in range(10)])
        result['predicted_y'] = np.dot(weights, labels)
    return result


# 计算仓位
def Cal_Position(Hold):
    '''
    函数名称：Cal_Position
    函数功能：根据持仓股票池计算下期仓位
    输入参数：Hold：持仓股票池
    输出参数：Position：仓位
    '''
    # 获取持仓股票池的周期列表
    PeriodList = Hold.TRADE_DT.unique()
    # 复制一份持仓股票池
    Position = Hold.copy()
    # 添加仓位列
    Position['POSITION'] = 1
    # 移动至下期
    Position.TRADE_DT = [PeriodList[np.where(PeriodList == i)[0][0]+1] if np.where(
        PeriodList == i)[0][0]+1 < len(PeriodList) else np.nan for i in Position.TRADE_DT]
    # 删除空行
    Position.dropna(subset=['TRADE_DT'], inplace=True)

    # 计算每个周期的仓位
    Position['POSITION'] = Position.groupby('TRADE_DT').apply(
        lambda x: x['POSITION'] / sum(x['POSITION'])).reset_index(drop=True).values
    # 添加最后一行
    Position.loc[len(Position.index)] = [PeriodList[np.where(
        PeriodList == Position.TRADE_DT.values[0])[0][0]-1], 'StockCode', 0]
    # 按照交易日期和股票代码排序
    Position = Position.sort_values(
        by=['TRADE_DT', 'S_INFO_WINDCODE']).reset_index(drop=True)
    # 返回仓位
    return Position


def Back_Testing(Position, Price, BenchReturn, RiskFreeReturn, frequency='D'):
    '''
    函数名称：Back_Testing
    函数功能：回测函数
    输入参数：Position：仓位数据；Price：股票价格数据；BenchReturn：基准收益率；RiskFreeReturn：无风险收益率；StartDate：起始日期；EndDate：截止日期
    输出参数：NetValue：净值；Perform：绩效统计
    '''
    PeriodList = Position.TRADE_DT.unique()
    # 计算策略的日频收益率
    # bins = np.insert(PeriodList, 0, '00000000')
    # Price['TRADE_PERIOD'] = pd.cut(
    #     Price['TRADE_DT'], bins=bins, labels=PeriodList[:])
    Price['TRADE_PERIOD'] = [
        PeriodList[np.where(PeriodList >= i)[0][0]] for i in Price.TRADE_DT]
    StrategyReturn = pd.DataFrame(Price.groupby(Price.TRADE_PERIOD).apply(
        M2D, Position).values, index=Price.TRADE_DT.unique(), columns=['RETURN'])
    StrategyReturn.fillna(0, inplace=True)
    NetValue = pd.DataFrame(index=BenchReturn.index,
                            columns=['基准净值', '策略净值', '相对净值', '策略收益'])
    NetValue['策略收益'] = StrategyReturn['RETURN']
    # 计算换手率
    TurnOverRate = Cal_TurnOver(Position)
    # 计算无风险利率（货基指数）的年化收益
    RiskfreeReturn, RiskfreeIntervalRet, RiskfreeIntervaStd, RiskfreeAnnualRet, RiskfreeAnnualStd, RiskfreeMaxdrawdown = PerfStatis(
        RiskFreeReturn, frequency)
    # 统计策略的净值、区间收益率、区间波动率、年化收益率、年化波动率、夏普比率、最大回撤
    StrategyReturn, StrategyIntervalRet, StrategyIntervaStd, StrategyAnnualRet, StrategyAnnualStd, StrategyMaxdrawdown = PerfStatis(
        StrategyReturn, frequency)
    # 统计基准的净值、区间收益率、区间波动率、年化收益率、年化波动率、夏普比率、最大回撤
    BenchReturn.loc[BenchReturn.index <= PeriodList[0], 'RETURN'] = 0
    BenchReturn, BenchIntervalRet, BenchIntervaStd, BenchAnnualRet, BenchAnnualStd, BenchMaxdrawdown = PerfStatis(
        BenchReturn, frequency)
    # 统计超额的净值、区间收益率、区间波动率、年化收益率、年化波动率、夏普比率、最大回撤
    initial_ExcessReturn = Cal_Excess(StrategyReturn, BenchReturn)
    ExcessReturn, ExcessIntervalRet, ExcessIntervaStd, ExcessAnnualRet, ExcessAnnualStd, ExcessMaxdrawdown = PerfStatis(
        initial_ExcessReturn, frequency)
    StrategyWinper = Cal_Winper(ExcessReturn)

    # 保存总体绩效统计
    Perform = pd.DataFrame(index=['绩效统计'], columns=['年化收益(%)', '基准年化收益(%)', '超额年化收益(%)', '年化波动(%)', '基准年化波动(%)', '超额年化波动(%)',
                           '最大回撤(%)', '基准最大回撤(%)', '超额最大回撤(%)', '夏普比率', '基准夏普比率', '信息比率', '收益回撤比', '基准收益回撤比', '超额收益回撤比', '胜率(%)', '换手率(年均)'])

    Perform['年化收益(%)'], Perform['基准年化收益(%)'], Perform['超额年化收益(%)'], Perform['年化波动(%)'], Perform['基准年化波动(%)'], Perform['超额年化波动(%)'], \
        Perform['最大回撤(%)'], Perform['基准最大回撤(%)'], Perform['超额最大回撤(%)'], Perform['收益回撤比'], Perform['基准收益回撤比'], Perform['超额收益回撤比'], Perform['胜率(%)'], Perform['换手率(年均)'] = \
        StrategyAnnualRet, BenchAnnualRet, ExcessAnnualRet, StrategyAnnualStd, BenchAnnualStd, ExcessAnnualStd, StrategyMaxdrawdown, BenchMaxdrawdown, ExcessMaxdrawdown, \
        -(StrategyAnnualRet / StrategyMaxdrawdown), -(BenchAnnualRet / BenchMaxdrawdown), - \
        (ExcessAnnualRet / ExcessMaxdrawdown), StrategyWinper, TurnOverRate
    Perform['夏普比率'] = (Perform['年化收益(%)'] -
                       RiskfreeAnnualRet) / Perform['年化波动(%)']
    Perform['基准夏普比率'] = (Perform['基准年化收益(%)'] -
                         RiskfreeAnnualRet) / Perform['基准年化波动(%)']
    Perform['信息比率'] = (Perform['超额年化收益(%)'] -
                       RiskfreeAnnualRet) / Perform['超额年化波动(%)']

    # 保存净值数据
    # NetValue = pd.DataFrame(index=BenchReturn.index,
    #                         columns=['基准净值', '策略净值', '相对净值'])
    NetValue['基准净值'] = BenchReturn['NETVALUE']
    NetValue['策略净值'] = StrategyReturn['NETVALUE']
    NetValue['相对净值'] = ExcessReturn['NETVALUE']

    return NetValue, Perform


def M2D(group, Position):
    '''
    函数名称：M2D
    函数功能：Backtest的内部函数，将月频策略转为日频收益率
            首先需要对资产的日频收益率进行再平衡：根据月频仓位计算每个月内的各资产日频净值，再反推每个月内的各资产日频收益率，以进行仓位再平衡（若只有一个资产，这一步不改变任何值，有多个资产时才起作用）
    输入参数：group：单月的日频各标的收益率；Position：仓位
    输出参数：Result：仓位对应的日频收益率
    '''
    Period = group.TRADE_PERIOD.values[0]
    PositionPeriod = Position[Position.TRADE_DT == Period]
    NetValueTmp = pd.DataFrame(
        index=group.TRADE_DT.unique(), columns=['NETVALUE'])
    group = pd.merge(group[['TRADE_DT', 'S_INFO_WINDCODE', 'RETURN']], PositionPeriod[[
                     'S_INFO_WINDCODE', 'POSITION']], on='S_INFO_WINDCODE', how='right')
    # 对于异常情况（月底买入但次月停牌，即次月无收益率，则填0）
    # group['TRADE_DT'].fillna(group.TRADE_DT.unique()[0], inplace=True)

    # 填充缺失值（缺失值填充为当月）
    group['TRADE_DT'].fillna(Period, inplace=True)
    group['RETURN'].fillna(0, inplace=True)

    # 如果当月无仓位，则组合收益率即当月股票收益率均值
    if (group['POSITION'] == 0).all():
        NetValueTmp.loc[:] = np.ones([len(NetValueTmp), 1])
    else:
        # 计算当月仓位
        PositionValue = pd.pivot(
            group, index='TRADE_DT', columns='S_INFO_WINDCODE', values='POSITION').iloc[0].values
        # 计算当月股票收益率
        groupPivot = pd.pivot(group, index='TRADE_DT',
                              columns='S_INFO_WINDCODE', values='RETURN')
        # 计算当月组合收益率
        NetValueTmp['NETVALUE'] = (
            PositionValue * np.cumprod(groupPivot + 1)).sum(axis=1).values

    # 计算组合收益率
    NetValueTmp['RETURN'] = NetValueTmp['NETVALUE'] / \
        NetValueTmp['NETVALUE'].shift(1) - 1
    # 第一天的组合收益率即第一天所有股票收益率均值
    NetValueTmp['RETURN'].values[0] = group[group.TRADE_DT ==
                                            min(group.TRADE_DT)]['RETURN'].mean()
    Result = NetValueTmp[['RETURN']]
    return Result


def Cal_TurnOver(Position):
    '''
    函数名称：Cal_TurnOver
    函数功能：计算换手率
    输入参数：Position：仓位数据
    输出参数：TurnOverRate：换手率
    '''
    Position_Pivot = pd.pivot(
        Position, index='TRADE_DT', columns='S_INFO_WINDCODE', values='POSITION')
    Position_Pivot.fillna(0, inplace=True)
    TurnOverRate = np.sum(
        abs(Position_Pivot.values - Position_Pivot.shift(1).values), axis=1)
    TurnOverRate[0] = 0
    TurnOverRate = np.sum(TurnOverRate) / len(TurnOverRate)
    return TurnOverRate


def PerfStatis(Return, frequency, if_Returned=True):
    '''
    函数名称：PerfStatis
    函数功能：Backtest的内部函数，用于统计绩效指标
    输入参数：Return：收益率；frequency：频率("M","W","D")
    输出参数：区间收益率 区间波动率 年化收益率 年化波动率 夏普比率 最大回撤
    '''
    if frequency == 'M':
        N = 12
    elif frequency == 'W':
        N = 52
    elif frequency == 'D':
        N = 250
    ET = len(Return)  # 有效长度
    # 计算净值
    if if_Returned:
        Return['NETVALUE'] = np.cumprod(Return['RETURN'] + 1)
    # 计算区间收益率、区间波动率、年化收益率和年化波动率
    IntervalRet = 100 * (Return['NETVALUE'].values[-1] /
                         Return['NETVALUE'].values[0] - 1)
    IntervaStd = 100 * np.std(Return['RETURN'].values)
    AnnualRet = 100 * \
        ((Return['NETVALUE'].values[-1] /
         Return['NETVALUE'].values[0]) ** (N / ET) - 1)
    AnnualStd = 100 * np.std(Return['RETURN'].values) * np.sqrt(N)
    # 计算最大回撤
    Maxdrawdown = 100 * min(Return['NETVALUE'] /
                            Return['NETVALUE'].cummax() - 1)
    return Return, IntervalRet, IntervaStd, AnnualRet, AnnualStd, Maxdrawdown


def Cal_Excess(StrategyReturn, BenchReturn):
    '''
    函数名称：Cal_Excess
    函数功能：计算超额收益
    输入参数：StrategyReturn：策略收益；BenchReturn：基准收益
    输出参数：ExcessReturn：超额收益
    '''
    ExcessReturn = pd.DataFrame(index=BenchReturn.index, columns=['RETURN'])
    ExcessReturn['NETVALUE'] = StrategyReturn['NETVALUE'].values / \
        BenchReturn['NETVALUE'].values
    ExcessReturn['RETURN'] = ExcessReturn['NETVALUE'] / \
        ExcessReturn['NETVALUE'].shift(1) - 1
    ExcessReturn['RETURN'].values[0] = 0
    return ExcessReturn


def Cal_Winper(Return):
    '''
    函数名称：Cal_Winper
    函数功能：计算胜率
    输入参数：Return：收益率数据
    输出参数：Winper：胜率
    '''
    Winper = len(Return[Return['RETURN'] > 0]) / \
        len(Return[Return['RETURN'] != 0])
    return Winper


def Load_SQLData(conn, Calender, test, Outputpath, local=False):
    '''
    函数名称：Load_SQLData
    函数功能：从sql server读取交易日、股票价格、市值、所属行业、ST日期、交易状态（是否停牌）、上市日期，并保存到本地pkl文件
    输入参数：conn：sql连接工具；StartDate：开始日期；EndDate：截止日期；local：是否从本地读取（初次运行需设为False，再次运行可设为True，节省时间）
    输出参数：Price：股票价格；MarketCap：市值；Industry：所属行业；
            ST：ST日期；Suspend：交易状态（是否停牌）；ListDate：上市日期
    '''
    StartDate, EndDate = Calender['TRADE_DT'].values[0], Calender['TRADE_DT'].values[-1]
    # 股票价格数据
    localfile = Outputpath + '/Price.pkl' if local else False  # 从本地读取
    Price = Load_Price(conn, StartDate, EndDate, Outputpath, localfile)
    # 市值数据
    localfile = Outputpath + '/MarketCap.pkl' if local else False  # 从本地读取
    MarketCap = Load_MarketCap(conn, StartDate, EndDate, Outputpath, localfile)
    # 所属行业数据
    localfile = Outputpath + '/Industry.pkl' if local else False  # 从本地读取
    Industry = Load_Industry(conn, Outputpath, localfile)
    # ST日期数据
    localfile = Outputpath + '/ST.pkl' if local else False  # 从本地读取
    ST = Load_ST(conn, Outputpath, localfile)
    # 交易状态（是否停牌）数据
    localfile = Outputpath + '/Suspend.pkl' if local else False  # 从本地读取
    Suspend = Load_Suspend(conn, localfile, StartDate,
                           EndDate, Outputpath, test)
    # 上市日期数据
    localfile = Outputpath + '/ListDate.pkl' if local else False  # 从本地读取
    ListDate = Load_ListDate(conn, Outputpath, localfile)
    MarketCap = pd.merge(Calender, MarketCap, on='TRADE_DT', how='left')
    Suspend = pd.merge(Calender, Suspend, on='TRADE_DT', how='left')

    return Price, MarketCap, Industry, ST, Suspend, ListDate


def Load_Price(conn, StartDate, EndDate, Outputpath, localfile=False):
    '''
    函数名称：Load_Price
    函数功能：从sql server读取股票价格数据
    输入参数：conn：sql连接工具；StartDate：开始日期；EndDate：截止日期；localfile：本地文件（取值False为从sql server读取）
    输出参数：data：股票价格数据
    '''
    if localfile:
        data = pd.read_pickle(localfile)
    else:
        sql = "select TRADE_DT,S_INFO_WINDCODE,S_DQ_ADJPRECLOSE,S_DQ_ADJCLOSE from AShareEODPrices where TRADE_DT>=\'" + \
            StartDate + "\' AND TRADE_DT<= \'" + EndDate + "\'"
        data = pd.read_sql(sql, conn)
        data.to_pickle('{}/Price.pkl'.format(Outputpath))

    data.TRADE_DT = data.TRADE_DT.apply(str)
    data = data[(data.TRADE_DT >= StartDate.replace('-','')) & (data.TRADE_DT <= EndDate.replace('-',''))]
    data.sort_values(by=['TRADE_DT', 'S_INFO_WINDCODE'], inplace=True)
    data['RETURN'] = data['S_DQ_ADJCLOSE'] / data['S_DQ_ADJPRECLOSE'] - 1
    data = data.reset_index(drop=True)
    return data


def Load_MarketCap(conn, StartDate, EndDate, Outputpath, localfile):
    '''
    函数名称：Load_MarketCap
    函数功能：从sql server读取市值数据
    输入参数：conn：sql连接工具；StartDate：开始日期；EndDate：截止日期；localfile：本地文件（取值False为从sql server读取）
    输出参数：data：市值数据
    '''
    if localfile:
        data = pd.read_pickle(localfile)
    else:
        sql = "select TRADE_DT,S_INFO_WINDCODE, S_VAL_MV from ASHAREEODDERIVATIVEINDICATOR where TRADE_DT>=\'" + \
            StartDate + "\' AND TRADE_DT<= \'" + EndDate + "\'"
        data = pd.read_sql(sql, conn)
        data.to_pickle('{}/MarketCap.pkl'.format(Outputpath))

    data.TRADE_DT = data.TRADE_DT.apply(str)
    data = data[(data.TRADE_DT >= StartDate.replace('-','')) & (data.TRADE_DT <= EndDate.replace('-',''))]
    data.sort_values(by=['TRADE_DT', 'S_INFO_WINDCODE'], inplace=True)
    data['S_VAL_MV'] = np.log(data['S_VAL_MV'] / 100)
    data = data.reset_index(drop=True)
    return data


def Load_Industry(conn, Outputpath, localfile=False):
    '''
    函数名称：Load_Industry
    函数功能：从sql server读取所属行业数据
    输入参数：conn：sql连接工具；localfile：本地文件（取值False为从sql server读取）
    输出参数：data：所属行业数据
    '''
    if localfile:
        data = pd.read_pickle(localfile)
    else:
        sql = 'select S_INFO_WINDCODE,substring(citics_ind_code, 1, 4) as IND_CODE, ENTRY_DT,REMOVE_DT from AShareIndustriesClassCITICS'
        data = pd.read_sql(sql, conn)
        data.to_pickle('{}/Industry.pkl'.format(Outputpath))

    data.sort_values(by=['S_INFO_WINDCODE'], inplace=True)
    data = data.reset_index(drop=True)
    return data


def Load_ST(conn, Outputpath, localfile):
    '''
    函数名称：Load_ST
    函数功能：从sql server读取ST日期数据
    输入参数：conn：sql连接工具；localfile：本地文件（取值False为从sql server读取）
    输出参数：data：ST日期数据
    '''
    if localfile:
        data = pd.read_pickle(localfile)
    else:
        sql = "select S_INFO_WINDCODE,ENTRY_DT,REMOVE_DT from ASHAREST"
        data = pd.read_sql(sql, conn)
        data.to_pickle('{}/ST.pkl'.format(Outputpath))

    data.sort_values(by=['S_INFO_WINDCODE'], inplace=True)
    data = data.reset_index(drop=True)
    return data


def Load_Suspend(conn, localfile, StartDate, EndDate, Outputpath, test):
    '''
    函数名称：Load_Suspend
    函数功能：从sql server读取交易状态（是否停牌）数据
    输入参数：conn：sql连接工具；localfile：本地文件（取值False为从sql server读取）
    输出参数：data：交易状态（是否停牌）数据
    '''
    if localfile:
        data = pd.read_pickle(localfile)
    else:
        if test:
            sql = "select TRADE_DT,S_INFO_WINDCODE,S_DQ_TRADESTATUSCODE from ASHAREEODPRICES where TRADE_DT>=\'" + \
                StartDate + "\' AND TRADE_DT<= \'" + EndDate + "\'"
        else:
            sql = "select TRADE_DT,S_INFO_WINDCODE,S_DQ_TRADESTATUSCODE from ASHAREEODPRICES"
        data = pd.read_sql(sql, conn)
        data.to_pickle('{}/Suspend.pkl'.format(Outputpath))

    data.sort_values(by=['S_INFO_WINDCODE'], inplace=True)
    data = data.reset_index(drop=True)
    return data


def Load_ListDate(conn, Outputpath, localfile):
    '''
    函数名称：Load_ListDate
    函数功能：从sql server读取上市日期数据
    输入参数：conn：sql连接工具；localfile：本地文件（取值False为从sql server读取）
    输出参数：data：上市日期数据
    '''
    if localfile:
        data = pd.read_pickle(localfile)
    else:
        sql = "select S_INFO_WINDCODE,S_INFO_LISTDATE from ASHAREDESCRIPTION"
        data = pd.read_sql(sql, conn)
        data.to_pickle('{}/ListDate.pkl'.format(Outputpath))

    data.sort_values(by=['S_INFO_WINDCODE'], inplace=True)
    data = data.reset_index(drop=True)
    return data


def Load_Return(conn, Calender):

    StartDate, EndDate = Calender['TRADE_DT'].values[0], Calender['TRADE_DT'].values[-1]

    sql = "select S_DQ_PCTCHANGE , TRADE_DT from AINDEXEODPRICES where S_INFO_WINDCODE = \'000905.SH\' AND TRADE_DT >=\'" + \
        StartDate + "\' AND TRADE_DT<= \'" + EndDate + "\'"
    Return = pd.read_sql(sql, conn)
    Return.rename(columns={'S_DQ_PCTCHANGE': 'RETURN'}, inplace=True)
    Return['RETURN'] = Return['RETURN'] / 100
    Return.set_index('TRADE_DT', inplace=True)
    Return.sort_index(inplace=True)
    return Return


def LoadRiskFreeReturn(filename, Calender):
    StartDate, EndDate = Calender['TRADE_DT'].values[0], Calender['TRADE_DT'].values[-1]

    data = pd.read_excel(filename, skiprows=1)
    data['Date'] = list(datetime.strftime(x, '%Y%m%d') for x in data['Date'])
    Return = data[(data['Date'] >= StartDate) & (data['Date'] <= EndDate)].drop(
        '885009.WI', axis=1).rename(columns={'无风险利率': 'RETURN'}).set_index('Date')
    Return.sort_index(inplace=True)
    return Return


def Cal_Sift(data, ST, Suspend, ListDate):
    '''
    函数名称：Cal_Sift
    函数功能：对股票池进行筛选（1：不为ST，2：不为停牌，3：上市满6个月）
    输入参数：data：股票池；ST：ST日期数据；Suspend：交易状态（是否停牌）数据；ListDate：上市日期数据
    输出参数：data_Sift：筛选后的股票池
    '''
    # 对ST条件进行筛选
    data_tmp1 = Sift_ST(data, ST)
    # 对停牌条件进行筛选
    data_tmp2 = Sift_Suspend(data_tmp1, Suspend)
    # 对上市满6个月条件进行筛选
    data_Sift = Sift_ListDate(data_tmp2, ListDate)
    data_Sift = data_Sift.reset_index(drop=True)
    return data_Sift


def Sift_ST(data, ST):
    '''
    函数名称：Sift_ST
    函数功能：根据是否处于ST筛选股票池
    输入参数：data：股票池；ST：ST日期数据；
    输出参数：Result：筛选后的股票池
    '''
    ST['REMOVE_DT'].fillna('99999999', inplace=True)
    data_merge = pd.merge(data, ST, on='S_INFO_WINDCODE', how='left')
    data_delete = data_merge.loc[(data_merge.TRADE_DT >= data_merge.ENTRY_DT) & (
        data_merge.TRADE_DT <= data_merge.REMOVE_DT), ['TRADE_DT', 'S_INFO_WINDCODE', 'FACTOR']]
    diff = pd.concat([data, data_delete, data_delete]
                     ).drop_duplicates(keep=False)
    Result = diff[['TRADE_DT', 'S_INFO_WINDCODE', 'FACTOR']]
    return Result


def Sift_Suspend(data, Suspend):
    '''
    函数名称：Sift_Suspend
    函数功能：根据是否处于停牌筛选股票池
    输入参数：data：股票池；Suspend：交易状态（是否停牌）数据；
    输出参数：Result：筛选后的股票池
    '''
    data = pd.merge(data, Suspend, on=[
                    'TRADE_DT', 'S_INFO_WINDCODE'], how='left')
    data = data[data['S_DQ_TRADESTATUSCODE'] != 0]
    Result = data[['TRADE_DT', 'S_INFO_WINDCODE', 'FACTOR']]
    return Result


def Sift_ListDate(data, ListDate):
    '''
    函数名称：Sift_ListDate
    函数功能：根据是否满足上市后6个月筛选股票池
    输入参数：data：股票池；ListDate：上市日期数据；
    输出参数：Result：筛选后的股票池
    '''
    data = pd.merge(data, ListDate, on=['S_INFO_WINDCODE'], how='left')
    data = data.dropna()
    Date = [datetime.strptime(i, "%Y%m%d") for i in data['TRADE_DT']]
    ListDate = [datetime.strptime(i, "%Y%m%d")
                for i in data['S_INFO_LISTDATE']]
    data['ListDelta'] = [12 * (Date[i].year - ListDate[i].year) +
                         (Date[i].month - ListDate[i].month) for i in range(len(data))]
    data['DaysDelta'] = [(Date[i].day - ListDate[i].day)
                         for i in range(len(data))]
    data = data[(data['ListDelta'] > 6) | (
        (data['ListDelta'] == 6) & (data['DaysDelta'] >= 0))]
    Result = data[['TRADE_DT', 'S_INFO_WINDCODE', 'FACTOR']]
    return Result


def Mad_Standard(group):
    '''
    函数名称：Mad_Standard
    函数功能：去极值和标准化，groupby内部函数
    输入参数：group：单日因子值
    输出参数：Result：处理后因子值
    '''
    # MAD去极值
    ColumnName = group.columns[2]
    median = np.nanmedian(group[ColumnName])
    diff_median = ((group[ColumnName] - median).abs()).quantile(0.5)
    max_range = median + 5 * diff_median
    min_range = median - 5 * diff_median
    group[ColumnName+'_Mad'] = np.clip(group[ColumnName], min_range, max_range)
    # ZSCORE标准化
    group[ColumnName+'_Mad_Stand'] = (group[ColumnName+'_Mad'] - np.nanmean(
        group[ColumnName+'_Mad'])) / np.nanstd(group[ColumnName+'_Mad'])
    Result = group[['TRADE_DT', 'S_INFO_WINDCODE',
                    ColumnName, ColumnName+'_Mad_Stand']]
    return Result


def Get_Residual(group, MarketCap, Industry):
    '''
    函数名称：Get_Residual
    函数功能：对因子进行市值和行业中性化，groupby内部函数
    输入参数：group：单日因子值；MarketCap：市值数据；Industry：所属行业数据
    输出参数：Result：处理后因子值
    '''
    # 中性化
    group['FACTOR_Neu'] = np.nan
    group = pd.merge(group, MarketCap, on=[
                'S_INFO_WINDCODE', 'TRADE_DT'], how='left')
    
    # 先取出[ENTRY_DT,REMOVE_DT]包含当日的所属行业数据
    Industry_group = pd.merge(
        Industry, group[['TRADE_DT', 'S_INFO_WINDCODE']], on='S_INFO_WINDCODE', how='left')
    Industry_group['REMOVE_DT'].fillna('21001231', inplace=True)
    Industry_group['TRADE_DT'].fillna(pd.to_datetime('19700101', format='%Y%m%d'), inplace=True)
    Industry_group = Industry_group[(Industry_group.TRADE_DT >= pd.to_datetime(Industry_group['ENTRY_DT'], format='%Y%m%d')) & (
        Industry_group.TRADE_DT <= pd.to_datetime(Industry_group['REMOVE_DT'], format='%Y%m%d'))]
    group = pd.merge(group, Industry_group[['TRADE_DT', 'S_INFO_WINDCODE', 'IND_CODE']], on=[
                     'TRADE_DT', 'S_INFO_WINDCODE'], how='left')
    # 对不为nan的部分进行线性回归取残差
    NotNanIndexs = np.logical_not(pd.isnull(group['zdf5_Mad_Stand'])).values & np.logical_not(
        pd.isnull(group['S_VAL_MV_Mad_Stand'])).values & np.logical_not(pd.isnull(group['IND_CODE'])).values
    if any(NotNanIndexs):
        X = np.concatenate([group.loc[NotNanIndexs, 'S_VAL_MV_Mad_Stand'].values.reshape(
            -1, 1), pd.get_dummies(group.loc[NotNanIndexs, 'IND_CODE']).values], axis=1)
        Y = group.loc[NotNanIndexs, 'zdf5_Mad_Stand'].values.reshape(-1, 1)
        reg = LR()
        reg.fit(X, Y)
        residual = Y - reg.predict(X)
        group.loc[NotNanIndexs, 'FACTOR_Neu'] = residual.flatten()
    Result = group[['TRADE_DT', 'S_INFO_WINDCODE', 'zdf5', 'FACTOR_Neu']]
    return Result


def Extract_model_time(string):
    '''
    函数名称:Extract_model_time
    函数功能:提取给定字符串中第一个形如"yyyy-mm-dd"格式的子串
    输入参数:string:待提取的字符串
    输出参数:match.group():提取出的形如"yyyy-mm-dd"格式的子串
    '''
    pattern = r'\d{4}-\d{2}-\d{2}'
    match = re.search(pattern, string)
    if match:
        return match.group()
    else:
        return None


def Load_Index_Weights(conn, Calender):
    '''
    函数名称:Load_Index_Weights
    函数功能:提取中证500指数成分股相关指标
    输入参数:conn, Calender
    输出参数:中证500指数成分股相关指标
    '''

    EndDate = Calender['TRADE_DT'].values[-1]
    StartDate = datetime.strptime(Calender['TRADE_DT'].values[0], '%Y%m%d')
    StartDate = (StartDate - timedelta(days=30)).strftime('%Y%m%d')

    sql = "select S_CON_WINDCODE,TRADE_DT, I_WEIGHT from dbo.AINDEXHS300FREEWEIGHT where S_INFO_WINDCODE = \'000905.SH\' AND TRADE_DT >=\'" + \
        StartDate + "\' AND TRADE_DT<= \'" + EndDate + "\'"
    Index_weights = pd.read_sql(sql, conn)
    Index_weights.set_index('TRADE_DT', inplace=True)
    Index_weights.sort_index(inplace=True)
    Index_weights.reset_index(inplace=True)
    Index_weights.rename(
        columns={'S_CON_WINDCODE': 'S_INFO_WINDCODE'}, inplace=True)

    return Index_weights


def Calculate_Weight(group):
    '''
    函数名称:Calculate_Weight
    函数功能:迭代计算中证500成分股日度权重
    输入参数:group
    输出参数:group
    '''
    group['I_WEIGHT'].fillna(method='ffill', inplace=True)
    group['cum_return'] = (group['RETURN'].shift(-1) + 1).cumprod()
    group['cum_return_shift'] = group['cum_return'].shift(1)
    group['cum_return_shift'].fillna(1, inplace=True)
    group['I_WEIGHT'] = group['I_WEIGHT']*group['cum_return_shift']
    return group


def Get_Weight(conn, date_start, date_end, OutputPath, if_local=True):
    '''
    函数名称:Get_Weight
    函数功能:计算中证500成分股日度权重
    输入参数:date_start, date_end, OutputPath, if_local
    输出参数:Share_500_all_Weight
    '''
    date_begin = np.datetime64(pd.to_datetime(date_start, format='%Y%m%d'))
    date_start = datetime.strptime(date_start, '%Y%m%d')
    date_start = (date_start - timedelta(days=7)).strftime('%Y%m%d')
    Calender_all = pd.DataFrame({'TRADE_DT': [date_start, date_end]})
    Price, MarketCap, Industry, ST, Suspend, ListDate = Load_SQLData(conn, Calender_all, False, OutputPath,
                                                                     local=if_local)
    Price['TRADE_DT'] = pd.to_datetime(Price['TRADE_DT'], format='%Y%m%d')
    Index_date = Load_Index_Weights(conn, Calender_all)
    Index_date.rename(
        columns={'S_CON_WINDCODE': 'S_INFO_WINDCODE'}, inplace=True)
    Index_date['TRADE_DT'] = pd.to_datetime(
        Index_date['TRADE_DT'], format='%Y%m%d')
    merge_df = pd.merge(Price, Index_date, on=[
                        'TRADE_DT', 'S_INFO_WINDCODE'], how='outer')

    Share_500_list = []
    for i in range(len(Index_date['TRADE_DT'].unique())-1):
        # 上月末
        start_date = pd.to_datetime(Index_date['TRADE_DT'].unique()[i])
        # 这月末
        end_date = pd.to_datetime(Index_date['TRADE_DT'].unique()[i+1])
        # 取出上月末的code列表
        index_list = Index_date[(
            Index_date['TRADE_DT'] == start_date)]['S_INFO_WINDCODE'].unique()
        # merge_df_1中本月在code范围内的股票
        merge_df_month = merge_df[(merge_df['TRADE_DT'] >= start_date) & (
            merge_df['TRADE_DT'] < end_date) & (merge_df['S_INFO_WINDCODE'].isin(index_list))]
        df = merge_df_month.groupby('S_INFO_WINDCODE').apply(Calculate_Weight)
        df['ADJ_I_WEIGHT'] = df.groupby(
            'TRADE_DT')['I_WEIGHT'].transform(lambda x: 100*x / x.sum())
        del df['cum_return'], df['cum_return_shift']
        Share_500_list.append(df)

    i = len(Index_date['TRADE_DT'].unique())-1
    start_date = pd.to_datetime(Index_date['TRADE_DT'].unique()[i])
    index_list = Index_date[(Index_date['TRADE_DT'] ==
                             start_date)]['S_INFO_WINDCODE'].unique()
    merge_df_month = merge_df[(merge_df['TRADE_DT'] >= start_date) & (
        merge_df['S_INFO_WINDCODE'].isin(index_list))]
    df = merge_df_month.groupby('S_INFO_WINDCODE').apply(Calculate_Weight)
    df['ADJ_I_WEIGHT'] = df.groupby(
        'TRADE_DT')['I_WEIGHT'].transform(lambda x: 100*x / x.sum())
    Share_500_list.append(df)
    Share_500_all = pd.concat(Share_500_list, ignore_index=True)
    Share_500_all_Weight = Share_500_all[[
        'TRADE_DT', 'S_INFO_WINDCODE', 'ADJ_I_WEIGHT']]
    Share_500_all_Weight.dropna(inplace=True)
    Share_500_all_Weight.to_pickle('{}/Weight.pkl'.format(OutputPath))

    return Share_500_all_Weight[Share_500_all_Weight['TRADE_DT'] >= date_begin]
#    Share_500_all_Weight.to_pickle('{}/Weight.pkl'.format(OutputPath))


def get_index_data(date, file_names, n, time_unit, data_interval_unit, conn, response_type, unit, if_train=True):
    '''
    函数名称:get_index_data
    函数功能:获取指数数据, 可选用于训练或用于回测
    输入参数:date, file_names, n, time_unit, data_interval_unit, conn, response_type, unit, if_train
    输出参数:data_500
    '''
    date = '-'.join([date[:4], date[4:6], date[6:]])

    if if_train:
        Calender_train = Load_Calender(
            file_names, date, n, time_unit, data_interval_unit, date_begin=False)  # 加载数据日期
        data_files_train = filter_file_name(file_names, Calender_train)

        Index_date = Load_Index_Weights(conn, Calender_train)
        Index_date['TRADE_DT'] = pd.to_datetime(
            Index_date['TRADE_DT'], format='%Y%m%d')
        Index_date['month'] = Index_date['TRADE_DT'].dt.to_period('M')

        train_data = read_data(
            data_files_train, response_type, unit, if_split=False)
        train_data['TRADE_DT'] = pd.to_datetime(
            train_data['TRADE_DT'], format='%Y%m%d')
        train_data['last_month'] = (
            train_data['TRADE_DT'] - pd.DateOffset(months=1)).dt.to_period('M')

        data_500_list = []
        for month in Index_date['month'].unique():
            code_500 = Index_date[Index_date['month']
                                  == month]['S_INFO_WINDCODE']
            data_500_list.append(train_data[(train_data['last_month'] == month) & (
                train_data['S_INFO_WINDCODE'].isin(code_500))])

        data_500 = pd.concat(data_500_list, ignore_index=True)
        del data_500['last_month']

    return data_500


def get_next_date(list, date, freq):
    '''
    函数名称:get_next_date
    函数功能:取日期
    输入参数:list date freq
    输出参数:df_next
    '''
    pattern = r'\d{4}\.\d{2}\.\d{2}'
    total_date = [pd.to_datetime(re.search(pattern, x).group()) for x in list]
    trade_date = [(x.strftime("%Y%m%d")) for x in total_date]
    df = pd.DataFrame()
    df['TRADE_DT'] = trade_date

    df['date'] = pd.to_datetime(df['TRADE_DT'], format='%Y%m%d')
    date_index = df[df['date'] == date].index[0]
    if freq == '周':
        next_date_index = date_index + 4
        next_date = df.loc[date_index:next_date_index, 'TRADE_DT']
    elif freq == '月':
        next_date_index = date_index + 19
        next_date = df.loc[date_index:next_date_index, 'TRADE_DT']
    elif freq == '年':
        next_date_index = date_index + 239
        next_date = df.loc[date_index:next_date_index, 'TRADE_DT']
    elif freq == 'half':
        next_date_index = date_index + 120
        next_date = df.loc[date_index:next_date_index, 'TRADE_DT']
    df_next = pd.DataFrame()
    df_next['TRADE_DT'] = next_date.values
    return df_next


def get_next_date_validation(list, date, freq):
    '''
    函数名称:get_next_date
    函数功能:取日期
    输入参数:list date freq
    输出参数:df_next
    '''
    pattern = r'\d{4}\.\d{2}\.\d{2}'
    total_date = [pd.to_datetime(re.search(pattern, x).group()) for x in list]
    trade_date = [(x.strftime("%Y%m%d")) for x in total_date]
    df = pd.DataFrame()
    df['TRADE_DT'] = trade_date

    df['date'] = pd.to_datetime(df['TRADE_DT'], format='%Y%m%d')
    date_index = df[df['date'] == date].index[0]
    if freq == '周':
        next_date_index = date_index + 4
        next_date = df.loc[date_index:next_date_index, 'TRADE_DT']
    elif freq == '月':
        next_date_index = date_index + 19
        next_date = df.loc[date_index:next_date_index, 'TRADE_DT']
    elif freq == '年':
        next_date_index = date_index + 239
        next_date = df.loc[date_index:next_date_index, 'TRADE_DT']
    elif freq == 'half':
        next_date_index = date_index + 120
        next_date = df.loc[date_index:next_date_index, 'TRADE_DT']
    df_next = pd.DataFrame()
    df_next['TRADE_DT'] = next_date.values
    return df_next

def get_before_date(list, date, freq):
    '''
    函数名称:get_before_date
    函数功能:取日期
    输入参数:list date freq
    输出参数:df_next
    '''
    pattern = r'\d{4}\.\d{2}\.\d{2}'
    total_date = [pd.to_datetime(re.search(pattern, x).group()) for x in list]
    trade_date = [(x.strftime("%Y%m%d")) for x in total_date]
    df = pd.DataFrame()
    df['TRADE_DT'] = trade_date

    df['date'] = pd.to_datetime(df['TRADE_DT'], format='%Y%m%d')
    date_index = df[df['date'] == date].index[0]
    if freq == '周':
        next_date_index = date_index - 4
        next_date = df.loc[next_date_index:date_index, 'TRADE_DT']
    elif freq == '月':
        next_date_index = date_index - 19
        next_date = df.loc[next_date_index:date_index, 'TRADE_DT']
    elif freq == '年':
        next_date_index = date_index - 239
        next_date = df.loc[next_date_index:date_index, 'TRADE_DT']
    elif freq == 'half':
        next_date_index = date_index - 120
        next_date = df.loc[next_date_index:date_index, 'TRADE_DT']
    df_next = pd.DataFrame()
    df_next['TRADE_DT'] = next_date.values
    return df_next



def get_trade_date(list, begin_date, freq):
    '''
    函数名称:get_trade_date
    函数功能:取交易日
    输入参数:list freq
    输出参数:calender
    '''
    pattern = r'\d{4}\.\d{2}\.\d{2}'
    total_date = [pd.to_datetime(re.search(pattern, x).group()) for x in list]
    df = pd.DataFrame()
    df['TRADE_DT'] = total_date
    df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'], format='%Y.%m.%d')
    begin_date = pd.to_datetime(begin_date, format='%Y.%m.%d')
    total_date = df[df['TRADE_DT'] >= begin_date].TRADE_DT.tolist()
    if freq == '周':
        trade_date = total_date[::5]
    elif freq == '月':
        trade_date = total_date[::20]

    trade_date = [(x.strftime("%Y%m%d")) for x in trade_date]
    calender = pd.DataFrame()
    calender['TRADE_DT'] = trade_date
    return calender


def cvxlp2_turn(Date, MU, bmw, w0, turn, A_leq, b_leq, Aeq, beq, UB, solver1):
    N = len(np.array(MU))
    # 目标函数
    c = matrix(np.array(MU) * -1)
    F = matrix(np.vstack((MU * -1, MU)))
    # 2不等式约束 Gx<=h
    # G-> A1=[A_leq;diag(ones(N,1));-diag(ones(N,1))];
    A_leq1 = np.vstack((A_leq, np.eye(N), -1 * np.eye(N)))
    b_leq1 = np.vstack((b_leq, UB - bmw.T, bmw.T))

    C1 = np.hstack((A_leq1, -1*A_leq1))
    c1 = np.vstack(b_leq1 - np.dot(A_leq1, np.array(w0.T)))

    G1 = matrix(C1)
    h1 = matrix(c1)
    # 等式约束 不加中括号，加了表示是list

    A2 = np.array(1.0 * Aeq)
    b2 = 1.0 * np.array(beq)
    C2 = np.hstack((A2, -1*A2))
    c2 = np.vstack(b2 - np.dot(A2, np.array(w0.T)))

    A1 = matrix(C2)
    b1 = matrix(c2)

    C3 = np.ones(2 * N)  # 不等式，限制换手 turn
    b3 = turn
    # 要求w+ w- 权重均大于0
    C4 = -np.eye(2 * N)
    b4 = np.matrix(np.zeros(2 * N)).T

    # 合并不等式
    G0 = matrix(np.vstack((C1, C3, C4)))
    h0 = matrix(np.vstack((c1, b3, b4)))

    # G0 = matrix(np.vstack((C1, C3)))
    # h0 = matrix(np.vstack((c1, b3)))

    # 3.2 和 matlab结果一直 加等式约束
    sol3 = solvers.lp(c=F, G=G0, h=h0, A=A1, b=b1, solver=solver1)
    if sol3['status'] == 'optimal':
        x_origin = np.array(sol3['x'])
        x1 = x_origin[0:N]
        x2 = x_origin[-N:]
        w5 = x1 - x2 + w0.T
    # 输出的是主动权重
        objective1 = -np.dot(np.array(w5.T), np.array(MU))
        return (np.array(w5).flatten(), objective1)
    # 如果无解，保持原权重并记录日期
    else:
        print(Date + sol3['status'])
        return (w0.values[0], Date)
