import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import warnings

warnings.filterwarnings(action='ignore') 

def temp1(df_in, want_etching_rate_):
    df_all = pd.read_csv('./data/DataSet.csv')
    df_oxidation = pd.read_csv('./data/Oxidation.csv')
    df_softbake = pd.read_csv('./data/Photo-SoftBake.csv')
    df_lithography = pd.read_csv('./data/Photo-Lithography.csv')
    df_etching = pd.read_csv('./data/Etching.csv')
    df_implatation = pd.read_csv('./data/Implantation.csv')
    df_inspect = pd.read_csv('./data/Inspect.csv')
    df_lot = pd.read_csv('./data/Inspect.csv')
    
    # Lot_Num 18 이상만 불러오기
    df_all = df_all[df_lot['Lot_Num']>=18]
    df_oxidation = df_oxidation[df_lot['Lot_Num']>=18]
    df_softbake = df_softbake[df_lot['Lot_Num']>=18]
    df_lithography = df_lithography[df_lot['Lot_Num']>=18]
    df_etching = df_etching[df_lot['Lot_Num']>=18]
    df_implatation = df_implatation[df_lot['Lot_Num']>=18]
    df_inspect = df_inspect[df_lot['Lot_Num']>=18]
    
    # etching_rate 생성
    df_etching['etching_rate_f1'] = 10 * df_oxidation['thickness'] - df_etching['Thin F1']
    df_etching['etching_rate_f2'] = df_etching['Thin F1'] - df_etching['Thin F2']
    df_etching['etching_rate_f3'] = df_etching['Thin F2'] - df_etching['Thin F3']
    df_etching['etching_rate_f4'] = df_etching['Thin F3'] - df_etching['Thin F4']
    
    df_all['etching_rate_f1'] = 10 * df_oxidation['thickness'] - df_etching['Thin F1']
    df_all['etching_rate_f2'] = df_etching['Thin F1'] - df_etching['Thin F2']
    df_all['etching_rate_f3'] = df_etching['Thin F2'] - df_etching['Thin F3']
    df_all['etching_rate_f4'] = df_etching['Thin F3'] - df_etching['Thin F4']
    
    # spin rate 생성
    df_softbake['spin_rate_2'] = df_all['spin2'] - df_all['spin1']
    df_softbake['spin_rate_3'] = df_all['spin3'] - df_all['spin2']
    
    df_all['spin_rate_2'] = df_all['spin2'] - df_all['spin1']
    df_all['spin_rate_3'] = df_all['spin3'] - df_all['spin2']
    
    # flux rate 생성
    df_implatation['flux_rate_90s'] = df_all['Flux90s'] - df_all['Flux60s']
    df_implatation['flux_rate_160s'] = df_all['Flux160s'] - df_all['Flux90s']
    df_implatation['flux_rate_480s'] = df_all['Flux480s'] - df_all['Flux160s']
    df_implatation['flux_rate_840s'] = df_all['Flux840s'] - df_all['Flux480s']
    
    df_all['flux_rate_90s'] = df_all['Flux90s'] - df_all['Flux60s']
    df_all['flux_rate_160s'] = df_all['Flux160s'] - df_all['Flux90s']
    df_all['flux_rate_480s'] = df_all['Flux480s'] - df_all['Flux160s']
    df_all['flux_rate_840s'] = df_all['Flux840s'] - df_all['Flux480s']
    
    # 불필요 변수 삭제
    df_all.drop(['No_Die', 'Datetime', 'Lot_Num',
                 'Ox_Chamber', 'process', 'Vapor', 
                 'photo_soft_Chamber', 'process 2',
                 'lithography_Chamber', 'Lamp', 'UV_type',
                 'Etching_Chamber', 'Process 3',
                 'Chamber_Num', 'process4', 'Current', 'Datetime.1', 'Error_message','Yield'], axis=1, inplace=True)
    df_oxidation.drop(['No_Die', 'Datetime', 'Lot_Num',
                       'Ox_Chamber', 'process', 'Vapor'], axis=1, inplace=True)
    df_softbake.drop(['No_Die', 'Datetime', 'Lot_Num',
                      'photo_soft_Chamber', 'process 2'], axis=1, inplace=True)
    df_lithography.drop(['No_Die', 'Datetime', 'Lot_Num',
                         'lithography_Chamber', 'Lamp', 'UV_type',], axis=1, inplace=True)
    df_etching.drop(['No_Die', 'Datetime', 'Lot_Num',
                     'Etching_Chamber', 'Process 3',], axis=1, inplace=True)
    df_implatation.drop(['No_Die', 'Datetime', 'Lot_Num',
                         'Chamber_Num', 'process4'], axis=1, inplace=True)
    
    df_all['Wafer_Num'] = df_all['Wafer_Num'].astype('str')
    df_oxidation['Wafer_Num'] = df_oxidation['Wafer_Num'].astype('str')
    df_softbake['Wafer_Num'] = df_softbake['Wafer_Num'].astype('str')
    df_lithography['Wafer_Num'] = df_lithography['Wafer_Num'].astype('str')
    df_etching['Wafer_Num'] = df_etching['Wafer_Num'].astype('str')
    df_implatation['Wafer_Num'] = df_implatation['Wafer_Num'].astype('str')
    
    # 전 공정을 포함한 현 공정
    df_softbake_plus_past_process =  pd.concat([df_oxidation, df_softbake], axis=1)
    df_lithography_plus_softbake = pd.concat([df_softbake, df_lithography], axis=1)
    df_lithography_plus_past_process = pd.concat([df_softbake_plus_past_process, df_lithography], axis=1)
    df_etching_plus_past_process = pd.concat([df_lithography_plus_past_process, df_etching], axis=1)
    df_implatation_plus_past_process = pd.concat([df_etching_plus_past_process, df_implatation], axis=1)
    
    # 중복열 제거
    df_etching_plus_past_process = df_etching_plus_past_process.loc[:,~df_etching_plus_past_process.columns.duplicated()]

    def dummy(scale_param, df) :
        df_dummy = pd.get_dummies(df)
    
        df_y = df_dummy[scale_param]
        df_x = df_dummy.drop(scale_param, axis=1)
    
        return df_x, df_y
    
    def kth_largest_number(arr, K):
        unique_nums = set(arr)
        sorted_nums = sorted(unique_nums, reverse=True)
        return sorted_nums[K-1]
    
    def temp_list_func(df_temp, temp_num, min_, max_, step_) :
        df_temp_list = pd.DataFrame()
    
        for i in np.arange(min_, max_, step_) :
            df_temp[temp_num] = i
            df_temp_list = pd.concat([df_temp_list, df_temp])
            
        return df_temp_list
    
    
    
    df_want_etching_rate = pd.DataFrame()
    df_want_etching_rate['want_etching_rate_f1'] = df_etching_plus_past_process['thickness'] * 10 - 5710
    df_want_etching_rate['want_etching_rate_f2'] = df_etching_plus_past_process['Thin F1'] - 3645
    df_want_etching_rate['want_etching_rate_f3'] = df_etching_plus_past_process['Thin F2'] - 1422
    df_want_etching_rate['want_etching_rate_f4'] = df_etching_plus_past_process['Thin F3'] - 213
    
    ####################################
    predict_temp_ethcing_lr = LinearRegression(fit_intercept=True)
    predict_temp_ethcing_lr = predict_temp_ethcing_lr.fit(df_want_etching_rate[['want_etching_rate_f1']], df_etching_plus_past_process['Temp_Etching'])
    
    
    ######## temp_etching1 ########
    df_for_temp_etching1 = df_etching_plus_past_process.drop(['Thin F4', 'Thin F3', 'Thin F2',  'Thin F1', 'Temp_Etching',
                                                              'etching_rate_f2', 'etching_rate_f3', 'etching_rate_f4',],axis=1)
    
    df_for_temp_etching1['Temp_Etching'] = df_etching_plus_past_process['Temp_Etching']
    
    ######## temp_etching2 ########
    df_for_temp_etching2 = df_etching_plus_past_process.drop(['Thin F4', 'Thin F3', 'Thin F2',
                                                              'etching_rate_f3', 'etching_rate_f4',],axis=1)
    
    df_for_temp_etching2['Temp_Etching2'] = predict_temp_ethcing_lr.predict(df_want_etching_rate[['want_etching_rate_f2']])
    
    ######## temp_etching3 ########
    df_for_temp_etching3 = df_etching_plus_past_process.drop(['Thin F4', 'Thin F3',
                                                              'etching_rate_f4',],axis=1)
    
    df_for_temp_etching3['Temp_Etching2'] = predict_temp_ethcing_lr.predict(df_want_etching_rate[['want_etching_rate_f2']])
    df_for_temp_etching3['Temp_Etching3'] = predict_temp_ethcing_lr.predict(df_want_etching_rate[['want_etching_rate_f3']])
    
    ######## temp_etching4 ########
    df_for_temp_etching4 = df_etching_plus_past_process.drop(['Thin F4'],axis=1)
    
    df_for_temp_etching4['Temp_Etching2'] = predict_temp_ethcing_lr.predict(df_want_etching_rate[['want_etching_rate_f2']])
    df_for_temp_etching4['Temp_Etching3'] = predict_temp_ethcing_lr.predict(df_want_etching_rate[['want_etching_rate_f3']])
    df_for_temp_etching4['Temp_Etching4'] = predict_temp_ethcing_lr.predict(df_want_etching_rate[['want_etching_rate_f4']])
    
    # model 정의
    df_x = dummy('etching_rate_f1', df_for_temp_etching1.select_dtypes(exclude='object'))[0]
    df_y = dummy('etching_rate_f1', df_for_temp_etching1)[1]
    
    df_train_x, df_test_valid_x, df_train_y, df_test_valid_y  = train_test_split(df_x, df_y, test_size=0.6, random_state = 1234)
    df_test_x, df_valid_x, df_test_y, df_valid_y  = train_test_split(df_test_valid_x, df_test_valid_y, test_size=0.5, random_state = 1234)
    
    model_temp1_gb = GradientBoostingRegressor(random_state=1234, n_estimators=41 , min_samples_leaf=2, min_samples_split=34, max_depth=8,learning_rate=0.1)
    model_temp1_gb.fit(df_x,df_y)
    
    df_temp_list = temp_list_func(df_in, 'Temp_Etching', 68.15, 73.081, 0.001)
    
    
    temp_pred = model_temp1_gb.predict(dummy('etching_rate_f1',df_temp_list.select_dtypes(exclude='object'))[0])
    
    secondary_value = kth_largest_number(abs(temp_pred - want_etching_rate_), -1)
    index_secondary_value = list(abs(temp_pred - want_etching_rate_)).index(secondary_value)
    
        
    return round(float(df_temp_list['Temp_Etching'].iloc[[index_secondary_value]]),3)

def temp2(df_in, want_etching_rate_):
    df_all = pd.read_csv('./data/DataSet.csv')
    df_oxidation = pd.read_csv('./data/Oxidation.csv')
    df_softbake = pd.read_csv('./data/Photo-SoftBake.csv')
    df_lithography = pd.read_csv('./data/Photo-Lithography.csv')
    df_etching = pd.read_csv('./data/Etching.csv')
    df_implatation = pd.read_csv('./data/Implantation.csv')
    df_inspect = pd.read_csv('./data/Inspect.csv')
    df_lot = pd.read_csv('./data/Inspect.csv')
    
    # Lot_Num 18 이상만 불러오기
    df_all = df_all[df_lot['Lot_Num']>=18]
    df_oxidation = df_oxidation[df_lot['Lot_Num']>=18]
    df_softbake = df_softbake[df_lot['Lot_Num']>=18]
    df_lithography = df_lithography[df_lot['Lot_Num']>=18]
    df_etching = df_etching[df_lot['Lot_Num']>=18]
    df_implatation = df_implatation[df_lot['Lot_Num']>=18]
    df_inspect = df_inspect[df_lot['Lot_Num']>=18]
    
    # etching_rate 생성
    df_etching['etching_rate_f1'] = 10 * df_oxidation['thickness'] - df_etching['Thin F1']
    df_etching['etching_rate_f2'] = df_etching['Thin F1'] - df_etching['Thin F2']
    df_etching['etching_rate_f3'] = df_etching['Thin F2'] - df_etching['Thin F3']
    df_etching['etching_rate_f4'] = df_etching['Thin F3'] - df_etching['Thin F4']
    
    df_all['etching_rate_f1'] = 10 * df_oxidation['thickness'] - df_etching['Thin F1']
    df_all['etching_rate_f2'] = df_etching['Thin F1'] - df_etching['Thin F2']
    df_all['etching_rate_f3'] = df_etching['Thin F2'] - df_etching['Thin F3']
    df_all['etching_rate_f4'] = df_etching['Thin F3'] - df_etching['Thin F4']
    
    # spin rate 생성
    df_softbake['spin_rate_2'] = df_all['spin2'] - df_all['spin1']
    df_softbake['spin_rate_3'] = df_all['spin3'] - df_all['spin2']
    
    df_all['spin_rate_2'] = df_all['spin2'] - df_all['spin1']
    df_all['spin_rate_3'] = df_all['spin3'] - df_all['spin2']
    
    # flux rate 생성
    df_implatation['flux_rate_90s'] = df_all['Flux90s'] - df_all['Flux60s']
    df_implatation['flux_rate_160s'] = df_all['Flux160s'] - df_all['Flux90s']
    df_implatation['flux_rate_480s'] = df_all['Flux480s'] - df_all['Flux160s']
    df_implatation['flux_rate_840s'] = df_all['Flux840s'] - df_all['Flux480s']
    
    df_all['flux_rate_90s'] = df_all['Flux90s'] - df_all['Flux60s']
    df_all['flux_rate_160s'] = df_all['Flux160s'] - df_all['Flux90s']
    df_all['flux_rate_480s'] = df_all['Flux480s'] - df_all['Flux160s']
    df_all['flux_rate_840s'] = df_all['Flux840s'] - df_all['Flux480s']
    
    # 불필요 변수 삭제
    df_all.drop(['No_Die', 'Datetime', 'Lot_Num',
                 'Ox_Chamber', 'process', 'Vapor', 
                 'photo_soft_Chamber', 'process 2',
                 'lithography_Chamber', 'Lamp', 'UV_type',
                 'Etching_Chamber', 'Process 3',
                 'Chamber_Num', 'process4', 'Current', 'Datetime.1', 'Error_message','Yield'], axis=1, inplace=True)
    df_oxidation.drop(['No_Die', 'Datetime', 'Lot_Num',
                       'Ox_Chamber', 'process', 'Vapor'], axis=1, inplace=True)
    df_softbake.drop(['No_Die', 'Datetime', 'Lot_Num',
                      'photo_soft_Chamber', 'process 2'], axis=1, inplace=True)
    df_lithography.drop(['No_Die', 'Datetime', 'Lot_Num',
                         'lithography_Chamber', 'Lamp', 'UV_type',], axis=1, inplace=True)
    df_etching.drop(['No_Die', 'Datetime', 'Lot_Num',
                     'Etching_Chamber', 'Process 3',], axis=1, inplace=True)
    df_implatation.drop(['No_Die', 'Datetime', 'Lot_Num',
                         'Chamber_Num', 'process4'], axis=1, inplace=True)
    
    df_all['Wafer_Num'] = df_all['Wafer_Num'].astype('str')
    df_oxidation['Wafer_Num'] = df_oxidation['Wafer_Num'].astype('str')
    df_softbake['Wafer_Num'] = df_softbake['Wafer_Num'].astype('str')
    df_lithography['Wafer_Num'] = df_lithography['Wafer_Num'].astype('str')
    df_etching['Wafer_Num'] = df_etching['Wafer_Num'].astype('str')
    df_implatation['Wafer_Num'] = df_implatation['Wafer_Num'].astype('str')
    
    # 전 공정을 포함한 현 공정
    df_softbake_plus_past_process =  pd.concat([df_oxidation, df_softbake], axis=1)
    df_lithography_plus_softbake = pd.concat([df_softbake, df_lithography], axis=1)
    df_lithography_plus_past_process = pd.concat([df_softbake_plus_past_process, df_lithography], axis=1)
    df_etching_plus_past_process = pd.concat([df_lithography_plus_past_process, df_etching], axis=1)
    df_implatation_plus_past_process = pd.concat([df_etching_plus_past_process, df_implatation], axis=1)
    
    # 중복열 제거
    df_etching_plus_past_process = df_etching_plus_past_process.loc[:,~df_etching_plus_past_process.columns.duplicated()]

    def dummy(scale_param, df) :
        df_dummy = pd.get_dummies(df)
    
        df_y = df_dummy[scale_param]
        df_x = df_dummy.drop(scale_param, axis=1)
    
        return df_x, df_y
    
    def kth_largest_number(arr, K):
        unique_nums = set(arr)
        sorted_nums = sorted(unique_nums, reverse=True)
        return sorted_nums[K-1]
    
    def temp_list_func(df_temp, temp_num, min_, max_, step_) :
        df_temp_list = pd.DataFrame()
    
        for i in np.arange(min_, max_, step_) :
            df_temp[temp_num] = i
            df_temp_list = pd.concat([df_temp_list, df_temp])
            
        return df_temp_list
    
    
    
    df_want_etching_rate = pd.DataFrame()
    df_want_etching_rate['want_etching_rate_f1'] = df_etching_plus_past_process['thickness'] * 10 - 5710
    df_want_etching_rate['want_etching_rate_f2'] = df_etching_plus_past_process['Thin F1'] - 3645
    df_want_etching_rate['want_etching_rate_f3'] = df_etching_plus_past_process['Thin F2'] - 1422
    df_want_etching_rate['want_etching_rate_f4'] = df_etching_plus_past_process['Thin F3'] - 213
    
    ####################################
    predict_temp_ethcing_lr = LinearRegression(fit_intercept=True)
    predict_temp_ethcing_lr = predict_temp_ethcing_lr.fit(df_want_etching_rate[['want_etching_rate_f1']], df_etching_plus_past_process['Temp_Etching'])
    
    
    ######## temp_etching1 ########
    df_for_temp_etching1 = df_etching_plus_past_process.drop(['Thin F4', 'Thin F3', 'Thin F2',  'Thin F1', 'Temp_Etching',
                                                              'etching_rate_f2', 'etching_rate_f3', 'etching_rate_f4',],axis=1)
    
    df_for_temp_etching1['Temp_Etching'] = df_etching_plus_past_process['Temp_Etching']
    
    ######## temp_etching2 ########
    df_for_temp_etching2 = df_etching_plus_past_process.drop(['Thin F4', 'Thin F3', 'Thin F2',
                                                              'etching_rate_f3', 'etching_rate_f4',],axis=1)
    
    df_for_temp_etching2['Temp_Etching2'] = predict_temp_ethcing_lr.predict(df_want_etching_rate[['want_etching_rate_f2']])
    
    ######## temp_etching3 ########
    df_for_temp_etching3 = df_etching_plus_past_process.drop(['Thin F4', 'Thin F3',
                                                              'etching_rate_f4',],axis=1)
    
    df_for_temp_etching3['Temp_Etching2'] = predict_temp_ethcing_lr.predict(df_want_etching_rate[['want_etching_rate_f2']])
    df_for_temp_etching3['Temp_Etching3'] = predict_temp_ethcing_lr.predict(df_want_etching_rate[['want_etching_rate_f3']])
    
    ######## temp_etching4 ########
    df_for_temp_etching4 = df_etching_plus_past_process.drop(['Thin F4'],axis=1)
    
    df_for_temp_etching4['Temp_Etching2'] = predict_temp_ethcing_lr.predict(df_want_etching_rate[['want_etching_rate_f2']])
    df_for_temp_etching4['Temp_Etching3'] = predict_temp_ethcing_lr.predict(df_want_etching_rate[['want_etching_rate_f3']])
    df_for_temp_etching4['Temp_Etching4'] = predict_temp_ethcing_lr.predict(df_want_etching_rate[['want_etching_rate_f4']])
    
    # model 정의
    df_x = dummy('etching_rate_f2', df_for_temp_etching2.select_dtypes(exclude='object'))[0]
    df_y = dummy('etching_rate_f2', df_for_temp_etching2)[1]
    
    df_train_x, df_test_valid_x, df_train_y, df_test_valid_y  = train_test_split(df_x, df_y, test_size=0.6, random_state = 1234)
    df_test_x, df_valid_x, df_test_y, df_valid_y  = train_test_split(df_test_valid_x, df_test_valid_y, test_size=0.5, random_state = 1234)
    
    model_temp2_gb = GradientBoostingRegressor(random_state=1234, n_estimators=41 , min_samples_leaf=2, min_samples_split=34, max_depth=8,learning_rate=0.1)
    model_temp2_gb.fit(df_x,df_y)
    
    df_temp_list = temp_list_func(df_for_temp_etching2.iloc[[0]], 'Temp_Etching2', 77.43, 78.19, 0.001)
    
    
    temp_pred = model_temp2_gb.predict(dummy('etching_rate_f2',df_temp_list.select_dtypes(exclude='object'))[0])
    
    secondary_value = kth_largest_number(abs(temp_pred - want_etching_rate_), -1)
    index_secondary_value = list(abs(temp_pred - want_etching_rate_)).index(secondary_value)


    return round(float(df_temp_list['Temp_Etching2'].iloc[[index_secondary_value]]),3)

def temp3(df_in, want_etching_rate_):
    df_all = pd.read_csv('./data/DataSet.csv')
    df_oxidation = pd.read_csv('./data/Oxidation.csv')
    df_softbake = pd.read_csv('./data/Photo-SoftBake.csv')
    df_lithography = pd.read_csv('./data/Photo-Lithography.csv')
    df_etching = pd.read_csv('./data/Etching.csv')
    df_implatation = pd.read_csv('./data/Implantation.csv')
    df_inspect = pd.read_csv('./data/Inspect.csv')
    df_lot = pd.read_csv('./data/Inspect.csv')
    
    # Lot_Num 18 이상만 불러오기
    df_all = df_all[df_lot['Lot_Num']>=18]
    df_oxidation = df_oxidation[df_lot['Lot_Num']>=18]
    df_softbake = df_softbake[df_lot['Lot_Num']>=18]
    df_lithography = df_lithography[df_lot['Lot_Num']>=18]
    df_etching = df_etching[df_lot['Lot_Num']>=18]
    df_implatation = df_implatation[df_lot['Lot_Num']>=18]
    df_inspect = df_inspect[df_lot['Lot_Num']>=18]
    
    # etching_rate 생성
    df_etching['etching_rate_f1'] = 10 * df_oxidation['thickness'] - df_etching['Thin F1']
    df_etching['etching_rate_f2'] = df_etching['Thin F1'] - df_etching['Thin F2']
    df_etching['etching_rate_f3'] = df_etching['Thin F2'] - df_etching['Thin F3']
    df_etching['etching_rate_f4'] = df_etching['Thin F3'] - df_etching['Thin F4']
    
    df_all['etching_rate_f1'] = 10 * df_oxidation['thickness'] - df_etching['Thin F1']
    df_all['etching_rate_f2'] = df_etching['Thin F1'] - df_etching['Thin F2']
    df_all['etching_rate_f3'] = df_etching['Thin F2'] - df_etching['Thin F3']
    df_all['etching_rate_f4'] = df_etching['Thin F3'] - df_etching['Thin F4']
    
    # spin rate 생성
    df_softbake['spin_rate_2'] = df_all['spin2'] - df_all['spin1']
    df_softbake['spin_rate_3'] = df_all['spin3'] - df_all['spin2']
    
    df_all['spin_rate_2'] = df_all['spin2'] - df_all['spin1']
    df_all['spin_rate_3'] = df_all['spin3'] - df_all['spin2']
    
    # flux rate 생성
    df_implatation['flux_rate_90s'] = df_all['Flux90s'] - df_all['Flux60s']
    df_implatation['flux_rate_160s'] = df_all['Flux160s'] - df_all['Flux90s']
    df_implatation['flux_rate_480s'] = df_all['Flux480s'] - df_all['Flux160s']
    df_implatation['flux_rate_840s'] = df_all['Flux840s'] - df_all['Flux480s']
    
    df_all['flux_rate_90s'] = df_all['Flux90s'] - df_all['Flux60s']
    df_all['flux_rate_160s'] = df_all['Flux160s'] - df_all['Flux90s']
    df_all['flux_rate_480s'] = df_all['Flux480s'] - df_all['Flux160s']
    df_all['flux_rate_840s'] = df_all['Flux840s'] - df_all['Flux480s']
    
    # 불필요 변수 삭제
    df_all.drop(['No_Die', 'Datetime', 'Lot_Num',
                 'Ox_Chamber', 'process', 'Vapor', 
                 'photo_soft_Chamber', 'process 2',
                 'lithography_Chamber', 'Lamp', 'UV_type',
                 'Etching_Chamber', 'Process 3',
                 'Chamber_Num', 'process4', 'Current', 'Datetime.1', 'Error_message','Yield'], axis=1, inplace=True)
    df_oxidation.drop(['No_Die', 'Datetime', 'Lot_Num',
                       'Ox_Chamber', 'process', 'Vapor'], axis=1, inplace=True)
    df_softbake.drop(['No_Die', 'Datetime', 'Lot_Num',
                      'photo_soft_Chamber', 'process 2'], axis=1, inplace=True)
    df_lithography.drop(['No_Die', 'Datetime', 'Lot_Num',
                         'lithography_Chamber', 'Lamp', 'UV_type',], axis=1, inplace=True)
    df_etching.drop(['No_Die', 'Datetime', 'Lot_Num',
                     'Etching_Chamber', 'Process 3',], axis=1, inplace=True)
    df_implatation.drop(['No_Die', 'Datetime', 'Lot_Num',
                         'Chamber_Num', 'process4'], axis=1, inplace=True)
    
    df_all['Wafer_Num'] = df_all['Wafer_Num'].astype('str')
    df_oxidation['Wafer_Num'] = df_oxidation['Wafer_Num'].astype('str')
    df_softbake['Wafer_Num'] = df_softbake['Wafer_Num'].astype('str')
    df_lithography['Wafer_Num'] = df_lithography['Wafer_Num'].astype('str')
    df_etching['Wafer_Num'] = df_etching['Wafer_Num'].astype('str')
    df_implatation['Wafer_Num'] = df_implatation['Wafer_Num'].astype('str')
    
    # 전 공정을 포함한 현 공정
    df_softbake_plus_past_process =  pd.concat([df_oxidation, df_softbake], axis=1)
    df_lithography_plus_softbake = pd.concat([df_softbake, df_lithography], axis=1)
    df_lithography_plus_past_process = pd.concat([df_softbake_plus_past_process, df_lithography], axis=1)
    df_etching_plus_past_process = pd.concat([df_lithography_plus_past_process, df_etching], axis=1)
    df_implatation_plus_past_process = pd.concat([df_etching_plus_past_process, df_implatation], axis=1)
    
    # 중복열 제거
    df_etching_plus_past_process = df_etching_plus_past_process.loc[:,~df_etching_plus_past_process.columns.duplicated()]

    def dummy(scale_param, df) :
        df_dummy = pd.get_dummies(df)
    
        df_y = df_dummy[scale_param]
        df_x = df_dummy.drop(scale_param, axis=1)
    
        return df_x, df_y
    
    def kth_largest_number(arr, K):
        unique_nums = set(arr)
        sorted_nums = sorted(unique_nums, reverse=True)
        return sorted_nums[K-1]
    
    def temp_list_func(df_temp, temp_num, min_, max_, step_) :
        df_temp_list = pd.DataFrame()
    
        for i in np.arange(min_, max_, step_) :
            df_temp[temp_num] = i
            df_temp_list = pd.concat([df_temp_list, df_temp])
            
        return df_temp_list
    
    
    
    df_want_etching_rate = pd.DataFrame()
    df_want_etching_rate['want_etching_rate_f1'] = df_etching_plus_past_process['thickness'] * 10 - 5710
    df_want_etching_rate['want_etching_rate_f2'] = df_etching_plus_past_process['Thin F1'] - 3645
    df_want_etching_rate['want_etching_rate_f3'] = df_etching_plus_past_process['Thin F2'] - 1422
    df_want_etching_rate['want_etching_rate_f4'] = df_etching_plus_past_process['Thin F3'] - 213
    
    ####################################
    predict_temp_ethcing_lr = LinearRegression(fit_intercept=True)
    predict_temp_ethcing_lr = predict_temp_ethcing_lr.fit(df_want_etching_rate[['want_etching_rate_f1']], df_etching_plus_past_process['Temp_Etching'])
    
    
    ######## temp_etching1 ########
    df_for_temp_etching1 = df_etching_plus_past_process.drop(['Thin F4', 'Thin F3', 'Thin F2',  'Thin F1', 'Temp_Etching',
                                                              'etching_rate_f2', 'etching_rate_f3', 'etching_rate_f4',],axis=1)
    
    df_for_temp_etching1['Temp_Etching'] = df_etching_plus_past_process['Temp_Etching']
    
    ######## temp_etching2 ########
    df_for_temp_etching2 = df_etching_plus_past_process.drop(['Thin F4', 'Thin F3', 'Thin F2',
                                                              'etching_rate_f3', 'etching_rate_f4',],axis=1)
    
    df_for_temp_etching2['Temp_Etching2'] = predict_temp_ethcing_lr.predict(df_want_etching_rate[['want_etching_rate_f2']])
    
    ######## temp_etching3 ########
    df_for_temp_etching3 = df_etching_plus_past_process.drop(['Thin F4', 'Thin F3',
                                                              'etching_rate_f4',],axis=1)
    
    df_for_temp_etching3['Temp_Etching2'] = predict_temp_ethcing_lr.predict(df_want_etching_rate[['want_etching_rate_f2']])
    df_for_temp_etching3['Temp_Etching3'] = predict_temp_ethcing_lr.predict(df_want_etching_rate[['want_etching_rate_f3']])
    
    ######## temp_etching4 ########
    df_for_temp_etching4 = df_etching_plus_past_process.drop(['Thin F4'],axis=1)
    
    df_for_temp_etching4['Temp_Etching2'] = predict_temp_ethcing_lr.predict(df_want_etching_rate[['want_etching_rate_f2']])
    df_for_temp_etching4['Temp_Etching3'] = predict_temp_ethcing_lr.predict(df_want_etching_rate[['want_etching_rate_f3']])
    df_for_temp_etching4['Temp_Etching4'] = predict_temp_ethcing_lr.predict(df_want_etching_rate[['want_etching_rate_f4']])
    
    # model 정의
    df_x = dummy('etching_rate_f3', df_for_temp_etching3.select_dtypes(exclude='object'))[0]
    df_y = dummy('etching_rate_f3', df_for_temp_etching3)[1]
    
    df_train_x, df_test_valid_x, df_train_y, df_test_valid_y  = train_test_split(df_x, df_y, test_size=0.6, random_state = 1234)
    df_test_x, df_valid_x, df_test_y, df_valid_y  = train_test_split(df_test_valid_x, df_test_valid_y, test_size=0.5, random_state = 1234)
    
    model_temp3_gb = GradientBoostingRegressor(random_state=1234, n_estimators=41 , min_samples_leaf=2, min_samples_split=34, max_depth=8,learning_rate=0.1)
    model_temp3_gb.fit(df_x,df_y)
    
    df_temp_list = temp_list_func(df_for_temp_etching3.iloc[[0]], 'Temp_Etching3', 78.96, 79.89, 0.001)
    
    
    temp_pred = model_temp3_gb.predict(dummy('etching_rate_f3',df_temp_list.select_dtypes(exclude='object'))[0])
    
    secondary_value = kth_largest_number(abs(temp_pred - want_etching_rate_), -1)
    index_secondary_value = list(abs(temp_pred - want_etching_rate_)).index(secondary_value)


    return round(float(df_temp_list['Temp_Etching3'].iloc[[index_secondary_value]]),3)


def temp4(df_in, want_etching_rate_):
    df_all = pd.read_csv('./data/DataSet.csv')
    df_oxidation = pd.read_csv('./data/Oxidation.csv')
    df_softbake = pd.read_csv('./data/Photo-SoftBake.csv')
    df_lithography = pd.read_csv('./data/Photo-Lithography.csv')
    df_etching = pd.read_csv('./data/Etching.csv')
    df_implatation = pd.read_csv('./data/Implantation.csv')
    df_inspect = pd.read_csv('./data/Inspect.csv')
    df_lot = pd.read_csv('./data/Inspect.csv')
    
    # Lot_Num 18 이상만 불러오기
    df_all = df_all[df_lot['Lot_Num']>=18]
    df_oxidation = df_oxidation[df_lot['Lot_Num']>=18]
    df_softbake = df_softbake[df_lot['Lot_Num']>=18]
    df_lithography = df_lithography[df_lot['Lot_Num']>=18]
    df_etching = df_etching[df_lot['Lot_Num']>=18]
    df_implatation = df_implatation[df_lot['Lot_Num']>=18]
    df_inspect = df_inspect[df_lot['Lot_Num']>=18]
    
    # etching_rate 생성
    df_etching['etching_rate_f1'] = 10 * df_oxidation['thickness'] - df_etching['Thin F1']
    df_etching['etching_rate_f2'] = df_etching['Thin F1'] - df_etching['Thin F2']
    df_etching['etching_rate_f3'] = df_etching['Thin F2'] - df_etching['Thin F3']
    df_etching['etching_rate_f4'] = df_etching['Thin F3'] - df_etching['Thin F4']
    
    df_all['etching_rate_f1'] = 10 * df_oxidation['thickness'] - df_etching['Thin F1']
    df_all['etching_rate_f2'] = df_etching['Thin F1'] - df_etching['Thin F2']
    df_all['etching_rate_f3'] = df_etching['Thin F2'] - df_etching['Thin F3']
    df_all['etching_rate_f4'] = df_etching['Thin F3'] - df_etching['Thin F4']
    
    # spin rate 생성
    df_softbake['spin_rate_2'] = df_all['spin2'] - df_all['spin1']
    df_softbake['spin_rate_3'] = df_all['spin3'] - df_all['spin2']
    
    df_all['spin_rate_2'] = df_all['spin2'] - df_all['spin1']
    df_all['spin_rate_3'] = df_all['spin3'] - df_all['spin2']
    
    # flux rate 생성
    df_implatation['flux_rate_90s'] = df_all['Flux90s'] - df_all['Flux60s']
    df_implatation['flux_rate_160s'] = df_all['Flux160s'] - df_all['Flux90s']
    df_implatation['flux_rate_480s'] = df_all['Flux480s'] - df_all['Flux160s']
    df_implatation['flux_rate_840s'] = df_all['Flux840s'] - df_all['Flux480s']
    
    df_all['flux_rate_90s'] = df_all['Flux90s'] - df_all['Flux60s']
    df_all['flux_rate_160s'] = df_all['Flux160s'] - df_all['Flux90s']
    df_all['flux_rate_480s'] = df_all['Flux480s'] - df_all['Flux160s']
    df_all['flux_rate_840s'] = df_all['Flux840s'] - df_all['Flux480s']
    
    # 불필요 변수 삭제
    df_all.drop(['No_Die', 'Datetime', 'Lot_Num',
                 'Ox_Chamber', 'process', 'Vapor', 
                 'photo_soft_Chamber', 'process 2',
                 'lithography_Chamber', 'Lamp', 'UV_type',
                 'Etching_Chamber', 'Process 3',
                 'Chamber_Num', 'process4', 'Current', 'Datetime.1', 'Error_message','Yield'], axis=1, inplace=True)
    df_oxidation.drop(['No_Die', 'Datetime', 'Lot_Num',
                       'Ox_Chamber', 'process', 'Vapor'], axis=1, inplace=True)
    df_softbake.drop(['No_Die', 'Datetime', 'Lot_Num',
                      'photo_soft_Chamber', 'process 2'], axis=1, inplace=True)
    df_lithography.drop(['No_Die', 'Datetime', 'Lot_Num',
                         'lithography_Chamber', 'Lamp', 'UV_type',], axis=1, inplace=True)
    df_etching.drop(['No_Die', 'Datetime', 'Lot_Num',
                     'Etching_Chamber', 'Process 3',], axis=1, inplace=True)
    df_implatation.drop(['No_Die', 'Datetime', 'Lot_Num',
                         'Chamber_Num', 'process4'], axis=1, inplace=True)
    
    df_all['Wafer_Num'] = df_all['Wafer_Num'].astype('str')
    df_oxidation['Wafer_Num'] = df_oxidation['Wafer_Num'].astype('str')
    df_softbake['Wafer_Num'] = df_softbake['Wafer_Num'].astype('str')
    df_lithography['Wafer_Num'] = df_lithography['Wafer_Num'].astype('str')
    df_etching['Wafer_Num'] = df_etching['Wafer_Num'].astype('str')
    df_implatation['Wafer_Num'] = df_implatation['Wafer_Num'].astype('str')
    
    # 전 공정을 포함한 현 공정
    df_softbake_plus_past_process =  pd.concat([df_oxidation, df_softbake], axis=1)
    df_lithography_plus_softbake = pd.concat([df_softbake, df_lithography], axis=1)
    df_lithography_plus_past_process = pd.concat([df_softbake_plus_past_process, df_lithography], axis=1)
    df_etching_plus_past_process = pd.concat([df_lithography_plus_past_process, df_etching], axis=1)
    df_implatation_plus_past_process = pd.concat([df_etching_plus_past_process, df_implatation], axis=1)
    
    # 중복열 제거
    df_etching_plus_past_process = df_etching_plus_past_process.loc[:,~df_etching_plus_past_process.columns.duplicated()]

    def dummy(scale_param, df) :
        df_dummy = pd.get_dummies(df)
    
        df_y = df_dummy[scale_param]
        df_x = df_dummy.drop(scale_param, axis=1)
    
        return df_x, df_y
    
    def kth_largest_number(arr, K):
        unique_nums = set(arr)
        sorted_nums = sorted(unique_nums, reverse=True)
        return sorted_nums[K-1]
    
    def temp_list_func(df_temp, temp_num, min_, max_, step_) :
        df_temp_list = pd.DataFrame()
    
        for i in np.arange(min_, max_, step_) :
            df_temp[temp_num] = i
            df_temp_list = pd.concat([df_temp_list, df_temp])
            
        return df_temp_list
    
    
    
    df_want_etching_rate = pd.DataFrame()
    df_want_etching_rate['want_etching_rate_f1'] = df_etching_plus_past_process['thickness'] * 10 - 5710
    df_want_etching_rate['want_etching_rate_f2'] = df_etching_plus_past_process['Thin F1'] - 3645
    df_want_etching_rate['want_etching_rate_f3'] = df_etching_plus_past_process['Thin F2'] - 1422
    df_want_etching_rate['want_etching_rate_f4'] = df_etching_plus_past_process['Thin F3'] - 213
    
    ####################################
    predict_temp_ethcing_lr = LinearRegression(fit_intercept=True)
    predict_temp_ethcing_lr = predict_temp_ethcing_lr.fit(df_want_etching_rate[['want_etching_rate_f1']], df_etching_plus_past_process['Temp_Etching'])
    
    
    ######## temp_etching1 ########
    df_for_temp_etching1 = df_etching_plus_past_process.drop(['Thin F4', 'Thin F3', 'Thin F2',  'Thin F1', 'Temp_Etching',
                                                              'etching_rate_f2', 'etching_rate_f3', 'etching_rate_f4',],axis=1)
    
    df_for_temp_etching1['Temp_Etching'] = df_etching_plus_past_process['Temp_Etching']
    
    ######## temp_etching2 ########
    df_for_temp_etching2 = df_etching_plus_past_process.drop(['Thin F4', 'Thin F3', 'Thin F2',
                                                              'etching_rate_f3', 'etching_rate_f4',],axis=1)
    
    df_for_temp_etching2['Temp_Etching2'] = predict_temp_ethcing_lr.predict(df_want_etching_rate[['want_etching_rate_f2']])
    
    ######## temp_etching3 ########
    df_for_temp_etching3 = df_etching_plus_past_process.drop(['Thin F4', 'Thin F3',
                                                              'etching_rate_f4',],axis=1)
    
    df_for_temp_etching3['Temp_Etching2'] = predict_temp_ethcing_lr.predict(df_want_etching_rate[['want_etching_rate_f2']])
    df_for_temp_etching3['Temp_Etching3'] = predict_temp_ethcing_lr.predict(df_want_etching_rate[['want_etching_rate_f3']])
    
    ######## temp_etching4 ########
    df_for_temp_etching4 = df_etching_plus_past_process.drop(['Thin F4'],axis=1)
    
    df_for_temp_etching4['Temp_Etching2'] = predict_temp_ethcing_lr.predict(df_want_etching_rate[['want_etching_rate_f2']])
    df_for_temp_etching4['Temp_Etching3'] = predict_temp_ethcing_lr.predict(df_want_etching_rate[['want_etching_rate_f3']])
    df_for_temp_etching4['Temp_Etching4'] = predict_temp_ethcing_lr.predict(df_want_etching_rate[['want_etching_rate_f4']])
    
    # model 정의
    df_x = dummy('etching_rate_f4', df_for_temp_etching4.select_dtypes(exclude='object'))[0]
    df_y = dummy('etching_rate_f4', df_for_temp_etching4)[1]
    
    df_train_x, df_test_valid_x, df_train_y, df_test_valid_y  = train_test_split(df_x, df_y, test_size=0.6, random_state = 1234)
    df_test_x, df_valid_x, df_test_y, df_valid_y  = train_test_split(df_test_valid_x, df_test_valid_y, test_size=0.5, random_state = 1234)
    
    model_temp4_gb = GradientBoostingRegressor(random_state=1234, n_estimators=41 , min_samples_leaf=2, min_samples_split=34, max_depth=8,learning_rate=0.1)
    model_temp4_gb.fit(df_x,df_y)
    
    df_temp_list = temp_list_func(df_for_temp_etching4.iloc[[0]], 'Temp_Etching4', 67.44, 72.314, 0.001)
    
    
    temp_pred = model_temp4_gb.predict(dummy('etching_rate_f4',df_temp_list.select_dtypes(exclude='object'))[0])
    
    secondary_value = kth_largest_number(abs(temp_pred - want_etching_rate_), -1)
    index_secondary_value = list(abs(temp_pred - want_etching_rate_)).index(secondary_value)


    return round(float(df_temp_list['Temp_Etching4'].iloc[[index_secondary_value]]),3)