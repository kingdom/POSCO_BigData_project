from flask import Flask
from flask import render_template
from flask import request
from models import dbMgr
import pandas as pd 
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib 

app = Flask(__name__,static_folder='./static')

@app.route('/')
def index():
    return render_template("/index.html")

@app.route('/admin')
def admin():
    return render_template("/admin.html")

@app.route('/predict_page')
def page():
    return render_template('predict_page.html')
@app.route('/predict_page2')
def page2():
    return render_template('predict_page2.html')
@app.route('/predict_page3')
def page3():
    return render_template('predict_page3.html')
@app.route('/predict_page4')
def page4():
    return render_template('predict_page4.html')    



#predict01

@app.route('/predict01',  methods=['POST'] )
def predict_data():
    Wafer_Num = request.form['Wafer_Num']
    type_ = request.form['type_']
    Temp_OXid = request.form['Temp_OXid']
    ppm = request.form['ppm']
    Pressure = request.form['Pressure']
    Oxid_time = request.form['Oxid_time']
    thickness = request.form['thickness']
    resist_target = request.form['resist_target']
    N2_HMDS = request.form['N2_HMDS']
    pressure_HMDS = request.form['pressure_HMDS']
    temp_HMDS = request.form['temp_HMDS']
    temp_HMDS_bake = request.form['temp_HMDS_bake']
    time_HMDS_bake = request.form['time_HMDS_bake']
    spin1 = request.form['spin1']
    spin2 = request.form['spin2']
    spin3 = request.form['spin3']
    photoresist_bake = request.form['photoresist_bake']
    temp_softbake = request.form['temp_softbake']
    time_softbake = request.form['time_softbake']
    spin_rate_2 = float(spin2) - float(spin1)
    spin_rate_3 = float(spin3) - float(spin2)
    Line_CD = request.form['Line_CD']
    Wavelength = request.form['Wavelength']
    Resolution = request.form['Resolution']
    Energy_Exposure = request.form['Energy_Exposure']
    Source_Power = request.form['Source_Power']
    Selectivity = request.form['Selectivity']
    etching_rate_f1 = 1
    Temp_Etching = 1
    
    global df
    
    df = pd.DataFrame()
    df['Wafer_Num'] = [Wafer_Num]
    df['type_'] = [type_]
    df['Temp_OXid'] = [float(Temp_OXid)]
    df['ppm'] = [float(ppm)]
    df['Pressure'] = [float(Pressure)]
    df['Oxid_time'] = [float(Oxid_time)]
    df['thickness'] = [float(thickness)]
    df['resist_target'] = [float(resist_target)]
    df['N2_HMDS'] = [float(N2_HMDS)]
    df['pressure_HMDS'] = [float(pressure_HMDS)]
    df['temp_HMDS'] = [float(temp_HMDS)]
    df['temp_HMDS_bake'] = [float(temp_HMDS_bake)]
    df['time_HMDS_bake'] = [float(time_HMDS_bake)]
    df['spin1'] = [float(spin1)]
    df['spin2'] = [float(spin2)]
    df['spin3'] = [float(spin3)]
    df['photoresist_bake'] = [float(photoresist_bake)]
    df['temp_softbake'] = [float(temp_softbake)]
    df['time_softbake'] = [float(time_softbake)]
    df['spin_rate_2'] = [float(spin_rate_2)]
    df['spin_rate_3'] = [float(spin_rate_3)]
    df['Line_CD'] = [float(Line_CD)]
    df['Wavelength'] = [float(Wavelength)]
    df['Resolution'] = [float(Resolution)]
    df['Energy_Exposure'] = [float(Energy_Exposure)]
    df['Source_Power'] = [float(Source_Power)]
    df['Selectivity'] = [float(Selectivity)]
    df['etching_rate_f1'] = [float(etching_rate_f1)]
    df['Temp_Etching'] = [float(Temp_Etching)]
    

    predict01 = dbMgr.temp1(df, 10 * float(df['thickness']) - 5710)
    df['Temp_Etching'] = [predict01]

   #그림1
    plt.style.use(['ggplot'])
    np.random.seed(seed = 1)
    normal = np.random.normal(78, 0.0028, 1000000)
    ax = sns.distplot(normal, hist=False)
    ax.get_xaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.get_yaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.grid(b=True, which='major', color='w', linewidth=1.0)
    ax.grid(b=True, which='minor', color='w', linewidth=0.5)  
    plt.savefig('./static/img/test1.png')


    return render_template('predict_page.html', predict01=predict01)


#predict02

@app.route('/predict02',  methods=['POST'] )
def predict_data02(): 
    Thin_F1= request.form['Thin_F1']
    df['Thin_F1'] = [float(Thin_F1)]
    df['etching_rate_f1'] = [float(df['thickness'].iloc[[0]])*10 - float(Thin_F1)]
    df['etching_rate_f2'] = [1]
    df['Temp_Etching2'] = [1]
    global df1
    
    df1 = df[['Wafer_Num', 'type_', 'Temp_OXid', 'ppm', 'Pressure', 'Oxid_time',
       'thickness', 'resist_target', 'N2_HMDS', 'pressure_HMDS', 'temp_HMDS',
       'temp_HMDS_bake', 'time_HMDS_bake', 'spin1', 'spin2', 'spin3',
       'photoresist_bake', 'temp_softbake', 'time_softbake', 'spin_rate_2',
       'spin_rate_3', 'Line_CD', 'Wavelength', 'Resolution', 'Energy_Exposure',
       'Temp_Etching', 'Source_Power', 'Selectivity', 'Thin_F1',
       'etching_rate_f1', 'etching_rate_f2', 'Temp_Etching2']]
    
    predict02 = dbMgr.temp2(df1, float(df1['Thin_F1']) - 3645)
    df1['Temp_Etching2'] = [predict02]
   
   #그림2
    plt.style.use(['ggplot'])
    np.random.seed(seed = 1)
    normal = np.random.normal(78, 0.0028, 1000000)
    ax = sns.distplot(normal, hist=False)
    ax.get_xaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.get_yaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.grid(b=True, which='major', color='w', linewidth=1.0)
    ax.grid(b=True, which='minor', color='w', linewidth=0.5)  
    plt.savefig('./static/img/test2.png')

    return render_template('predict_page2.html', predict02=predict02)
    
#predict03

@app.route('/predict03',  methods=['POST'] )
def predict_data03():
    Thin_F2= request.form['Thin_F2']
    df1['Thin_F2'] = [float(Thin_F2)]
    df1['etching_rate_f2'] = [float(df1['Thin_F1'].iloc[[0]]) - float(Thin_F2)]
    df1['etching_rate_f3'] = [1]
    df1['Temp_Etching3'] = [1]
    global df2
    
    df2 = df1[['Wafer_Num', 'type_', 'Temp_OXid', 'ppm', 'Pressure', 'Oxid_time',
       'thickness', 'resist_target', 'N2_HMDS', 'pressure_HMDS', 'temp_HMDS',
       'temp_HMDS_bake', 'time_HMDS_bake', 'spin1', 'spin2', 'spin3',
       'photoresist_bake', 'temp_softbake', 'time_softbake', 'spin_rate_2',
       'spin_rate_3', 'Line_CD', 'Wavelength', 'Resolution', 'Energy_Exposure',
       'Thin_F2', 'Thin_F1', 'Temp_Etching', 'Source_Power', 'Selectivity',
       'etching_rate_f1', 'etching_rate_f2', 'etching_rate_f3',
       'Temp_Etching2', 'Temp_Etching3']]
    
    predict03 = dbMgr.temp3(df2, float(df2['Thin_F2']) - 1422)
    df2['Temp_Etching3'] = [predict03]
    
   #그림3
    plt.style.use(['ggplot'])
    np.random.seed(seed = 1)
    normal = np.random.normal(78, 0.0028, 1000000)
    ax = sns.distplot(normal, hist=False)
    ax.get_xaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.get_yaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.grid(b=True, which='major', color='w', linewidth=1.0)
    ax.grid(b=True, which='minor', color='w', linewidth=0.5)  
    plt.savefig('./static/img/test3.png')

    return render_template('predict_page3.html', predict03=predict03)


#predict04

@app.route('/predict04',  methods=['POST'] )
def predict_data04():
    Thin_F3= request.form['Thin_F3']
    df2['Thin_F3'] = [float(Thin_F3)]
    df2['etching_rate_f3'] = [float(df2['Thin_F2'].iloc[[0]]) - float(Thin_F3)]
    df2['etching_rate_f4'] = [1]
    df2['Temp_Etching4'] = [1]
    global df3
    
    df3 = df2[['Wafer_Num', 'type_', 'Temp_OXid', 'ppm', 'Pressure', 'Oxid_time',
       'thickness', 'resist_target', 'N2_HMDS', 'pressure_HMDS', 'temp_HMDS',
       'temp_HMDS_bake', 'time_HMDS_bake', 'spin1', 'spin2', 'spin3',
       'photoresist_bake', 'temp_softbake', 'time_softbake', 'spin_rate_2',
       'spin_rate_3', 'Line_CD', 'Wavelength', 'Resolution', 'Energy_Exposure',
       'Temp_Etching', 'Source_Power', 'Selectivity', 'Thin_F1', 'Thin_F2',
       'Thin_F3', 'etching_rate_f1', 'etching_rate_f2', 'etching_rate_f3',
       'etching_rate_f4', 'Temp_Etching2', 'Temp_Etching3', 'Temp_Etching4']]
    
    predict04 = dbMgr.temp4(df3, float(df3['Thin_F3']) - 213)

    #그림4
    plt.style.use(['ggplot'])
    np.random.seed(seed = 1)
    normal = np.random.normal(78, 0.0028, 1000000)
    ax = sns.distplot(normal, hist=False)
    ax.get_xaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.get_yaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.grid(b=True, which='major', color='w', linewidth=1.0)
    ax.grid(b=True, which='minor', color='w', linewidth=0.5)  
    plt.savefig('./static/img/test4.png')


    return render_template('predict_page4.html', predict04=predict04)

if __name__ == '__main__':
    app.run()
