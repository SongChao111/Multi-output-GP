import numpy as np
import pandas as pd
from eppy import modeleditor
from eppy.modeleditor import IDF
import subprocess
import csv

def LHSample( D,bounds,N):
    ''' :param D:参数个数 :param bounds:参数对应范围（list） :param N:拉丁超立方层数 :return:样本数据 '''

    result = np.empty([N, D])
    temp = np.empty([N])
    d = 1.0 / N

    for i in range(D):

        for j in range(N):
            temp[j] = np.random.uniform(
                low=j * d, high=(j + 1) * d, size = 1)[0]

        np.random.shuffle(temp)

        for j in range(N):
            result[j, i] = temp[j]

    #对样本数据进行拉伸
    b = np.array(bounds)
    lower_bounds = b[:,0]
    upper_bounds = b[:,1]
    if np.any(lower_bounds > upper_bounds):
        print('范围出错')
        return None

    np.add(np.multiply(result,
                       (upper_bounds - lower_bounds),
                       out=result),
           lower_bounds,
           out=result)
    return result

def comp_data_reader(eso_file, yc_keys, xc_keys):
    with open(eso_file) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        Eplusout = []
        for row in readCSV:
            Eplusout.append(row)

    yc_mtr_number=[]
    for i in range(len(yc_keys)):
        for row in Eplusout:
            if len(row)>3:
                if yc_keys[i] in row[2]:
                    yc_mtr_number.append(row[0])

    yc_mtr_values=[]
    for i in range(len(yc_mtr_number)):
        yc_mtr_value=[]
        for row in Eplusout:
            if len(row)>3:
                if yc_mtr_number[i] == row[0]:
                    if yc_keys[i] not in row[2]:
                        yc_mtr_value.append(float(row[1])/3600000)
        yc_mtr_values.append(yc_mtr_value)

    xc_mtr_number=[]
    for i in range(len(xc_keys)):
        for row in Eplusout:
            if len(row)>3:
                if xc_keys[i] in row[3]:
                    xc_mtr_number.append(row[0])

    xc_mtr_values=[]
    for i in range(len(xc_mtr_number)):
        xc_mtr_value=[]
        for row in Eplusout:
            if len(row)>3:
                if xc_mtr_number[i] == row[0]:
                    if xc_keys[i] not in row[3]:
                        xc_mtr_value.append(float(row[1]))
        xc_mtr_values.append(xc_mtr_value)
    return [yc_mtr_values,xc_mtr_values]

def datafield(yc_keys,xc_keys):
    iddfile = "./Energy+9.1.idd"
    IDF.setiddname(iddfile)


    idfname = "./RefBldgLargeOfficeNew2004_Chicago.idf"
    idf = IDF(idfname)

    # change the output variable and meter
    # output_frequency='Daily'
    output_frequency='Monthly'

    variable=[]
    for i in range(len(xc_keys)):
        variable1 = idf.newidfobject("Output:Variable".upper())
        variable1.Key_Value = '*'
        variable1.Variable_Name = xc_keys[i]
        variable1.Reporting_Frequency = output_frequency
        variable.append(variable1)
    idf.idfobjects['Output:Variable'.upper()]=variable

    meter=[]
    for i in range(len(yc_keys)):
        meter1 = idf.newidfobject("Output:Meter".upper())
        meter1.Key_Name = yc_keys[i]
        meter1.Reporting_Frequency = output_frequency
        meter.append(meter1)
    idf.idfobjects['Output:Meter'.upper()]=meter

    idf.idfobjects['RUNPERIOD'][0].Begin_Month=1
    idf.idfobjects['RUNPERIOD'][0].Begin_Day_of_Month=1
    idf.idfobjects['RUNPERIOD'][0].End_Month=12
    idf.idfobjects['RUNPERIOD'][0].End_Day_of_Month=31

    idf.saveas('C:/Users/songc/Desktop/work file/Updated_Model.idf')
    idfname1 = 'C:/Users/songc/Desktop/work file/Updated_Model.idf'  # This IDF file is updated at each iteration.
    epwfile = './SPtMasterTable_52384_2011_amy.epw'
    subprocess.call(['C:/EnergyPlusV9-1-0/energyplus.exe', '-d', "C:/Users/songc/Desktop/work file/result_folder", '-w', epwfile, idfname1])

    eso_file='./result_folder/eplusout.eso'
    [ycoutput,xcoutput]=comp_data_reader(eso_file, yc_keys, xc_keys)

    yc_df1 = pd.DataFrame(ycoutput, index=yc_keys).T
    xc_df1 = pd.DataFrame(xcoutput, index=xc_keys).T   

    epwfile2 = './SPtMasterTable_52384_2012_amy.epw'
    subprocess.call(['C:/EnergyPlusV9-1-0/energyplus.exe', '-d', "C:/Users/songc/Desktop/work file/result_folder", '-w', epwfile2, idfname1])

    eso_file='./result_folder/eplusout.eso'
    [ycoutput,xcoutput]=comp_data_reader(eso_file, yc_keys, xc_keys)

    yc_df2 = pd.DataFrame(ycoutput, index=yc_keys).T
    xc_df2 = pd.DataFrame(xcoutput, index=xc_keys).T   

    yc_df = pd.concat([yc_df1,yc_df2],axis=0)
    xc_df = pd.concat([xc_df1,xc_df2],axis=0)

    df = pd.concat([yc_df,xc_df],axis=1)
    df.to_csv('DATAFIELD_Multi.csv',index=False)
    df_single = pd.concat([yc_df.iloc[:,0],xc_df],axis=1)
    df_single.to_csv('DATAFIELD_Single.csv',index=False)


def datacomp(yc_keys,xc_keys,tc_keys):
    iddfile = "./Energy+9.1.idd"
    IDF.setiddname(iddfile)

    yc_df = pd.DataFrame(columns=yc_keys)
    xc_df = pd.DataFrame(columns=xc_keys)
    tc_df = pd.DataFrame(columns=tc_keys)
    for n in range(len(LHS_result)):
        idfname = "./RefBldgLargeOfficeNew2004_Chicago.idf"
        idf = IDF(idfname)

        # change the output variable and meter
        # output_frequency='Daily'
        output_frequency='Monthly'

        variable=[]
        for i in range(len(xc_keys)):
            variable1 = idf.newidfobject("Output:Variable".upper())
            variable1.Key_Value = '*'
            variable1.Variable_Name = xc_keys[i]
            variable1.Reporting_Frequency = output_frequency
            variable.append(variable1)
        idf.idfobjects['Output:Variable'.upper()]=variable

        meter=[]
        for i in range(len(yc_keys)):
            meter1 = idf.newidfobject("Output:Meter".upper())
            meter1.Key_Name = yc_keys[i]
            meter1.Reporting_Frequency = output_frequency
            meter.append(meter1)
        idf.idfobjects['Output:Meter'.upper()]=meter

        # change the runperiod and other idf objects
        for i in range(len(idf.idfobjects['LIGHTS'])):
            idf.idfobjects['LIGHTS'][i].Watts_per_Zone_Floor_Area=LHS_result[n][0]

        for i in range(len(idf.idfobjects['ELECTRICEQUIPMENT'])):
            idf.idfobjects['ELECTRICEQUIPMENT'][i].Watts_per_Zone_Floor_Area=LHS_result[n][1]

        for i in range(len(idf.idfobjects['FAN:VARIABLEVOLUME'])):
            # idf.idfobjects['FAN:VARIABLEVOLUME'][i].Pressure_Rise=LHS_result[n][2]
            idf.idfobjects['FAN:VARIABLEVOLUME'][i].Fan_Total_Efficiency=LHS_result[n][2]

        for i in range(len(idf.idfobjects['ZONEINFILTRATION:DESIGNFLOWRATE'])):
            idf.idfobjects['ZONEINFILTRATION:DESIGNFLOWRATE'][i].Flow_per_Exterior_Surface_Area=LHS_result[n][3]

        for i in range(len(idf.idfobjects['CHILLER:ELECTRIC:REFORMULATEDEIR'])):
            idf.idfobjects['CHILLER:ELECTRIC:REFORMULATEDEIR'][i].Reference_COP=LHS_result[n][4]

        for i in range(len(idf.idfobjects['BOILER:HOTWATER'])):
            idf.idfobjects['BOILER:HOTWATER'][i].Nominal_Thermal_Efficiency=LHS_result[n][5]
        # for i in range(len(idf.idfobjects['SCHEDULE:COMPACT'])):
        #     if 'Building_Cooling_Sp_Schedule' in idf.idfobjects['SCHEDULE:COMPACT'][i].Name:
        #         idf.idfobjects['SCHEDULE:COMPACT'][i].Field_4=LHS_result[n][4]

        idf.idfobjects['RUNPERIOD'][0].Begin_Month=1
        idf.idfobjects['RUNPERIOD'][0].Begin_Day_of_Month=1
        idf.idfobjects['RUNPERIOD'][0].End_Month=12
        idf.idfobjects['RUNPERIOD'][0].End_Day_of_Month=31

        idf.saveas('C:/Users/songc/Desktop/work file/Updated_Model.idf')
        idfname1 = 'C:/Users/songc/Desktop/work file/Updated_Model.idf'  # This IDF file is updated at each iteration.
        epwfile = './SPtMasterTable_52384_2011_amy.epw'
        subprocess.call(['C:/EnergyPlusV9-1-0/energyplus.exe', '-d', "C:/Users/songc/Desktop/work file/result_folder", '-w', epwfile, idfname1])

        eso_file='./result_folder/eplusout.eso'
        [ycoutput,xcoutput]=comp_data_reader(eso_file, yc_keys, xc_keys)
        yc_df = yc_df.append(pd.DataFrame(ycoutput, index=yc_keys).T)
        xc_df = xc_df.append(pd.DataFrame(xcoutput, index=xc_keys).T)
        tc_df = tc_df.append(pd.DataFrame(np.reshape(list(LHS_result[n])*len(ycoutput[0]),(len(ycoutput[0]),len(tc_keys))),columns=tc_keys))

    
    df = pd.concat([yc_df,xc_df,tc_df],axis=1)
    df.to_csv('DATACOMP_Multi.csv',index=False)
    df_single = pd.concat([yc_df.iloc[:,0],xc_df,tc_df],axis=1)
    df_single.to_csv('DATACOMP_Single.csv',index=False)




bounds=[[10.76*0.8,10.76*1.2],[10.76*0.8,10.76*1.2],[0.605*0.8,0.605*1.2],[0.000302*0.8,0.000302*1.2],[5.5*0.8,5.5*1.2],[0.78*0.8,0.78*1.2]]
LHS_result=LHSample(6,bounds,30)

# bounds=[[10.76*0.8,10.76*1.2],[10.76*0.8,10.76*1.2],[0.605*0.8,0.605*1.2],[0.000302*0.8,0.000302*1.2]]
# LHS_result=LHSample(4,bounds,30)

# bounds=[[10.76*0.8,10.76*1.2],[10.76*0.8,10.76*1.2],[0.605*0.8,0.605*1.2]]
# LHS_result=LHSample(3,bounds,30)

# bounds=[[10.76*0.8,10.76*1.2],[10.76*0.8,10.76*1.2]]
# LHS_result=LHSample(2,bounds,30)

# yc_keys=['Electricity:Facility','InteriorLights:Electricity','Fans:Electricity','InteriorEquipment:Electricity']
yc_keys=['Electricity:Facility','InteriorLights:Electricity','InteriorEquipment:Electricity','Electricity:HVAC','Heating:Gas']
# yc_keys=['Electricity:Facility','InteriorEquipment:Electricity']
# yc_keys=['Electricity:Facility','Cooling:Electricity']
xc_keys=['Site Outdoor Air Drybulb Temperature','Site Outdoor Air Relative Humidity','Site Direct Solar Radiation Rate per Area']
tc_keys=['tc1','tc2','tc3','tc4','tc5','tc6']

datafield(yc_keys, xc_keys)
datacomp(yc_keys, xc_keys, tc_keys)