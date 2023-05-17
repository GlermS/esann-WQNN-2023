import math
import numpy as np
import pandas as pd
import pvlib
import datetime

from math import acos, cos, sin, radians, degrees

def getSunDf(plantInfo,startDate,endDate,freq='5min'):

    latitude = plantInfo['latitude']
    longitude = plantInfo['longitude']
    altitude = plantInfo['altitude']
    azimuth = plantInfo['azimuth']
    tz = plantInfo['tz']

    loc = pvlib.location.Location(latitude, longitude, tz, altitude, 'x')

    times = pd.date_range(start=startDate, end=endDate, freq=freq, tz=loc.tz)

    sun = loc.get_solarposition(times)
    
    df = sun
    
    df['azimuth'] = [ a if a < 180 else (a -360) for a in df['azimuth']]

    df['zenith'] = -df['zenith']
    
    df.index = df.index.tz_localize(None)
    df.index.name = 'ts'
    
    
    return df


def getClearSkyModelDf(plantInfo,startDate,endDate,freq='5min', sun=None, gcr=None,albedo=None):

    latitude = plantInfo['latitude']
    longitude = plantInfo['longitude']
    altitude = plantInfo['altitude']
    azimuth = plantInfo['azimuth']
    gcr = plantInfo['gcr']
    tz = plantInfo['tz']

    loc = pvlib.location.Location(latitude, longitude, tz, altitude, 'x')

    times = pd.date_range(start=startDate, end=endDate, freq='5min', tz=None)

    if sun is None:
        sun = getSunDf(plantInfo,startDate,endDate,freq=freq)
        
    sun = sun[~sun.index.duplicated()]
    
    cs = loc.get_clearsky(times, solar_position = sun)

    df = sun.join(cs)

    df = df.rename(columns={'ghi': 'mghi', 'dni': 'mdni', 'dhi': 'mdhi'})

    dft = getTrackingModel(plantInfo,startDate,endDate,gcr=gcr,sun=sun,freq='5min')

    df = df.join(dft)
    
    if albedo is None:
        albedo = plantInfo['albedo']

    # print ('albedo:', albedo)

    dfg = pvlib.irradiance.get_total_irradiance(
        surface_tilt=-df['mangle'],
        surface_azimuth=azimuth,
        dni=df['mdni'],
        ghi=df['mghi'],
        dhi=df['mdhi'],
        solar_zenith=-df['zenith'],
        solar_azimuth=df['azimuth'], albedo=albedo)

    df = df.join(dfg)

    df = df.rename(columns={'poa_global': 'mg'})

    return df

    
def getClearSkyDaysDf(dfw, gstd=100, gmean=500):
    
    df = dfw.copy()

    df = df[['g']]

    index = (df.index.hour>=10) & (df.index.hour<=15)
    df = df[index]

    dfh = df

    df = df.resample('1d').std()

    df = df.rename(columns={'g': 'gstd'})

    df1 = df

    df = dfh

    df = df.resample('1d').mean()

    df['gstd'] = df1['gstd']

    index = (df['gstd'] < gstd) & (df['g']>gmean)

    df = df[index]

    bestDates = df.index
    df = pd.DataFrame({'ts': pd.to_datetime([str(d) for d in bestDates])})
    df['ts'] = pd.to_datetime(df['ts']).dt.date
    bestDates = df['ts']
    
    
    df = dfw.copy()
    df['ts'] = [str(d) for d in df.index]
    df['dt'] = pd.to_datetime(df['ts']).dt.date

    df['clearsky'] = 0

    df.loc[df['dt'].isin(bestDates), 'clearsky'] = 1

    df = df.drop(['dt','ts'],axis=1)

    return df


def gtilt(panelAzimuth,sunAzimuth,sunZenith):

        
    a = radians(panelAzimuth)
    
    sz = radians(sunZenith)
    sa = radians(sunAzimuth)
    
    # sun = (sin(sz)*cos(sa), sin(sz)*sin(sa), cos(sz) )
    # panel = (sin(z)*cos(a), sin(z)*sin(a), cos(z)) 

    # p = sinz*c + cosz*cos(sz) # projection = sinz*sin(sz)*cos(a-sa) + cosz*(cos sz)

    c =  cos(a)*sin(sz)*cos(sa) + sin(a)*sin(sz)*sin(sa) # = sin(sz)*cos(a - sa)

    # dp/dz = c*cosz - sinz*cos(sz) 
    # se dp/dz = 0, c*cosz = sinz*cos(sz) -> (c*cosz)**2 = (1 - (cosz)**2)*(cos(sz))**2
    # -> (cosz)**2(c**2 + cos(sz)**2)=cos(sz)**2 -> cosz = cos(sz)/sqrt(c**2 + cos(sz)**2)
    
    z = acos(cos(sz)*math.sqrt(1/(c**2 + cos(sz)**2) ))    

    dzplus = c*cos(z) - cos(sz)*sin(z)
    dzminus = c*cos(-z) - cos(sz)*sin(-z)

    if abs(dzplus) < abs(dzminus):
        z = degrees(z)
    else:
        z = degrees(-z)

    return z

def gtilt1(panelAzimuth, sunAzimuth, sunZenith, dni, dhi):
    
    maxG = 0
    maxT  = 0

    for t in range(-90,90):
        beam = pvlib.irradiance.beam_component(t, panelAzimuth, sunZenith, sunAzimuth, dni)
        cs_sky = 0#pvlib.irradiance.isotropic(t, dhi)

        g = beam + cs_sky
        
        if g > maxG:
            maxG = g
            maxT = t
            
    z = maxT

    return z


def getTrackingModel(plantInfo, startDate, endDate, freq='5min', gcr=None,sun=None):

    info = plantInfo

    azimuth = info['azimuth']
    
    if gcr is None:
        gcr = info['gcr']
    tz = info['tz']

    if sun is None:
        sun = getSunDf(plantInfo, startDate, endDate, freq=freq)

    df = pvlib.tracking.singleaxis(
    apparent_zenith=sun['apparent_zenith'],
    apparent_azimuth=sun['azimuth'],
    axis_tilt=0,
    axis_azimuth=azimuth+90,
    max_angle=60,
    backtrack=True,
    gcr=gcr)

    df = df.fillna(0)
    df = df.rename(columns={'tracker_theta': 'mangle'})

    df['ts'] = [t for t in df.index]
    df['ts'] = df['ts'].dt.tz_localize(None)
    df = df.set_index(pd.DatetimeIndex(df['ts']))

    return df[['mangle']]


def estimatePoaFromGhi(plantInfo, df, freq='5min',albedo=None):

    plantAzimuth = plantInfo['azimuth']
    
    startDate = str(df.index[0])
    endDate = str(df.index[-1])

    cs = getClearSkyModelDf(plantInfo,startDate,endDate,freq,albedo)

    df = df.join(cs)

    if not 'angle' in df.columns:
        df['angle'] = df['mangle']

    df['p'] = [0 for i in range(len(df))]
    df['poa'] = [0 for i in range(len(df))]
    df['n'] = [0 for i in range(len(df))]

    for i in range(len(df)):
        a = radians(plantAzimuth)
        sa = radians(df['azimuth'].iloc[i])
        sz = radians(df['zenith'].iloc[i])
        z = radians(df['angle'].iloc[i])

        c =  cos(a)*sin(sz)*cos(sa) + sin(a)*sin(sz)*sin(sa)
        p = max(0,sin(z)*c + cos(z)*cos(sz))

        df['p'].iloc[i] = p
        
        ghi = df['ghi'].iloc[i]
        dhi = df['mdhi'].iloc[i]
        n = np.nan

        if not np.isnan(ghi):
            n = min(1200,(ghi)/cos(sz))
        
        df['n'].iloc[i] = n

    df['poa'] = df['p']*df['n']
    
    return df

def getSolarProjection(plantInfo, df, startDate,endDate,freq='5min'):
    
    plantAzimuth = plantInfo['azimuth']
    
    startDate = str(df.index[0])
    endDate = str(df.index[-1])

    cs = getClearSkyModelDf(plantInfo,startDate,endDate,freq)

    df = df.join(cs)

    df['p'] = [0 for i in range(len(df))]

    for i in range(len(df)):
        a = radians(plantAzimuth)
        sa = radians(df['azimuth'].iloc[i])
        sz = radians(df['zenith'].iloc[i])
        z = radians(df['angle'].iloc[i])

        c =  cos(a)*sin(sz)*cos(sa) + sin(a)*sin(sz)*sin(sa)
        p = max(0,sin(z)*c + cos(z)*cos(sz))

        df['p'].iloc[i] = p
    
    return df[['p']]

def estimatePoaFromGhi2(plantInfo, df, freq='5min',albedo=None):
    
    plantAzimuth = plantInfo['azimuth']
    
    startDate = str(df.index[0])
    endDate = str(df.index[-1])

    cs = getClearSkyModelDf(plantInfo,startDate,endDate,freq,albedo)
    #dft = getModelTrackingForPlant(plantName,startDate,endDate,freq)

    df = df.join(cs)#.join(dft)

    df['p'] = [0 for i in range(len(df))]
    df['poa'] = [0 for i in range(len(df))]
    df['n'] = [0 for i in range(len(df))]

    for i in range(len(df)):
        a = radians(plantAzimuth)
        sa = radians(df['azimuth'].iloc[i])
        sz = radians(df['zenith'].iloc[i])
        z = radians(df['angle'].iloc[i])

        c =  cos(a)*sin(sz)*cos(sa) + sin(a)*sin(sz)*sin(sa)
        p = max(0,sin(z)*c + cos(z)*cos(sz))

        df['p'].iloc[i] = p
        
        ghi = df['ghi'].iloc[i]
        dhi = df['mdhi'].iloc[i]
        n = np.nan

        if not np.isnan(ghi):
            n = min(1200,(ghi)/cos(sz))
        
        df['n'].iloc[i] = n

    df['poa'] = df['p']*df['n']

    dfg = pvlib.irradiance.get_total_irradiance(
    surface_tilt=-df['angle'],
    surface_azimuth=plantAzimuth,
    dni=df['n'],
    ghi=df['ghi'],
    dhi=df['mdhi'],
    solar_zenith=-df['zenith'],
    solar_azimuth=df['azimuth'], albedo=albedo)
    
    return df[['g','ghi', 'poa', 'angle', 'zenith', 'azimuth', 'p']].join(dfg)

def gcoef(data, coef, powerLimit, alpha=0.004):
    
    irr = np.transpose(np.transpose(data['g']))
    pwr = data['P']
    pt = data['panT']
    
                           
    metricsAndCoefs = []
    
    ci = 0.65*coef
    cf = 1.25*coef
    step = 0.005*coef
    
    
    n = int((cf-ci)/step)
    
    coefs = [ci + i*step for i in range(n)]
    
    metrics = [0 for i in range(n)]
    
    for j in range(len(irr)):
        
        df = pd.DataFrame()
        
        df['pwr'] = pwr
        df['irr'] = irr[j]
        df['pt'] = pt
        
        df = df[ (df['pwr']>1) & (df['irr'] > 75)]
        
        y = df['pwr'].values
        
        for i in range(n):
            c = coefs[i]
            
            y_ = c*df['irr']*(1+alpha*(25-df['pt']))
            y_ = y_.values
            y_ = [p if p<=powerLimit else powerLimit for p in y_]

            errors = [abs(y[k]-y_[k])/y[k] if y[k] != 0 else abs(y[k] - y_[k]) for k in range(len(y))]
            
            e_ = 0.01
            
            m = sum([(1-e) if e <= e_ else 0 for e in errors])
            
            metrics[i] += m
    
    metricsAndCoefs = [(metrics[i], coefs[i]) for i in range(n)]
    import heapq
    
#     print(heapq.nlargest(5, metricsAndCoefs))
    return heapq.nlargest(1, metricsAndCoefs)[0][1]

def gfusion(df,gcols,params=None,f=None,p=0.05):

    data = {'P': df['P'].values, 'g': [], 'panT': df['panT'].values}

    for c in gcols:
        data['g'] += [df[c].values]
    
    if f is None:
        def f(g,panT,params):

            return min(params['k']*g*(1+params['a']*(25-panT)),params['limit'])
    
    irr = np.transpose(data['g'])
    pwr = data['P']
    pt = data['panT']
    
    estimatedIrradiance = np.array([0.0 for i in range(len(pwr)) ])
    for i in range(len(pwr)):
        
        validIrrValues = [v for v in irr[i] if not np.isnan(v) and v >=5]
        
        if not np.isnan(pwr[i]) and pwr[i] > 0 and len(validIrrValues) > 0:

            g = gcentroids(np.sort(validIrrValues),p=p)
            
            pwr_ = [f(v,pt[i],params) for v in g]

            errors = [ abs(pwr[i] - pwr_[j]) for j in range(len(g)) ]
            
            index = np.argmin(errors)
            estimatedIrradiance[i] = g[index]
            

        elif len(validIrrValues) == 0:
            estimatedIrradiance[i] = np.nan
        else:
            try:
                estimatedIrradiance[i] = np.median(validIrrValues)
            except:
                estimatedIrradiance[i] = np.nan
 
            
    return estimatedIrradiance
print ('ok')

def gfilter(data, coef, powerLimit, k=5, e0=0.1,alpha=0.004):
    
    c = coef
    
    running = False
    begin=0
    end=0
    
    df = pd.DataFrame(data)
    
    pwr = df['power'].values
    irr = df['irradiance'].values
    pt = df['panelTemp'].values

    
    filt = [0 for i in range(len(irr))]
    
    y = pwr
    y_ = c*irr*(1+alpha*(25-pt))
    
    sign = 0
    oldSign = 0
    
    pThreshold = 50
    for i in range(len(irr)):
        if pwr[i] < pThreshold:
            running = False
        else:
            e = (y[i] - y_[i])/y[i]

            if e > 0:
                sign = 1
            else:
                sign = -1

            if not running:
                if abs(e) >= e0:
                    running = True
                    begin = i
            else:
                if sign != oldSign or abs(e) < 0.5*e0:
                    running = False
                    end = i

                    runLength = end - begin + 1

                    if runLength < k:
                        filt[begin:(end+1)] = [1 for i in range(begin,(end+1))]

            oldSign = sign
                    
    newIrradiance = [irr[i] if filt[i] == 0 else pwr[i]/(c*(1+alpha*(25-pt[i]))) for i in range(len(irr))]
    
        
    return newIrradiance 

def gcentroids(values,p=0.05):
    
    prev = [0 for v in values]
    
    for i in range(1,len(values)):
        j = prev[i-1]
        
        while values[i] > (values[j]*(1+p)):
            j += 1
            if j > i:
                print ('i:', i, 'j:',j)
                print ('v[i]:', values[i], 'v[j]:', values[j])
            
        prev[i] = j
    
    centroids = [[] for i in range(len(values))]
    centroids[0] = [0]
    
    for i in range(1,len(values)):
        if values[i] <= ( values[centroids[i-1][-1]]*(1+p) ):
            centroids[i] = centroids[i-1]
        else:
            i_ = prev[prev[i]] - 1
            
            centroids[i] = centroids[i_] + [prev[i]]
    
    #   return [values[i] for i in centroids[-1]]
    lastI = -1
    
    
    cm = []
       
    for c in centroids[-1]:
        i = c+1
        
        a = [v for v in values[lastI+1:c+1]]
        
        while i < len(values) and values[c]*(1+p) >= values[i]:
            a += [values[i]]
            i+=1
        
        lastI = i-1
        cm += [np.median(a)]
        # print (c, a, cm)
    
    return cm

            
        
            
        