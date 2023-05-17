import pandas as pd
import numpy as np
import datetime
import re
import math
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
class LoadDfs:
    def __init__(self, path = '../data/') -> None:
        self.path = path
        pass
    
    def __iter__(self):
        self.files = os.listdir(self.path)
        self.i = 0
        return self
    def __next__(self):
        if self.i >= len(self.files):
            raise StopIteration
            
        file = self.files[self.i]
        df = pd.read_csv(os.path.join(self.path, file), index_col='ts', parse_dates=['ts'])
        self.i+= 1            
        return file, df


def makeDfFromData(data):
    remap = []
    for dataSourceId in data:
        for value in data[dataSourceId]:
            remap.append({'ts': datetime.datetime.fromtimestamp(value[0]), dataSourceId: value[1]})
    df = pd.DataFrame(remap, dtype=float)
    
    if len(df)>=1:
        return df.set_index('ts')
    else:
        print('No data')
        return None



def filterValues(df,cols,qb=0.25,qt=0.75, func=np.nanmedian):
    
    n = len(cols)

    filteredValues = []

    for i in range(len(df)):
        s = df.iloc[i][cols].sum()

        validValuesForThisRow = []
        for c in cols:
            v = df.iloc[i][c]
            if np.isnan(v) or v <= 0:
                pass
            else:
                validValuesForThisRow += [v]

        try:
            b = np.quantile(validValuesForThisRow,qb)
            t = np.quantile(validValuesForThisRow,qt)

            finalValuesForThisRow = []
            for v in validValuesForThisRow:
                if v < qb or v > qt:
                    pass
                else:
                    finalValuesForThisRow += [v]
                        
            fv = func(validValuesForThisRow)
        except:
            fv = np.nan

        filteredValues += [fv]
        
    return filteredValues

def csvTextToDf(csvText):
    data = csvText.splitlines()

    header = data[0].split(',')

    header[0] = 'timestamp'

    dic = {}

    for h in header:
        dic[h] = []

    for i in range(1,len(data)):
        values = data[i].split(',')
        for j in range(len(header)):
            dic[header[j]] += [values[j]]

    df = pd.DataFrame(dic)

    return df


def applyFirstOrderFilterToData(data,k,threshold):
    
    if isinstance(data, (list,np.ndarray)) and len(data) > 0:
            if isinstance(data[0],list):
                return [firstOrderFilter(d,k,threshold) for d in data]
            else:
                # print ('bla!')
                return firstOrderFilter(data,k,threshold)

    elif isinstance(data, dict):
        newData = {}

        for key,v in data.items():
            newData[key] = applyFirstOrderFilterToData(v,k,threshold)
        return newData

    else:
        return data

def firstOrderFilter(x, k, threshold):
    
    if isinstance(x,(list,np.ndarray)) and len(x) > 0 and isinstance(x[0],(np.int32, np.float, float,int)):
        y = [0 for v in x]
        y[0] = x[0]

        for i in range(1, len(x)):
            if x[i] > threshold and y[i-1] > threshold:
                y[i] = y[i-1] + float(x[i] - y[i-1])/k
            else:
                y[i] = x[i]
        
        return y
    else:
        return x

def foFilter(x, k):
    
    if isinstance(x,(list,np.ndarray, pd.Series)) and len(x) > 0 and isinstance(x[0],(np.int32, np.float,float,int)):
        y = [0 for v in x]
        y[0] = x[0]

        for i in range(1, len(x)):
                y[i] = y[i-1] + float(x[i] - y[i-1])/k
        
        return y
    else:
        return x

def formatDecimals(data,numberOfDecimals=2):
    
    if isinstance(data,(int,float)):
        return '{:.{}f}'.format(data, numberOfDecimals)
    else:
        if isinstance(data,dict):
            newData = {}
            for k,v in data.items():
                newData[k] = formatDecimals(v, numberOfDecimals)
            
            return newData
        
        elif isinstance(data,(list,np.ndarray)):
            return [formatDecimals(x, numberOfDecimals) for x in data]
        else:
            return str(data)

def toHtml(data,fileName=None):
    import json2html

    html = json2html.json2html.convert(json = data, table_attributes="id=\"info-table\" class=\"table table-bordered\"")
    html = '<head> <link href="https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/4.0.0-beta/css/bootstrap.min.css" rel="stylesheet" /></head>' + html

    if fileName is not None:
        file = open(fileName, 'w')
        
        file.write(html)
        file.close()

    return html

def rename(names,nameMap):
    newNames = []
    for name in names:
        found = False
        for regex,newName in nameMap.items():
            if re.search(regex,name) is not None:
                newNames += [newName]
                found = True
                break
        if not found:
            newNames += [name]
    return newNames

def remap(data, fieldMap):
    
    newData = {}
    for k,v in data.items():
        
        found = False
        
        for regex, newFieldName in fieldMap.items():    
            if re.search(regex, k) is not None:
                
                if newData.get(newFieldName) is None:
                    newData[newFieldName] = v
                    
                elif isinstance(newData[newFieldName], list):
                    if isinstance(newData.get(newFieldName)[0], list):
                        newData[newFieldName] += [v]
                    else:
                        newData[newFieldName] = [newData[newFieldName]] + [v]
                
                found = True
                break
                
        if not found:
            newData[k] = v
    
    return newData
            
def getMinListSize(data):

    if isinstance(data, (list,np.ndarray)):
        if len(data) == 0:
            return 0
        elif not isinstance(data[0], (list,np.ndarray)):
            return len(data)
        else:
            return min([ getMinListSize(x) for x in data] )
    elif isinstance(data, dict):
        return min( [getMinListSize(v) for v in list(data.values())])
            
    else:
        return float('inf')


def adjustListSizes(data, size):
    
    if isinstance(data,(list,np.ndarray)):
        if len(data) == 0:
            return data
        elif isinstance(data[0], (list,np.ndarray)):
            return [adjustListSizes(x, size) for x in data]
        else:
            return data[:size]
        
    elif isinstance(data,dict):
        newData = {}
        
        for k,v in data.items():
            newData[k] = adjustListSizes(v, size)
        return newData
    else:
        return data

def renameFields(d, fieldMap):
    newD = {}
    for k,v in d.items():
        if fieldMap.get(k) is not None:
            newD[fieldMap[k]] = v
        else:
            newD[k] = v
        
    return newD

print ('ok')

def getField(obj, fieldNameRegex):
    data = None

    if fieldNameRegex is None:
        return None
    
    if isinstance(obj, dict):
        for k,v in obj.items():

            if k == fieldNameRegex or re.search(fieldNameRegex,k) is not None:
                data = v
                
    elif isinstance(obj, list):
        data = []
        for element in obj:
            data += [getField(element,fieldName)]
                
    return data

def get(obj,fieldNameRegex):
    return getField(obj,fieldNameRegex)

def makeDf(data, startDate, endDate, samplingInterval=5):
        
        # print ('data:', data)

        startDt = datetime.datetime.fromisoformat(startDate)
        endDt = datetime.datetime.fromisoformat(endDate)

        startDate = str(startDt.replace(minute=samplingInterval*math.floor(startDt.minute/samplingInterval)))[:16]
        endDate = str(endDt.replace(minute=samplingInterval*math.floor(endDt.minute/samplingInterval)))[:16]

        times = pd.date_range(start=startDate, end=endDate, freq=str(samplingInterval)+'min')
        df = pd.DataFrame({'ts': [str(t) for t in times]})
        df = df.set_index(pd.DatetimeIndex(pd.to_datetime(df['ts'])))
        df = df.drop(['ts'],axis=1)
        df0 = df.copy()
        dfTotal = df

        for key,dic in data.items():

            for k,d in dic.items():
                # print (d.keys())
                df = pd.DataFrame(d)
                df = df.set_index(pd.DatetimeIndex(pd.to_datetime(df['ts'])))
                df = df[~df.index.duplicated(keep='first')]

                df = df0.join(df,how='left')

                dfTotal[key + ' - ' + k] = df['values'].values

                del df

        return dfTotal

def selectFromJson(obj, filterMap):
    data = []
    
    if isinstance(obj, dict):
        for k,v in filterMap.items():
            if obj.get(k) is not None and re.search(v, obj.get(k)) is not None:
                data += [obj]

        for v in obj.values():
            if isinstance(v, dict) or isinstance(v, list):
                data += selectFromJson(v,filterMap )
                
    elif isinstance(obj, list):
        for element in obj:
            data += selectFromJson(element,filterMap)
                
    return data

def stringConcatenateWithLabels(tuples, labels,prefix='', numberOfDecimals=2):
    
    s = []
    for tup in tuples:
        si = prefix 
        
        for j in range(len(tup)):
            si += labels[j] + ': ' + formatDecimals(tup[j],numberOfDecimals) + ' '
            
        s += [si]
        
    return s


def writeToXlsx(data,fileName):

    import xlsxwriter

    wb = xlsxwriter.Workbook(fileName + '.xlsx')
    ws = wb.add_worksheet()

    ws.write(0,0, 'timestamp')
    
    timestamps = data['timestamps']
    
    i = 1
    for timestamp in timestamps:
        ws.write(i,0,timestamp)
        i+=1

    columns = data['columns']
    
    j=1
    for col in columns:
        ws.write(0,j,col['name'])
        i=1
        for val in col['values']:
            ws.write(i,j, val)
            i+=1
        j+=1

    wb.close()

def splitEveryNumberOfDays(index, vectors, legend, numberOfDays, graphType='markers'):
    tuples = []
    
    firstDatetime = index[0]
    lastDatetime = index[-1]
    
    startDatetime = firstDatetime
    
    while startDatetime < lastDatetime:
        
        endDatetime = startDatetime + datetime.timedelta(days=numberOfDays)
        
        indices = (index > startDatetime) & (index < endDatetime )


        idx = [i for i in list(range(len(indices))) if indices[i] == True]

        vectorsForThisPeriod = []

        for vector in vectors:
            v = [vector[i] for i in idx]
            vectorsForThisPeriod.append(v)

        l = legend + ' in ('+ str(startDatetime) + ',' + str(endDatetime) +')' 

        tup = (vectorsForThisPeriod, l, graphType)

        tuples += [tup]
        
        startDatetime = endDatetime
    
    return tuples

# def getUniformDf(df, column, binWidth):
    
#     import math

#     x = df[column].values

#     numOfBins = math.ceil((max(x) - min(x))/binWidth) + 1
# #     print (numOfBins)
#     bins = [[] for i in range(numOfBins)]

#     minX = min(x)
#     for v in x:
#         binNumber = math.ceil((v - minX)/binWidth)

#         bins[binNumber] += [v]
    
#     lens = [len(b) for b in bins]
#     quantile = np.quantile(lens,0.2)
#     print ('quantile:', quantile)
    
#     bins = [b for b in bins if len(b) > 0]
    
#     uniformBinSize = du.getMinListSize(bins)
#     print (uniformBinSize)
#     if uniformBinSize == 0:
#         print ('please choose wider bins')
#         return
    

#     uniformValues = []
#     print('bins[-1]:', bins[-1])
#     size = len(bins[-1])
#     uniformBins = du.adjustListSizes(bins,size)
# #     uniformBins = du.adjustListSizes(bins,uniformBinSize)
# #     uniformBins = [du.adjustListSizes(bins[i],int(size/5)) if i < (numOfBins -1) else du.adjustListSizes(bins[i],size) for i in range(len(bins))]
    
# #     print (uniformBins)
# #     print ('len(uniformBins):', len(uniformBins))

#     mustIncludeValue = {}

#     for b in uniformBins:
#         for v in b:
#             uniformValues += v
#             mustIncludeValue[v] = True

#     index = [True if mustIncludeValue.get(v) is not None else False for v in x] 

#     return df[index]
# print ('ok')