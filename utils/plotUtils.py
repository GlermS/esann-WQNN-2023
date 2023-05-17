import plotly.offline
import plotly.graph_objects as go
import utils.dataUtils as du

def plot3d(listOfData, fileName):

    fig = go.Figure()
    for (vectors,legend, mode) in listOfData:

        fig.add_trace(go.Scatter3d(x=vectors[0], y=vectors[1], z=vectors[2],hovertext=vectors[3],mode=mode,name=legend))

    plotly.offline.plot(fig, filename=fileName,auto_open=False) 

def plot(listOfData, fileName=None):

    fig = go.Figure()
    for (vectors,legend, mode) in listOfData:

        fig.add_trace(go.Scatter(x=vectors[0], y=vectors[1],hovertext=vectors[2],mode=mode,name=legend))
    if fileName is not None:
        plotly.offline.plot(fig, filename=fileName,auto_open=False) 
    else: 
        return fig

def plotDf(df, fileName, mode='lines', legend={}, z=None, numOfDecimals=2):

    fig = go.Figure()
    
    if z is None:
        z = du.stringConcatenateWithLabels(du.formatDecimals(df.values,numOfDecimals), df.columns)

    for col in [c for c in df.columns if c != 'timestamp' and c!= 'ts' and c != 'ds']:
        name = col
        l = legend.get(col)
        if l is not None:
            name += ': ' + l

        x = None

        if 'timestamp' in df.columns:
            x = df['timestamp']
        elif 'ts' in df.columns:
            x = df['ts']
        elif 'ds' in df.columns:
            x = df['ds']            
        else:
            x = df.index

        fig.add_trace(go.Scatter(x=x, y=df[col],hovertext=z,mode=mode,name=name))

    plotly.offline.plot(fig, filename=fileName,auto_open=False) 