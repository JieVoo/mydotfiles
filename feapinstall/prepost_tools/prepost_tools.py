
from mpl_toolkits.axes_grid1 import make_axes_locatable
from paraview.simple import *
from matplotlib import cm

import matplotlib.colorbar as colorbar
import xml.etree.ElementTree as ET
import matplotlib.ticker as ticker
import matplotlib.colors as cols
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import itertools as itt
import distutils.spawn
import shutil as shu
import numpy as np
import subprocess
import smtplib
import inspect
import string
import errno
import glob
import math
import time
import csv
import sys
import os
import re

try:
    import cPickle as pickle
except:
    import pickle

cmap = cm.get_cmap('jet',257)
inds = np.random.permutation(257).astype(float)/256.

r=cmap(inds)[:,0]
g=cmap(inds)[:,1]
b=cmap(inds)[:,2]

rtpl = []
gtpl = []
btpl = []
for i in range(256):
    rtpl.append( (float(i)/255, r[i], r[i+1]) )
    gtpl.append( (float(i)/255, g[i], g[i+1]) )
    btpl.append( (float(i)/255, b[i], b[i+1]) )

cdict = {'red':rtpl, 'green':gtpl, 'blue':btpl}

cmap_luc=cols.LinearSegmentedColormap('luc',cdict,N=256)

#class FilterInfo{{{1
class FilterInfo(object):
    def __init__(self,Name,Args):
        self.Name = Name
        self.Args = Args
#1}}}
#def GetFilterDict{{{1
def GetFilterDict(Filter):
    psot_dict  = {'Function':psot,
                  'FileType':'csv'}

    iv_dict    = {'Function':iv,
                  'FileType':'csv'}

    wbv_dict   = {'Function':wbv,
                  'FileType':'vtu'}

    clip_dict  = {'Function':clip,
                  'FileType':'vtu'}

    vf_dict    = {'Function':vf,
                  'FileType':'png'}

    fildict = {'psot'         :psot_dict,
               'iv'           :iv_dict,
               'wbv'          :wbv_dict,
               'clip'         :clip_dict,
               'vf'           :vf_dict}

    return fildict[Filter]
#1}}}
#def PPTSave{{{1
def PPTSave(PipeLine,FileOut,FileType):

    if FileType.lower() == 'csv':
        #Write to csv
        SetActiveSource(PipeLine)
        CSVF = CSVWriter()
        CSVF.FileName = FileOut
        CSVF.UpdatePipeline()

        #rename csv back to original name since paraview appends a 0
        spl = FileOut.split('.')
        FileOut0 = '.'.join([spl[0]+"0",spl[1]])
        os.rename(FileOut0,FileOut)
    elif FileType.lower() == 'png':
        pass
    else:
        print "No method implemented to save data type",FileType
#1}}}
#def PPTLoad{{{1
def PPTLoad(FileOut,FileType):

    if FileType.lower() == 'csv':
        #load csv
        return pvcsvin(FileOut)
    elif FileType.lower() == 'png':
        return None
    else:
        print "No method implemented to load data type",FileType
#1}}}
#def ApplyFilters{{{1
def ApplyFilters(PVDin,FileOut,FilterList,ViewTime="Last"):
    #Open pvd
    PipeLine = PVDReader( FileName=PVDin )
    PipeLine.UpdatePipeline()
    LastName = PVDin
    
    
    #set view time
    if isinstance(ViewTime,basestring):
        if ViewTime.lower() == "last":
            VT = PipeLine.TimestepValues[-1]
        if ViewTime.lower() == "first":
            VT = PipeLine.TimestepValues[0]
    elif isinstance(ViewTime,int):
        print('ViewTime is step {:d}'.format(ViewTime))
        VT = PipeLine.TimestepValues[ViewTime]
    elif isinstance(ViewTime,(float,np.int64)):
        VT = ViewTime
    else:
        print("Error, ViewTime must be a string (first or last), integer index, or float time value")
        return


    #Apply filters
    for Filter in FilterList:
        #Get info for current filter
        FilterName = Filter.Name
        FilterArgs = Filter.Args
        FilterDict     = GetFilterDict(FilterName)
        FilterFunction = FilterDict['Function']
        FileType       = FilterDict['FileType']

        if FileType.lower() == 'png':
            FilterArgs["OutputFile"] = FileOut
            FilterArgs["ViewTime"]   = VT

        #Apply
        PipeLine = FilterFunction(PipeLine,**FilterArgs)
        print "Applied ",FilterName," to ",LastName
        LastName = FilterName

    #Save data with method based off of the last filter applied
    PPTSave(PipeLine,FileOut,FileType)

    #Clear pipeline
    for f in GetSources().values():
        Delete(f)


    return PPTLoad(FileOut,FileType)
#1}}}
#def iv{{{1
def iv(PipeLine,AutoScale=True,Scale=None):
    '''
    This function applies the paraview integrate variable filter which returns the volume integral of the field values multiplied by the volume of the domain. This volume is divided out with the AutoScale flag
    Arguments:
    
    - *PipeLine* the paraview pipeline
    - *AutoScale* flag to automatically divide out the volume in the integrate variables filter
    '''

    SetActiveSource(PipeLine)
    IV = IntegrateVariables()
    IV.UpdatePipeline()

    if AutoScale:
        SetActiveSource(IV)
        script = "inp = inputs[0]\n" + \
                 "#Determine length, area, or volume\n" + \
                 "if " + str(Scale) +" is None:\n" +\
                 "    lav = ['Length','Area','Volume']\n" + \
                 "    if not inp.GetCellData() is None:\n" + \
                 "        if inp.GetCellData().GetNumberOfArrays() > 0:\n" + \
                 "            for CellArrayIdx in range(0,inp.GetCellData().GetNumberOfArrays()):\n" + \
                 "                data = inp.GetCellData().GetArray(CellArrayIdx)\n" + \
                 "                name = inp.GetCellData().GetArrayName(CellArrayIdx)\n" + \
                 "                if name in lav:\n" + \
                 "                    LAV = data.GetTuple(0)[0]\n" + \
                 "else:\n" +\
                 "    LAV="+str(Scale)+"\n"+\
                 "#Scale point data by LAV\n" + \
                 "npa = inp.GetPointData().GetNumberOfArrays()\n" + \
                 "nams= [0]*npa\n" + \
                 "if not inp.GetPointData() is None:\n" + \
                 "    if inp.GetPointData().GetNumberOfArrays() > 0:\n" + \
                 "        for PointArrayIdx in range(0,inp.GetPointData().GetNumberOfArrays()):\n" + \
                 "            name = inp.GetPointData().GetArrayName(PointArrayIdx)\n" + \
                 "            nams[PointArrayIdx] = name\n" + \
                 "for n in nams:\n" + \
                 "    data = inp.PointData[n]\n" + \
                 "    data = data/LAV\n" + \
                 "    output.PointData.append(data,n)"

        IV = ProgrammableFilter()
        IV.Script = script
        IV.UpdatePipeline()

    return IV
    
#1}}}
#def psot{{{1
def psot(PipeLine,NodNum=0,Cell=False,StatsOnly=False,Frustum=None,ExtractMode="POINT"):
    '''
    This function applies the paraview plot selection over time filter and optionally returns the result using pvcsvin.
    Arguments:
    
    - *PipeLine* the paraview pipeline
    - *NodNum* the number of the node to which the filter is applied
    - *Cell* flag plot cell data
    - *StatsOnly* flag to only compute statstics of the fields
    - *Frustum* Frustum defining selection
    - *ExtractMode* Indicates if frustum extracts cells or points

    .. todo::
        extracting selection is not very elegant, it could be much better if there was a way to convert a frustum selection source to an id selection source
    '''

    SetActiveSource(PipeLine)
    #Extract selection if prompted{{{2
    if Frustum is not None:
        Frustum = getFrustumFromBounds(Frustum)
        selectionSourceF = FrustumSelectionSource(ContainingCells=0,InsideOut=0,FieldType="CELL",Frustum=Frustum)
        #Create frustum selection and apply extract selection filter
        Dat = ExtractSelection()
        Dat.Selection = selectionSourceF
        Dat.UpdatePipeline()

        if ExtractMode == "POINT":
            numPointsSelected = Dat.GetDataInformation().DataInformation.GetNumberOfPoints()
            IDs = []
            for idx in range(numPointsSelected):
                IDs += [0L, idx]

            selectionSource = IDSelectionSource( ContainingCells=0, InsideOut=0, FieldType='POINT', IDs=IDs )
#2}}}
    #Setup selection if not already defined{{{2
    try:
        selectionSource
    except NameError:
        IDs = [0L,long(NodNum)]
        if Cell:
            selectionSource = IDSelectionSource( ContainingCells=0, InsideOut=0, FieldType='CELL', IDs=[0L, NodNum] )
        else:
            selectionSource = IDSelectionSource( ContainingCells=0, InsideOut=0, FieldType='POINT', IDs=IDs )
#2}}}
    
    #Apply plot selection over time filter
    PSOT = PlotSelectionOverTime()
    if StatsOnly:
        PSOT.OnlyReportSelectionStatistics = 1
    else:
        PSOT.OnlyReportSelectionStatistics = 0

    PSOT.Selection = selectionSource
    PSOT.UpdatePipeline()
    
    return PSOT
    
#1}}}
#def wbv{{{1
def wbv(PipeLine,WarpVec=None,WarpScale=1.0):
    '''
    This function applies the paraview warp by vector filter
    Arguments:
    
    - *PipeLine* the paraview pipeline
    - *WarpVec* the vector which warps the mesh
    - *WarpScale* scaling factor
    '''
    
    SetActiveSource(PipeLine)
    num_arr = PipeLine.GetPointDataInformation().NumberOfArrays
    arr_list = [PipeLine.GetPointDataInformation().GetArray(x).Name for x in range(num_arr)]
    if WarpVec in arr_list:
        WBV = WarpByVector()
        WBV.ScaleFactor = WarpScale
        WBV.Vectors = ['POINTS',WarpVec]
        WBV.UpdatePipeline()
        return WBV
    else:
        print("Vector {0:s} not found, warp by vector filter not applied".format(WarpVec))
        return PipeLine
#1}}}
#def vf{{{1
def vf(PipeLine,FieldName="Displacements",FieldType="Point",Component="Magnitude",DataRange="Auto",ColorBar="off",CBopt=dict(),Representation="Surface",ViewSize="Tight",FigPix=600,RCP={},figsize=(0,),ViewBounds="Auto",showPOL=False,LineWidth=1.0,OutputFile=None,ViewTime=0):
    '''
    Generates 3D paraview surface plots and saves them to a .png file. The png figure is then loaded by matplotlib and if needed the colorbar is added. Finally the figure saved as a pdf.
    Arguements:
    
    - *DataName* Input pvd file
    - *PNGName* Name of png file to write to, note that the final figure will have the same name with pdf extension
    - *FieldName* Name of field for coloring the plot
    - *Component* The field component for coloring the plot
    - *DataRange* tuple specifying the data range for the color plot. By defualt, the range is determined automatically from the pvd data.
    - *ColorBar* flag to display a colorbar
    - *CBopt* Dictionary of options to display the colobar so far the accepted keys are Orientation, Location, Size, Pad, Label, Ticks, TickFmt, cmap and N
    - *CBOrientation* Orientation of colorbar, "Horizontal" or "Vertical" are the two options. By default the orientation is selected automatically based on the aspect ratio of the data
    - *CBLocation* Location of colorbar, either top, bottom, left, or right. By default the location is bottom if the orientation is horizontal and right if the orientation is vertical
    - *CBSize* size of the colorbar as a % of the figure size
    - *CBPad* padding between axes and colorbar
    - *CBLabel* label for the colorbar, by default it uses the FieldName value
    - *CBTicks* number of ticks for the colorbar
    - *CBTickFmt* specification for the tick format
    - *cmap* string containing the name of a matplotlib color map
    - *N* number of points in the color map
    - *RCP* matplotlib rcParams from parent process
    - *figsize* matplotlib figure size, taken from rcParams if unspecified
    - *WarpVec* Name of a vector field in the pvd data to use with the warp by vector filter.
    - *WarpScale* Scaling factor for warp by vector filter
    - *ClipData* Dictionary containing the option to perform a clip filter before wrinting an image 

    - *ViewTime* Time in the simulation that the plot is generated. ViewTime must be a string (first or last), integer time step index, or float time value
    - *Representation* Representation of the data, for example surface, surface with edges, wireframe, etc. Full details are in the paraview documentation
    - *ViewSize* size of the view in pixels, if ViewSize is Tight, than the larger dimension of the view if equal to the variable FigPix, and the smaller is computed from the aspect ratio
    - *FigPix* Larger dimension of the view if ViewSize is Tight
    '''
    SetActiveSource(PipeLine)
    #Unpacking of the CBopt parameters
    #CBopt = dict(Orientation="best", Location="best", Size="5%", Pad=0.05, Label='', Ticks=2, TickFmt='{:1.4f}', cmap="jet", N=256)
    CBOrientation = CBopt['Orientation'] if 'Orientation' in CBopt.keys() else 'best'
    CBLocation = CBopt['Location'] if 'Location' in CBopt.keys() else 'best'
    CBSize     = CBopt['Size'] if 'Size' in CBopt.keys() else '5%'
    CBPad      = CBopt['Pad'] if 'Pad' in CBopt.keys() else 0.05
    CBLabel    = CBopt['Label'] if 'Label' in CBopt.keys() else ''
    CBTicks    = CBopt['Ticks'] if 'Ticks' in CBopt.keys() else 2
    CBTickFmt  = CBopt['TickFmt'] if 'TickFmt' in CBopt.keys() else '{0:1.4f}'
    cmap       = CBopt['cmap'] if 'cmap' in CBopt.keys() else 'jet'
    N          = CBopt['N'] if 'N' in CBopt.keys() else 256
    
    
    #Set the DatInfo object to corresponding paraview object
    if FieldType=="Point":
        DatInfo = PipeLine.GetPointDataInformation()
    elif FieldType=="Cell":
        DatInfo = PipeLine.GetCellDataInformation()
    
    #Check for invalid fieldname
    if DatInfo.GetArray(FieldName) is None:
        print("Error, invalid {:s} FieldName {:s} for {:s}".format(FieldType,FieldName,DataName))
        return

    #Determine component
    if isinstance(Component,basestring):
        VectorMode = "Magnitude"
        VectorComponent = 0
    else:
        VectorMode = "Component"
        VectorComponent = Component

    #Set basic info{{{2
    RV1 = GetRenderView()
    RV1.ViewTime = ViewTime
    RenderView1 = GetRenderView()
    RenderView1.OrientationAxesVisibility = 0
    RenderView1.CenterAxesVisibility = 0
    DataRepresentation1 = Show()
    DataRepresentation1.ScalarOpacityUnitDistance = 0.00041136335173854968
    DataRepresentation1.EdgeColor = [0.0, 0.0, 0.50000762951094835]
    RenderView1.CameraClippingRange = [0.013523651497465943, 0.013865157848412055]
    RenderView1.Background2 = [0.0, 0.0, 0.16470588235294117]
    RenderView1.Background = [1.0, 1.0, 1.0]
    #2}}}
    #Get data bounds{{{2
    AutoBounds = PipeLine.GetDataInformation().GetBounds()
    if ViewBounds == "Auto":
        bounds = AutoBounds
    elif not isinstance(ViewBounds, basestring):
        VB = [auto if given == "Auto"  else given for auto,given in zip(AutoBounds,ViewBounds)]
        bounds = VB

    bounds_dx = bounds[1] - bounds[0]
    bounds_dy = bounds[3] - bounds[2]
    bounds_dz = bounds[5] - bounds[4]
    bounds_cx = (bounds[0] + bounds[1])/2.0
    bounds_cy = (bounds[2] + bounds[3])/2.0
    bounds_cz = (bounds[4] + bounds[5])/2.0
    if (bounds_dx == 0):
        # yz
        dimMode = 2
        aspect = bounds_dz/bounds_dy
    
    elif (bounds_dy == 0):
        # xz
        dimMode = 1
        aspect = bounds_dz/bounds_dx
    
    elif (bounds_dz == 0):
        #xy
        dimMode = 0
        aspect = bounds_dy/bounds_dx
    
    else:
        #3d
        dimMode = 3
        aspect = 1.0 # TODO
    #2}}}
    #Set view size{{{2
    if ViewSize == "Tight":
        ViewSize = [FigPix,int(float(FigPix)*aspect)]
    else:
        ViewSize = [FigPix,FigPix]

    RenderView1.ViewSize         = ViewSize
    #2}}}
    #Get data range for plot if not specified and return if DBOnly{{{2
    if isinstance(FieldName,basestring):
        if FieldName!= "":
            if VectorMode == "Magnitude":
                AutoRange = DatInfo.GetArray(FieldName).GetRange(-1)
            else:
                AutoRange = DatInfo.GetArray(FieldName).GetRange(Component)
                
        if DataRange == "Auto":
            DataRange = AutoRange
        elif not isinstance(DataRange, basestring):
            DR = [auto if given == "Auto"  else given for auto,given in zip(AutoRange,DataRange)]
            DataRange = DR
    #2}}}
    #Color plot {{{2
    DataRepresentation1.Representation = Representation
    if cmap == 'random':
        cmap    = cm.get_cmap(cmap_luc,N)
    else:
        cmap = cm.get_cmap(cmap,N)
        
    if isinstance(FieldName,basestring):
        if FieldName != "":
            rgb = mpl2pv(cmap,DataRange[0],DataRange[1],N)
            PVLookupTable = CreateLookupTable(RGBPoints=rgb, VectorMode=VectorMode,VectorComponent=VectorComponent, ColorSpace='HSV', ScalarRangeInitialized=1.0)


            DataRepresentation1.ColorArrayName = FieldName
            DataRepresentation1.LookupTable    = PVLookupTable
    elif isinstance(FieldName,list):
        DataRepresentation1.AmbientColor = FieldName
    #2}}}
    #Set up color bar{{{2
    cbd = {"horizontal":["bottom","top"],"vertical":["right","left"]}
    if ColorBar.lower() == "on":
        if CBOrientation.lower() == "best":
            #Determine best location based on aspect ratio
            if aspect > 1.0:
                CBOrientation = "vertical"
            else:
                CBOrientation = "horizontal"
        if CBLocation.lower() not in cbd[CBOrientation.lower()]:
            CBLocation = cbd[CBOrientation.lower()][0]
    #2}}}
    #Show plot over line{{{2
    if showPOL:
        #Line is (x1,x2,y1,y2,z1,z2)
        if Line == "Auto":
            Line = AutoBounds
        elif not isinstance(Line, basestring):
            line = []
            for auto,given in zip(AutoBounds,Line):
                if given == "Auto":
                    line.append(auto)
                elif given == "Xmin":
                    line.append(AutoBounds[0])
                elif given == "Xmax":
                    line.append(AutoBounds[1])
                elif given == "Ymin":
                    line.append(AutoBounds[2])
                elif given == "Ymax":
                    line.append(AutoBounds[3])
                elif given == "Zmin":
                    line.append(AutoBounds[4])
                elif given == "Zmax":
                    line.append(AutoBounds[5])
                else:
                    line.append(given)

            Line = list(line)



        SP1 = Line[0::2]
        SP2 = Line[1::2]
        #Plot over line
        POL = PlotOverLine( Source="High Resolution Line Source" )
        POL.Source.Point1 = SP1
        POL.Source.Point2 = SP2
        DR2 = Show()
        DR2.LineWidth=LineWidth
    #2}}}
    #Set camera position{{{2

    tanCamAn = math.tan(math.radians(RenderView1.CameraViewAngle))
    # position the camera
    if (dimMode == 0):
        # xy
        bounds_dx = bounds_dx*1.05
        bounds_dy = bounds_dy*1.05
        pos = max(bounds_dx, bounds_dy)
        pos2= pos/tanCamAn
        camUp = [0.0, 1.0, 0.0]
        camPos = [bounds_cx, bounds_cy,  pos2] #camera is positioned at the center of the image 
        camFoc = [bounds_cx, bounds_cy, 0.0] #camera will point at the center of the image
    
    elif (dimMode == 1):
        # xz
        bounds_dx = bounds_dx*Vfac
        bounds_cx = bounds_cx*Vfac
        bounds_dz = bounds_dz*Hfac
        bounds_cz = bounds_cz*Hfac
        pos = max(bounds_dx, bounds_dz)
        pos2= pos/tanCamAn
        camUp = [0.0, 0.0, 1.0]
        camPos = [bounds_cx, -pos2,  bounds_cz]
        camFoc = [bounds_cx,  0.0, bounds_cz]
    
    elif (dimMode == 2):
        # yz
        bounds_dy = bounds_dy*Vfac
        bounds_cy = bounds_cy*Vfac
        bounds_dz = bounds_dz*Hfac
        bounds_cz = bounds_cz*Hfac
        pos = max(bounds_dy, bounds_dz)
        pos2= pos/tanCamAn
        camUp = [0.0, 0.0, 1.0]
        camPos = [ pos2, bounds_cy, bounds_cz]
        camFoc = [0.0, bounds_cy, bounds_cz]
    
    else:
        # 3d
        print '3d cam position is yet TODO'
    
    RenderView1.CameraViewUp     = camUp
    RenderView1.CameraPosition   = camPos
    RenderView1.CameraFocalPoint = camFoc
    #2}}}

    #Save
    if OutputFile is None:
        print "Must specify output file name"
        return PipeLine

    WriteImage(OutputFile)
    print('Just wrote the image: {0:s}'.format(OutputFile))

    #Get final figure in matplotlib
    #Transfer matplotlib rcParams
    if len(RCP.keys()) > 0:
        plt.rcParams = RCP
    else:
        print("Information: rcParams not passed to view_field. For consistency, pass the rcParams of the current matplotlib environment with the RCP arguement")


    if len(figsize) < 2:
        figsize = plt.rcParams["figure.figsize"]
    fig = plt.figure(figsize=figsize)
    ax  = plt.gca()
    ax.set_frame_on(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    #load png that was just created by paraview
    img     = mpimg.imread(OutputFile)
    norm    = cols.Normalize(vmin=DataRange[0],vmax=DataRange[1])
    imgplot = plt.imshow(img,cmap=cmap,norm=norm)
    if ColorBar.lower() == "on":
        divider = make_axes_locatable(ax)
        cax     = divider.append_axes(CBLocation, size=CBSize, pad=CBPad)
        cbar    = plt.colorbar(imgplot,cax=cax,cmap=cmap,norm=norm,orientation=CBOrientation.lower())
        cbar.solids.set_edgecolor("face")
        if CBLabel:
            cbar.set_label(CBLabel)
        else:
            cbar.set_label(FieldName)
        print('vmin={0:1.4e}, vmax={1:1.4e}'.format(DataRange[0],DataRange[1]))
        ticks = np.linspace(DataRange[0],DataRange[1],num=CBTicks)
        tick_labels = [CBTickFmt.format(tick) for tick in ticks]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(tick_labels)
    root,ext = os.path.splitext(OutputFile)
    fig.savefig(root+".pdf",bbox_inches='tight')
    plt.close(fig)


#1}}}
#def clip{{{1
def clip(PipeLine,ClipType='Scalar',InsideOut=0,Scalars=None,Value=None,Radius=None,Center=None):
    '''
    This function applies the paraview warp by vector filter
    Arguments:
    
    - *PipeLine* the paraview pipeline
    - *ClipType* the type of clip
    - *InsideOut* Flag to invert clip
    '''
    
    SetActiveSource(PipeLine)
    if ClipType == 'Scalar':
        if Scalars is not None and Value is not None:
            num_arr = Dat.GetPointDataInformation().NumberOfArrays
            arr_list = [Dat.GetPointDataInformation().GetArray(x).Name for x in range(num_arr)]
            if Scalars in arr_list:
                Clp = Clip(ClipType=ClipType)
                Clp.Scalars   = ["POINTS",Scalars]
                Clp.Value     = Value
                Clp.InsideOut = InsideOut
                Clp.UpdatePipeline()
            else:
                print("Field {0:s} not found, clip filter not applied".format(Scalars))
                return PipeLine
        else:
            print("Keyword arguments Scalars and Value must be set for ClipType=Scalar, clip filter not applied")
            return PipeLine
    elif ClipType == 'Sphere':
        if Center is not None and Radius is not None:
            Clp = Clip(ClipType=ClipType)
            Clp.ClipType.Radius     = Radius
            Clp.ClipType.Center     = Center
            Clp.InsideOut = InsideOut
            Clp.UpdatePipeline()
        else:
            print("Keyword arguments Radius and Center must be set for ClipType=Sphere, clip filter not applied")
            return PipeLine



    return Clp
#1}}}
#def pvcsvin{{{1
def pvcsvin(csvfile):
    '''
    This function reads a csv file generated by a paraview filter. By paraview convention the first entry in each column is a name and the following entries are floats. pvcsvin thus returns a dictionary where the keys are the column names and the value is a list containing the float values.
    Arguements:
    
    - *csvfile* the file to be read
    '''
    #open .csv file and determine number of entries
    
    try:
        csvopen = open(csvfile,'r')
    except:
        print "Error, file does not exsist"
    
    fieldnames = csvopen.readline()
    fieldnames = fieldnames.split(',')
    fields = len(fieldnames)
    csvopen.close()
    
    reader = csv.reader(open(csvfile, "rb"),delimiter=',', quoting=csv.QUOTE_ALL)
    header = []
    records = []
    #get data from .csv
    for row, record in enumerate(reader):
    	if len(record) != fields:
    		print "Skipping malformed record %i, contains %i fields (%i expected)" %(record, len(record), fields)
    	else:
    		records.append(record)
    	
    fieldnames = records[0]
    #convert to float and transpose list
    result = [[float(r[col]) for r in records[1:]] for col in range(fields)]
    #store results in dictionary
    d = {}
    for i in range(fields):
    	d[fieldnames[i]] = result[i]
    	
    return d
#1}}}
#def iv_psot{{{1
def iv_psot(PVDFName,CSVFName,NodNum=0,ivflg=False,retcsv=True,AutoScale=True,Scale=None,Cell=False,StatsOnly=False,plotOverLine=False,Line="Auto",computeBoundary=False,WarpVec=None,Frustum=None,ExtractMode="POINT"):
    '''
    This function applies the paraview plot selection over time filter and optionally returns the result using pvcsvin. Integrate variable filter can also be optionally applied before the plot selection over time filter. The integrate varibles filter returns the volume integral of the field values multiplied by the volume of the domain. This volume is divided out with the AutoScale flag
    Arguments:
    
    - *PVDFName* the pvd file taken as input
    - *CSVFName* the csv file to write the filter results
    - *NodNum* the number of the node to which the filter is applied
    - *ivflg* flag to apply integrate variables filter
    - *retcsv* flag to return the results of the filter
    - *AutoScale* flag to automatically divide out the volume in the integrate variables filter
    - *Cell* flag to indicate if the selection is a cell and not a node
    '''
    paraview.simple._DisableFirstRenderCameraReset()
    if Cell:
        m1 = " on element number "+str(NodNum)
    else:   
        m1 = " on node number "+str(NodNum)


    #Open PVD
    PVDF = PVDReader( FileName=PVDFName )
    PVDF.UpdatePipeline()

    #Extract selection if prompted{{{2
    if Frustum is not None:
        Frustum = getFrustumFromBounds(Frustum)
        selectionSourceF = FrustumSelectionSource(ContainingCells=0,InsideOut=0,FieldType="CELL",Frustum=Frustum)
        #Create frustum selection and apply extract selection filter
        Dat = ExtractSelection()
        Dat.Selection = selectionSourceF
        Dat.UpdatePipeline()

        if ExtractMode == "POINT":
            numPointsSelected = Dat.GetDataInformation().DataInformation.GetNumberOfPoints()
            IDs = []
            for idx in range(numPointsSelected):
                IDs += [0L, idx]

            selectionSource = IDSelectionSource( ContainingCells=0, InsideOut=0, FieldType='POINT', IDs=IDs )
#2}}}
    #Setup selection if not already defined{{{2
    try:
        selectionSource
    except NameError:
        IDs = [0L,long(NodNum)]
        if Cell:
            selectionSource = IDSelectionSource( ContainingCells=0, InsideOut=0, FieldType='CELL', IDs=[0L, NodNum] )
        else:
            selectionSource = IDSelectionSource( ContainingCells=0, InsideOut=0, FieldType='POINT', IDs=IDs )
#2}}}
    #Plot over line if prompted{{{2
    if plotOverLine:
        AutoBounds = PVDF.GetDataInformation().GetBounds()
        #Line is (x1,x2,y1,y2,z1,z2)
        if Line == "Auto":
            Line = AutoBounds
        elif not isinstance(Line, basestring):
            line = []
            for auto,given in zip(AutoBounds,Line):
                if given == "Auto":
                    line.append(auto)
                elif given == "Xmin":
                    line.append(AutoBounds[0])
                elif given == "Xmax":
                    line.append(AutoBounds[1])
                elif given == "Ymin":
                    line.append(AutoBounds[2])
                elif given == "Ymax":
                    line.append(AutoBounds[3])
                elif given == "Zmin":
                    line.append(AutoBounds[4])
                elif given == "Zmax":
                    line.append(AutoBounds[5])
                else:
                    line.append(given)

            Line = list(line)



        SP1 = Line[0::2]
        SP2 = Line[1::2]
        AutoScale = False
        #Plot over line
        POL = PlotOverLine( Source="High Resolution Line Source" )
        POL.Source.Resolution = 100
        POL.Source.Point1 = SP1
        POL.Source.Point2 = SP2
        POL.UpdatePipeline()

        if computeBoundary:
            #assumes 2D
            vec  = np.array(SP1)-np.array(SP2)
            nrml = np.array([vec[1],-vec[0],0.])
            norm=np.linalg.norm(nrml)
            if norm==0: 
                pass
            else:
                nrml = nrml/norm

            SetActiveSource(POL)
            #Note the PK1 stresses are stored as follows:
            #[0,0] contains PK1_11
            #[1,0] contains PK1_22
            #[2,0] contains PK1_33
            #[0,1] contains PK1_12
            #[1,1] contains PK1_21
            script = "inp = inputs[0]\n" + \
                     "stressMeasure='Stresses'\n" + \
                     "prefix=''\n" + \
                     "npa = inp.GetPointData().GetNumberOfArrays()\n" + \
                     "nams= [0]*npa\n" + \
                     "if not inp.GetPointData() is None:\n" + \
                     "    if inp.GetPointData().GetNumberOfArrays() > 0:\n" + \
                     "        for PointArrayIdx in range(0,inp.GetPointData().GetNumberOfArrays()):\n" + \
                     "            name = inp.GetPointData().GetArrayName(PointArrayIdx)\n" + \
                     "            if name=='PK1 Stress':\n" + \
                     "                stressMeasure=name\n" + \
                     "                prefix='Reference '\n" + \
                     "            nams[PointArrayIdx] = name\n" + \
                     "for n in nams:\n" + \
                     "    data = inp.PointData[n]\n" + \
                     "    output.PointData.append(data,n)\n"+\
                     "nrml=["+str(nrml[0])+","+str(nrml[1])+"]\n" + \
                     "stressTensor = inp.PointData[stressMeasure]\n" + \
                     "velocity = inp.PointData['Velocities']\n" + \
                     "if 'Ref' in prefix:\n" +\
                     "    traction1 = nrml[0]*stressTensor[:,0,0]+nrml[1]*stressTensor[:,0,1]\n" +\
                     "    traction2 = nrml[0]*stressTensor[:,1,1]+nrml[1]*stressTensor[:,1,0]\n" +\
                     "    traction3 = 0.*stressTensor[:,0,0]\n" +\
                     "else:\n" +\
                     "    traction1 = nrml[0]*stressTensor[:,0]+nrml[1]*stressTensor[:,3]\n" +\
                     "    traction2 = nrml[0]*stressTensor[:,3]+nrml[1]*stressTensor[:,1]\n" +\
                     "    traction3 = 0.*stressTensor[:,0]\n" +\
                     "traction1.flatten()\n"+\
                     "traction2.flatten()\n"+\
                     "traction3.flatten()\n"+\
                     "Traction = numpy.array([traction1,traction2,traction3])\n" +\
                     "Traction = numpy.transpose(Traction[:,:,0])\n"+\
                     "boundaryWork = velocity[:,0]*traction1+velocity[:,1]*traction2\n" +\
                     "output.PointData.append(Traction,prefix+'Traction')\n" + \
                     "output.PointData.append(boundaryWork,prefix+'Boundary Work Rate')"
            ppf = ProgrammableFilter()
            ppf.Script = script
            ppf.UpdatePipeline()
            #2}}}
    #Integrate variables if prompted{{{2
    if (ivflg and not Cell):
        #Warp by vector if prompted 
        if WarpVec != None:
            num_arr = PVDF.GetPointDataInformation().NumberOfArrays
            arr_list = [PVDF.GetPointDataInformation().GetArray(x).Name for x in range(num_arr)]
            if WarpVec in arr_list:
                WBV = WarpByVector()
                WBV.Vectors = ['POINTS',WarpVec]
                WBV.UpdatePipeline()
            else:
                print "Vector",WarpVec,"not found, warp by vector filter not applied"
        m1 = ", integrate variables filter applied."
        IV = IntegrateVariables()
        IV.UpdatePipeline()
        NodNum = 0L
        if AutoScale:
            SetActiveSource(IV)
            script = "inp = inputs[0]\n" + \
                     "#Determine length, area, or volume\n" + \
                     "if " + str(Scale) +" is None:\n" +\
                     "    lav = ['Length','Area','Volume']\n" + \
                     "    if not inp.GetCellData() is None:\n" + \
                     "        if inp.GetCellData().GetNumberOfArrays() > 0:\n" + \
                     "            for CellArrayIdx in range(0,inp.GetCellData().GetNumberOfArrays()):\n" + \
                     "                data = inp.GetCellData().GetArray(CellArrayIdx)\n" + \
                     "                name = inp.GetCellData().GetArrayName(CellArrayIdx)\n" + \
                     "                if name in lav:\n" + \
                     "                    LAV = data.GetTuple(0)[0]\n" + \
                     "else:\n" +\
                     "    LAV="+str(Scale)+"\n"+\
                     "#Scale point data by LAV\n" + \
                     "npa = inp.GetPointData().GetNumberOfArrays()\n" + \
                     "nams= [0]*npa\n" + \
                     "if not inp.GetPointData() is None:\n" + \
                     "    if inp.GetPointData().GetNumberOfArrays() > 0:\n" + \
                     "        for PointArrayIdx in range(0,inp.GetPointData().GetNumberOfArrays()):\n" + \
                     "            name = inp.GetPointData().GetArrayName(PointArrayIdx)\n" + \
                     "            nams[PointArrayIdx] = name\n" + \
                     "for n in nams:\n" + \
                     "    data = inp.PointData[n]\n" + \
                     "    data = data/LAV\n" + \
                     "    output.PointData.append(data,n)"

            ppf = ProgrammableFilter()
            ppf.Script = script
            ppf.UpdatePipeline()
    #2}}}}
    
    #Apply plot selection over time filter
    PSOT = PlotSelectionOverTime()
    if StatsOnly:
        PSOT.OnlyReportSelectionStatistics = 1
    else:
        PSOT.OnlyReportSelectionStatistics = 0

    PSOT.Selection = selectionSource
    PSOT.UpdatePipeline()
    
    #Write to csv
    SetActiveSource(PSOT)
    CSVF = CSVWriter()
    CSVF.FileName = CSVFName
    CSVF.UpdatePipeline()

    #rename csv back to original name since paraview appends a 0
    spl = CSVFName.split('.')
    CSVFName0 = '.'.join([spl[0]+"0",spl[1]])
    os.rename(CSVFName0,CSVFName)

    pdir,pnam = os.path.split(PVDFName)
    message   = "Plot selection over time created for "+pnam+m1
    print message,"\n"
    
    for f in GetSources().values():
        Delete(f)
    #return the csv data if prompted
    if retcsv:
        return pvcsvin(CSVFName)
    
#1}}}
#def mpl2pv{{{1
def mpl2pv(cmap,xmin,xmax,N):
    '''
    Convert a matplotlib colorspace to the rgb format used by paraview
    Arguements

    - *cmap* a matplotlib colorspace
    - *xmin* minimum value in data
    - *xmax* maximum value in data
    - *N* number of points in colorspace
    '''
    x   = np.linspace(xmin,xmax,N)
    rgb = cmap(np.arange(N))

    rgbout = []
    for i,ls in enumerate(rgb):
        rgbout += [x[i],ls[0],ls[1],ls[2]]

    return rgbout
#1}}}
#def view_field{{{1
def view_field(DataName,PNGName,FieldName="Displacements",FieldType="Point",Component="Magnitude",DataRange="Auto",ColorBar="off",CBopt=dict(),WarpVec=None,WarpScale=1.0,ViewTime="Last",Representation="Surface",ViewSize="Tight",FigPix=600,RCP={},figsize=(0,),ViewBounds="Auto",showPOL=False,Line="Auto",LineWidth=1.0,Frustum=None,ClipData=None):
    '''
    Generates 3D paraview surface plots and saves them to a .png file. Optionally a warp by vector filter can be applied. The png figure is then loaded by matplotlib and if needed the colorbar is added. Finally the figure saved as a pdf.
    Arguements:
    
    - *DataName* Input pvd file
    - *PNGName* Name of png file to write to, note that the final figure will have the same name with pdf extension
    - *FieldName* Name of field for coloring the plot
    - *Component* The field component for coloring the plot
    - *DataRange* tuple specifying the data range for the color plot. By default, the range is determined automatically from the pvd data.
    - *ColorBar* flag to display a colorbar
    - *CBopt* Dictionary of options to display the colobar so far the accepted keys are Orientation, Location, Size, Pad, Label, Ticks, TickFmt, cmap and N
    - *CBOrientation* Orientation of colorbar, "Horizontal" or "Vertical" are the two options. By default the orientation is selected automatically based on the aspect ratio of the data
    - *CBLocation* Location of colorbar, either top, bottom, left, or right. By default the location is bottom if the orientation is horizontal and right if the orientation is vertical
    - *CBSize* size of the colorbar as a % of the figure size
    - *CBPad* padding between axes and colorbar
    - *CBLabel* label for the colorbar, by default it uses the FieldName value
    - *CBTicks* number of ticks for the colorbar
    - *CBTickFmt* specification for the tick format
    - *cmap* string containing the name of a matplotlib color map
    - *N* number of points in the color map
    - *RCP* matplotlib rcParams from parent process
    - *figsize* matplotlib figure size, taken from rcParams if unspecified
    - *WarpVec* Name of a vector field in the pvd data to use with the warp by vector filter.
    - *WarpScale* Scaling factor for warp by vector filter
    - *ClipData* Dictionary containing the option to perform a clip filter before wrinting an image 
        ClipData["Type"]      = so far only "Scalar" is implemented
        ClipData["Field"]     = the field used to perform the clip
        ClipData["Value"]     = threshold value underwitch data is clipped
        ClipData["InsideOut"] = 1 or 0

    .. todo::
        Maps with opacity other than 1 may not work properly

    - *ViewTime* Time in the simulation that the plot is generated. ViewTime must be a string (first or last), integer time step index, or float time value
    - *Representation* Representation of the data, for example surface, surface with edges, wireframe, etc. Full details are in the paraview documentation
    - *ViewSize* size of the view in pixels, if ViewSize is Tight, than the larger dimension of the view if equal to the variable FigPix, and the smaller is computed from the aspect ratio
    - *FigPix* Larger dimension of the view if ViewSize is Tight
    '''
    #Unpacking of the CBopt parameters
    #CBopt = dict(Orientation="best", Location="best", Size="5%", Pad=0.05, Label='', Ticks=2, TickFmt='{:1.4f}', cmap="jet", N=256)
    CBOrientation = CBopt['Orientation'] if 'Orientation' in CBopt.keys() else 'best'
    CBLocation = CBopt['Location'] if 'Location' in CBopt.keys() else 'best'
    CBSize     = CBopt['Size'] if 'Size' in CBopt.keys() else '5%'
    CBPad      = CBopt['Pad'] if 'Pad' in CBopt.keys() else 0.05
    CBLabel    = CBopt['Label'] if 'Label' in CBopt.keys() else ''
    CBTicks    = CBopt['Ticks'] if 'Ticks' in CBopt.keys() else 2
    CBTickFmt  = CBopt['TickFmt'] if 'TickFmt' in CBopt.keys() else '{0:1.4f}'
    cmap       = CBopt['cmap'] if 'cmap' in CBopt.keys() else 'jet'
    N          = CBopt['N'] if 'N' in CBopt.keys() else 256
    CBNoLabel  = CBopt['NoLabel']
    
    
    Dat = PVDReader(FileName=DataName)
    Dat.UpdatePipeline()
    
    #Set the DatInfo object to corresponding paraview object
    if FieldType=="Point":
        DatInfo = Dat.GetPointDataInformation()
    elif FieldType=="Cell":
        DatInfo = Dat.GetCellDataInformation()
    
    #Check for invalid fieldname
    if DatInfo.GetArray(FieldName) is None:
        print("Error, invalid {:s} FieldName {:s} for {:s}".format(FieldType,FieldName,DataName))
        return
   #set view time
    if isinstance(ViewTime,basestring):
        if ViewTime.lower() == "last":
            VT = Dat.TimestepValues[-1]
        if ViewTime.lower() == "first":
            VT = Dat.TimestepValues[0]
    elif isinstance(ViewTime,int):
        VT = Dat.TimestepValues[ViewTime]
    elif isinstance(ViewTime,(float,np.int64)):
        VT = ViewTime
    else:
        print("Error, ViewTime must be a string (first or last), integer index, or float time value")
        return
    
    
    #Determine component
    if isinstance(Component,basestring):
        VectorMode = "Magnitude"
        VectorComponent = 0
    else:
        VectorMode = "Component"
        VectorComponent = Component

    #Extract selection if prompted
    if Frustum is not None:
        #Create frustum selection and apply extract selection filter
        Frustum = getFrustumFromBounds(Frustum)
        selectionSource = FrustumSelectionSource(ContainingCells=0,InsideOut=0,FieldType="CELL",Frustum=Frustum)
        Dat = ExtractSelection()
        Dat.Selection = selectionSource
        Dat.UpdatePipeline()

    #Warp by vector if prompted{{{2 
    if WarpVec != None:
        num_arr = Dat.GetPointDataInformation().NumberOfArrays
        arr_list = [Dat.GetPointDataInformation().GetArray(x).Name for x in range(num_arr)]
        if WarpVec in arr_list:
            Dat = WarpByVector()
            Dat.ScaleFactor = WarpScale
            Dat.Vectors = ['POINTS',WarpVec]
            Dat.UpdatePipeline()
        else:
            print("Vector {:s} not found, warp by vector filter not applied".format(WarpVec))
    #2}}}
    #Clip by scalar if prompted{{{2 
    if ClipData != None:
        ClipType      = ClipData["Type"]
        ClipInsideOut = ClipData["InsideOut"]
        num_arr = Dat.GetPointDataInformation().NumberOfArrays
        arr_list = [Dat.GetPointDataInformation().GetArray(x).Name for x in range(num_arr)]
        if ClipType == 'Scalar':
            ClipField     = ClipData["Scalars"]
            if ClipField in arr_list:
                Dat = Clip(ClipType=ClipType)
                Dat.Scalars   = ["POINTS",ClipField]
                Dat.Value     = ClipData["Value"]
                Dat.InsideOut = ClipInsideOut
                Dat.UpdatePipeline()
        else:
            print("Field {:s} not found, clip filter not applied".format(ClipField))
    print VT
    #2}}}
    #Set basic info{{{2
    RV1 = GetRenderView()
    RV1.ViewTime = VT
    RenderView1 = GetRenderView()
    RenderView1.OrientationAxesVisibility = 0
    RenderView1.CenterAxesVisibility = 0
    DataRepresentation1 = Show()
    DataRepresentation1.ScalarOpacityUnitDistance = 0.00041136335173854968
    DataRepresentation1.EdgeColor = [0.0, 0.0, 0.50000762951094835]
    DataRepresentation1.Representation = 'Surface With Edges'
    RenderView1.CameraClippingRange = [0.013523651497465943, 0.013865157848412055]
    RenderView1.Background2 = [0.0, 0.0, 0.16470588235294117]
    RenderView1.Background = [1.0, 1.0, 1.0]
    #2}}}
    #Get data bounds{{{2
    AutoBounds = Dat.GetDataInformation().GetBounds()
    if ViewBounds == "Auto":
        bounds = AutoBounds
    elif not isinstance(ViewBounds, basestring):
        VB = [auto if given == "Auto"  else given for auto,given in zip(AutoBounds,ViewBounds)]
        bounds = VB

    bounds_dx = bounds[1] - bounds[0]
    bounds_dy = bounds[3] - bounds[2]
    bounds_dz = bounds[5] - bounds[4]
    bounds_cx = (bounds[0] + bounds[1])/2.0
    bounds_cy = (bounds[2] + bounds[3])/2.0
    bounds_cz = (bounds[4] + bounds[5])/2.0
    if (bounds_dx == 0):
        # yz
        dimMode = 2
        aspect = bounds_dz/bounds_dy
    
    elif (bounds_dy == 0):
        # xz
        dimMode = 1
        aspect = bounds_dz/bounds_dx
    
    elif (bounds_dz == 0):
        #xy
        dimMode = 0
        aspect = bounds_dy/bounds_dx
    
    else:
        #3d
        dimMode = 3
        aspect = 1.0 # TODO
    #2}}}
    #Set view size{{{2
    if ViewSize == "Tight":
        ViewSize = [FigPix,int(float(FigPix)*aspect)]
    else:
        ViewSize = [FigPix,FigPix]

    RenderView1.ViewSize         = ViewSize
    #2}}}
    #Get data range for plot if not specified and return if DBOnly{{{2
    if isinstance(FieldName,basestring):
        if FieldName!= "":
            if VectorMode == "Magnitude":
                AutoRange = DatInfo.GetArray(FieldName).GetRange(-1)
            else:
                AutoRange = DatInfo.GetArray(FieldName).GetRange(Component)

                
        if DataRange == "Auto":
            DataRange = AutoRange
        elif not isinstance(DataRange, basestring):
            DR = [auto if given == "Auto"  else given for auto,given in zip(AutoRange,DataRange)]
            DataRange = DR

    #2}}}
    #Color plot {{{2
    DataRepresentation1.Representation = Representation
    if cmap == 'random':
        cmap    = cm.get_cmap(cmap_luc,N)
    else:
        cmap = cm.get_cmap(cmap,N)
        
    if isinstance(FieldName,basestring):
        if FieldName != "":
            rgb = mpl2pv(cmap,DataRange[0],DataRange[1],N)
            PVLookupTable = CreateLookupTable(RGBPoints=rgb, VectorMode=VectorMode,VectorComponent=VectorComponent, ColorSpace='HSV', ScalarRangeInitialized=1.0)


            DataRepresentation1.ColorArrayName = FieldName
            DataRepresentation1.LookupTable    = PVLookupTable
    elif isinstance(FieldName,list):
        DataRepresentation1.AmbientColor = FieldName
    #2}}}
    #Set up color bar{{{2
    cbd = {"horizontal":["bottom","top"],"vertical":["right","left"]}
    if ColorBar.lower() == "on":
        if CBOrientation.lower() == "best":
            #Determine best location based on aspect ratio
            if aspect > 1.0:
                CBOrientation = "vertical"
            else:
                CBOrientation = "horizontal"
        if CBLocation.lower() not in cbd[CBOrientation.lower()]:
            CBLocation = cbd[CBOrientation.lower()][0]
    #2}}}
    #Show plot over line{{{2
    if showPOL:
        #Line is (x1,x2,y1,y2,z1,z2)
        if Line == "Auto":
            Line = AutoBounds
        elif not isinstance(Line, basestring):
            line = []
            for auto,given in zip(AutoBounds,Line):
                if given == "Auto":
                    line.append(auto)
                elif given == "Xmin":
                    line.append(AutoBounds[0])
                elif given == "Xmax":
                    line.append(AutoBounds[1])
                elif given == "Ymin":
                    line.append(AutoBounds[2])
                elif given == "Ymax":
                    line.append(AutoBounds[3])
                elif given == "Zmin":
                    line.append(AutoBounds[4])
                elif given == "Zmax":
                    line.append(AutoBounds[5])
                else:
                    line.append(given)

            Line = list(line)



        SP1 = Line[0::2]
        SP2 = Line[1::2]
        #Plot over line
        POL = PlotOverLine( Source="High Resolution Line Source" )
        POL.Source.Point1 = SP1
        POL.Source.Point2 = SP2
        DR2 = Show()
        DR2.LineWidth=LineWidth
    #2}}}
    #Set camera position{{{2

    tanCamAn = math.tan(math.radians(RenderView1.CameraViewAngle))
    # position the camera
    if (dimMode == 0):
        # xy
        bounds_dx = bounds_dx*1.05
        bounds_dy = bounds_dy*1.05
        pos = max(bounds_dx, bounds_dy)
        pos2= pos/tanCamAn
        camUp = [0.0, 1.0, 0.0]
        camPos = [bounds_cx, bounds_cy,  pos2] #camera is positioned at the center of the image 
        camFoc = [bounds_cx, bounds_cy, 0.0] #camera will point at the center of the image
    
    elif (dimMode == 1):
        # xz
        bounds_dx = bounds_dx*Vfac
        bounds_cx = bounds_cx*Vfac
        bounds_dz = bounds_dz*Hfac
        bounds_cz = bounds_cz*Hfac
        pos = max(bounds_dx, bounds_dz)
        pos2= pos/tanCamAn
        camUp = [0.0, 0.0, 1.0]
        camPos = [bounds_cx, -pos2,  bounds_cz]
        camFoc = [bounds_cx,  0.0, bounds_cz]
    
    elif (dimMode == 2):
        # yz
        bounds_dy = bounds_dy*Vfac
        bounds_cy = bounds_cy*Vfac
        bounds_dz = bounds_dz*Hfac
        bounds_cz = bounds_cz*Hfac
        pos = max(bounds_dy, bounds_dz)
        pos2= pos/tanCamAn
        camUp = [0.0, 0.0, 1.0]
        camPos = [ pos2, bounds_cy, bounds_cz]
        camFoc = [0.0, bounds_cy, bounds_cz]
    
    else:
        # 3d
        print '3d cam position is yet TODO'
    
    RenderView1.CameraViewUp     = camUp
    RenderView1.CameraPosition   = camPos
    RenderView1.CameraFocalPoint = camFoc
    #2}}}

    #Save
    WriteImage(PNGName)
    print('Just wrote the image: {0:s}'.format(PNGName))
    for f in GetSources().values():
        Delete(f)

    #Get final figure in matplotlib
    #Transfer matplotlib rcParams
    if len(RCP.keys()) > 0:
        plt.rcParams = RCP
    else:
        print("Information: rcParams not passed to view_field. For consistency, pass the rcParams of the current matplotlib environment with the RCP arguement")


    if len(figsize) < 2:
        figsize = plt.rcParams["figure.figsize"]
    fig = plt.figure(figsize=figsize)
    ax  = plt.gca()
    ax.set_frame_on(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    #load png that was just created by paraview
    img     = mpimg.imread(PNGName)
    norm    = cols.Normalize(vmin=DataRange[0],vmax=DataRange[1])
    imgplot = plt.imshow(img,cmap=cmap,norm=norm)
    if ColorBar == "On":
        divider = make_axes_locatable(ax)
        cax     = divider.append_axes(CBLocation, size=CBSize, pad=CBPad)
        cbar    = plt.colorbar(imgplot,cax=cax,cmap=cmap,norm=norm,orientation=CBOrientation.lower())
        cbar.solids.set_edgecolor("face")
        if not CBNoLabel:
            if CBLabel:
                cbar.set_label(CBLabel)
            else:
                cbar.set_label(FieldName)
        print('vmin={0:1.4e}, vmax={1:1.4e}'.format(DataRange[0],DataRange[1]))
        ticks = np.linspace(DataRange[0],DataRange[1],num=CBTicks)
        tick_labels = [CBTickFmt.format(tick) for tick in ticks]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(tick_labels)
    root,ext = os.path.splitext(PNGName)
    fig.savefig(root+".pdf",bbox_inches='tight')
    plt.close(fig)


#1}}}
#def IsOld{{{1
def IsOld(Data,Graphic,Coerce=False):
    '''
    Return true if file graphic is older than file data, if file graphic does not exsist, or if Coerce is true. This is used to prevent unneccesary recomputation of post processing functions.
    Arguements:
    
    - *Data* File 1, typically some kind of raw data file that would be used as an input to post rpocessing function
    - *Graphic* File 2, typically some kind of graphic or output file from a post processing function
    - *Coerce* Automatically return True if Coerce is True
    '''
    if Coerce or not Graphic: return True
    try:
        with open(Graphic) as f: pass
    except IOError as e:
        return True

    GStat = os.stat(Graphic).st_mtime
    DStat = os.stat(Data).st_mtime
    if DStat > GStat: return True

    return False
#1}}}
#def addtopvd{{{1
def addtopvd(fo,vtkname,ttim,nproc=1):
    '''
    Used internally by prepost_tools.pvdgen to add a timestep to a pvd collection 
    Arguements:

    - *fo* pvd file for output
    - *vtkname* name of the vtu to be added to the collection
    - *ttim* time value associated with the vtu
    - *nproc* number of processors
    '''
    wrtfo = fo.write
    for j in xrange(nproc):
        l1 = '<DataSet timestep="'+ttim+'" part="'+str(j)+'" file="'+vtkname+'"/>'+'\n'
        wrtfo(l1)
    return None
        
#1}}}
#def pvdgen {{{1
def pvdgen(PVDDir,PVDName,keep="All",nproc=1,rmvtk=False,donaninfck="last"):
    '''
    Generate pvd file from series of vtu files 
    Arguements:

    - *PVDDir* Directory where vtu files reside, also the directory that the pvd file will be written to.
    - *PVDName* Name of the output pvd file
    - *keep* used to generate indices of vtu files to keep in the pvd collection. The default is "All" which keeps all vtus. If keep is not "All" it must be a list of integer triplet list where each triplet is in the form [start, end, stride]. An arbirtrary number of start-end-stride triplets can be used, and stride is assumed to be one if it is unspecified. For example keep = [[0,100],[100,500,20]] will keep every vtu from index 0 to 100 and every 20 vtus from index 100 to 500.

    - *nproc* number of processors

    .. todo::
        It appears the multiprocessor case is only partially implemented

    - *rmvtk* Flag for deletion of vtu files which are not in the keep list
    - *donaninfck* Flag to check vtus for Nan or infinity which will cause a paraview error. If nan or inf is found in the vtu it is not added to the pvd collection. donaninfck may be "last" which only checks the vtu file with the largest index. This is where nan or inf is mostly found in vtus and might occur if your simulation exited with a nan or inf residual. donaninfck = "all" checks all vtus, however this is very time consuming.

    .. todo::
        This may be obsolete due to a naninf check in umacr1.f where the paraview data is written.
    '''
    ofile = os.path.join(PVDDir,PVDName)
    
    #{{{2 Generate list of items to keep if not all
    keeplist = []
    if keep != "All":
        for trip in keep:
            trip[1] +=1 #Add 1 to end so the range is inclusive
            keeplist += range(*trip)
        mxk = max(keeplist)
    #2}}}
    #generate .pvd  {{{2
    th = '<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n<Collection>\n'
    bh = '</Collection>\n</VTKFile>'
    fo = open(ofile,'w')
    fo.write(th)
    prm = os.remove
    pex = os.path.exists
    rmtm = 0.0
    adtm = 0.0
        
    vlist = glob.glob( os.path.join(PVDDir, '*.vtu') )
    vlist.sort()
    lenv  = len(vlist)
    for vnum,vtknam in enumerate(vlist):
        
        #get vtu file name
        ts = vtknam[-10:-4]
        tsl = re.sub("^0+","",ts)
        if tsl == "":
            tsi = 0
        else:
            tsi = int(tsl)

        #Skip vtu if nan or inf is found
        dn1 = donaninfck.lower() == "last" and vnum == lenv-1
        dn2 = donaninfck.lower() == "all" 
        if dn1 or dn2:
            grp = subprocess.Popen(('grep','-i','-w','nan\|-infinity\|+infinity\|infinity','%s'%vtknam),stdout = subprocess.PIPE)
            grpout   = grp.communicate()[0]
            naninfck = len(grpout) == 0
        else:
            naninfck = True
        keepck   = tsi in keeplist or keep == 'All'
        if keepck and naninfck:
            #determine time 
            fi = open(vtknam,'r')
            for j,line in enumerate(fi):
                if j == 5:
                    ttim = line.split()[0]
                elif j > 5:
                    break
            fi.close()
            addtopvd(fo,os.path.basename(vtknam),ttim)
        else:
            if rmvtk == True:
                prm(vtknam)
    
    fo.write(bh)
    fo.close()
    
    print "Paraview collection created for",PVDName
    #2}}}
#1}}}
#def getcpuit{{{1
def getcpuit(FilDirs,EndStep,parfeap='only'):
    '''
    Function used to extract the CPU time spent at each time step and the overall wall-time of the simulation. The number of Nonlinear iterations is also returned.
    Arguements:
    
    - *FilDirs* directory containing the Lfile to parse
    - *EndStep* assumed final step of the simulation
    - *parfeap* flag to indicate if the information will be stored in a file ended with the parallel flag '_0001'.

    .. todo::
        At present this takes several directories at once because this function is typically used to compare different meshes. However to be more in line with the conventions of the rest of the module, I think it should only do 1 job at a time. Users can more easily create mesh convergence plots using the FEAPJob.ReshapeJobs() function.
    '''
    pfeap_opt = {'all':'[L]*_[0001]', 'only':'[L]*_0001', 'none':'[L]*'}
    lenfd = len(FilDirs)
    Memory    = [0.0]*lenfd
    CPUTime   = [0.0]*lenfd
    Tang      = [0]*lenfd
    Form      = [0]*lenfd
    FinalStep = [0]*lenfd
    for i,Dir in enumerate(FilDirs):
        flist=glob.glob(os.path.join(Dir, pfeap_opt[parfeap]))
        for ilg in flist:
            f = open(ilg,'r')
            begin = 0
            end   = 0
            for j,line in enumerate(f):
                sp = line.split()
                if "Step" in sp:
                    begin = j+2
                if (begin != 0) and not sp:
                    break
                elif (begin != 0) and (j >= begin) and sp:
                    Tang[i] += int(sp[1])
                    Form[i] += int(sp[2])
                    CPUTime[i] = float(sp[-2])
                    Memory[i] = float(sp[-1])
                    FinalStep[i] += 1

            f.close()

    return {'Form': Form,'Tang': Tang,'CPUTime': CPUTime, 'Memory':Memory, 'FinalStep': FinalStep}
#1}}}
#def getconv{{{1
def getconv(LFN):
    "Extract convergence informations from an L file generated by FEAP"
    f = open(LFN,'r')
    begin = 0
    record  = np.zeros( (1,11) )
    records = np.empty( (0,11) )
    for j,line in enumerate(f):
        sp = line.split()
	if "Step" in sp:
            begin = j+2
	if (begin != 0) and (j >= begin):
            #Check if we reached the end of the input!
            if not sp:
                break
	    for k,item in enumerate(sp):
                record[0,k] = float(item)
	    records = np.vstack( (records,record) )
    f.close()
    return records
#1}}}
#def pol_minmax{{{1
def pol_minmax(name):
    '''
    Returns a dictionary containg the minimum and maximum values of a time series of plot over line filters
    Arguements:
    
    - *name* name of the csv collection containing the results of the pol filter

    '''
    fileName, fileExtension = os.path.splitext(name)
    minmax_dict={}
    for cn in glob.glob(fileName+'*'+fileExtension):
        pol = pvcsvin(cn)
        for key in pol:
            keymin = np.nanmin(pol[key])
            keymax = np.nanmax(pol[key])
            if key not in minmax_dict.keys():
                minmax_dict[key] = [keymin,keymax]
            else:
                newmin = np.nanmin([keymin,minmax_dict[key][0]])
                newmax = np.nanmax([keymax,minmax_dict[key][1]])
                minmax_dict[key] = [newmin,newmax]
        
    return minmax_dict
#1}}}
#def getpolcsv{{{1
def getpolcsv(PVDFName,CSVFName,rtime="Last"):
    '''
    This function gets the name and number of the data (csv) file for a pol result in the desired timestep.
    Output: ['nameofcsv.rtime.csv','rtime']
    Arguements:
    
    - *PVDFName* name of the pvd collection
    - *CSVFName* name of the csv collection containing the results of the pol filter
    - *rtime* Time in the simulation that the plot is generated. ViewTime must be a string (last) pr integer time step index

    '''
    
    #Open PVD
    PVDF = PVDReader( FileName=PVDFName )
    PVDF.UpdatePipeline()

    lasttime=len(PVDF.TimestepValues)-1

    if isinstance(rtime,basestring):
        if rtime.lower() == "last":
           rtime = str(lasttime)
    
    if lasttime<int(rtime):
        rtime=str(lasttime)
        print('Warning-POL: Timestep number too big. Maximum used')

    spl = CSVFName.split('.')
    CSVFName0 = '.'.join([spl[0]+"."+rtime,spl[1]])
   
    for f in GetSources().values():
        Delete(f)

    return [CSVFName0,rtime] 
   
#1}}}
#def pol{{{1
def pol(PVDFName,CSVFName,Line="Auto",WarpScale=1.0,WarpVec=None,retcsv=True,WriteAllTimeSteps=1,ViewTime="Last",Resolution=100):
    '''
    Applies paraview plot over line filter. A warp by vector filter can optionally be applied before the plot over line filter
    Arguements:
    
    - *PVDFName* name of the pvd collection
    - *CSVFName* name of the csv collection containing the results of the pol filter
    - *SP1* X,Y,Z coordinates of point 1 defining the line
    - *SP2* X,Y,Z coordinates of point 2 defining the line
    - *WarpVec* Name of a vector field in the pvd data to use with the warp by vector filter.
    - *retcsv* flag to return the results of the filter
    - *WriteAllTimeSteps* flag to create a plot over line csv for all time steps in the data
    - *rtime* Time in the simulation that the plot is generated. ViewTime must be a string (last) or integer time step index
    - *Resolution* resolution of the pol filter.

    '''
    m1 = ""
    m2 = ""
    #Open PVD
    Dat = PVDReader( FileName=PVDFName )
    Dat.UpdatePipeline()
    if WriteAllTimeSteps == 1:
        m2 = ", all time steps written."
    else:
        m2 = ", for time " + str(ViewTime)

    #set view time
    if isinstance(ViewTime,basestring):
        if ViewTime.lower() == "last":
            VT = Dat.TimestepValues[-1]
        if ViewTime.lower() == "first":
            VT = Dat.TimestepValues[0]
    elif isinstance(ViewTime,int):
        VT = Dat.TimestepValues[ViewTime]
    elif isinstance(ViewTime,(float,np.int64)):
        VT = ViewTime
    else:
        print "Error, ViewTime must be a string (first or last), integer index, or float time value"
        return
    #Warp by vector if prompted{{{2 
    if WarpVec != None:
        num_arr = Dat.GetPointDataInformation().NumberOfArrays
        arr_list = [Dat.GetPointDataInformation().GetArray(x).Name for x in range(num_arr)]
        if WarpVec in arr_list:
            Dat = WarpByVector()
            Dat.ScaleFactor = WarpScale
            Dat.Vectors = ['POINTS',WarpVec]
            Dat.UpdatePipeline()
        else:
            print "Vector",WarpVec,"not found, warp by vector filter not applied"
    #2}}}

    RV1 = GetRenderView()
    RV1.ViewTime = VT
    RenderView1 = GetRenderView()
    DataRepresentation1 = Show()

    AutoBounds = Dat.GetDataInformation().GetBounds()
    #Line is (x1,x2,y1,y2,z1,z2)
    if Line == "Auto":
        Line = AutoBounds
    elif not isinstance(Line, basestring):
        line = []
        for auto,given in zip(AutoBounds,Line):
            if given == "Auto":
                line.append(auto)
            elif given == "Xmin":
                line.append(AutoBounds[0])
            elif given == "Xmax":
                line.append(AutoBounds[1])
            elif given == "Ymin":
                line.append(AutoBounds[2])
            elif given == "Ymax":
                line.append(AutoBounds[3])
            elif given == "Zmin":
                line.append(AutoBounds[4])
            elif given == "Zmax":
                line.append(AutoBounds[5])
            else:
                line.append(given)

        Line = list(line)



    SP1 = Line[0::2]
    SP2 = Line[1::2]
    #Plot over line
    POL = PlotOverLine( Source="High Resolution Line Source" )
    POL.Source.Resolution = Resolution
    POL.Source.Point1 = SP1
    POL.Source.Point2 = SP2
    POL.UpdatePipeline()

    #Write to csv
    SetActiveSource(POL)
    CSVF = CreateWriter(CSVFName,POL)
    CSVF.FieldAssociation = "Points"
    CSVF.WriteAllTimeSteps = WriteAllTimeSteps
    CSVF.UpdatePipeline()
    spl = CSVFName.split('.')
    if WriteAllTimeSteps == 1:
        CSVFName0 = '.'.join([spl[0]+"."+ViewTime,spl[1]])
    else:
        CSVFName0 = CSVFName

    pdir,pnam = os.path.split(PVDFName)
    message   = "Plot over line created for "+pnam+m1+m2
    print message,"\n"

    for f in GetSources().values():
        Delete(f)
    #return the csv data if prompted
    if retcsv:
        return pvcsvin(CSVFName0)

#1}}}
#def GetArrayRange{{{1
def GetArrayRange(File,FieldName,ViewTime='Last',Component="Magnitude"):
    '''
    Determine the range of the data for a certain field in a vtu
    Arguments:
    
    - *File* name of the vtu
    - *FieldName* name of the field
    '''

    Dat = PVDReader( FileName=File)
    Dat.UpdatePipeline()
    #Check for invalid fieldname
    if Dat.GetPointDataInformation().GetArray(FieldName) is None:
        print "Error, invalid FieldName ",FieldName,"for ",DataName
        return
    #set view time
    if isinstance(ViewTime,basestring):
        if ViewTime.lower() == "last":
            VT = Dat.TimestepValues[-1]
        if ViewTime.lower() == "first":
            VT = Dat.TimestepValues[0]
    elif isinstance(ViewTime,int):
        VT = Dat.TimestepValues[ViewTime]
    elif isinstance(ViewTime,(float,np.int64)):
        VT = ViewTime
    else:
        print "Error, ViewTime must be a string (first or last), integer index, or float time value"
        return

    RV1 = GetRenderView()
    RV1.ViewTime = VT
    RenderView1 = GetRenderView()
    DataRepresentation1 = Show()
    #Determine component
    if isinstance(Component,basestring):
        VectorMode = "Magnitude"
        VectorComponent = 0
    else:
        VectorMode = "Component"
        VectorComponent = Component

    if isinstance(FieldName,basestring):
        if FieldName!= "":
            if VectorMode == "Magnitude":
                AutoRange = Dat.GetPointDataInformation().GetArray(FieldName).GetRange()
            else:
                AutoRange = Dat.GetPointDataInformation().GetArray(FieldName).GetRange(Component)

    Delete(Dat)
    return AutoRange
#1}}}
#def write_file {{{1
def write_file(fil,content):
    '''
    Write a string to a file line by line. Used internally by FEAPJob.createjobs() to generate simulation files from base files
    Arguements:
    
    - *fil* name output file
    - *content* string containing the contents of the file
    '''
    with open(fil,"w") as f:
        for line in content:
            f.write("%s\n" % line)

#1}}}
#def read_file {{{1
def read_file(fil):
    '''
    Read a file to a string. Used internally by FEAPJob.createjobs() to read base files
    Arguements:
    
    - *fil* name output file
    - *content* string containing the contents of the file
    '''
    with open(fil) as f:
        lines = f.read().splitlines()

    return lines
#1}}}
#def sub_files{{{1
def sub_files(files,subdict,ap=""):
    '''
    Substitute lines in a string based on lines containing keywords. Used internally by FEAPJob.createjobs() to substitute parameters into base file strings
    Arguements:
    
    - *files* List containing [File_1,File_2...,File_n] Where the Files are lists of strings where each list item is a line of the file. This is the output format of prepost_tools.read_file().
    - *subdict* Dictionary whose keys are keywords in the base files. Each keyword in the base files will be replaced with the corresponding dictionary value.
    - *ap* string to be appended to the end of every line substituted

    '''
    subbedFiles = []
    for f in files:
        sf = []
        for line in f:
            if line in subdict.keys():
                sline = subdict[line] + ap
            else:
                sline = line
            sf.append(sline)
        subbedFiles.append(sf)


    return subbedFiles
            
#1}}}
#def RunFEAPJob{{{1
def RunFEAPJob(jobdir,jobname,feapex,parfeap=True,PETScLinearSolverOptions="-ksp_type preonly -pc_type lu -ksp_diagonal_scale",NURBS=False):
    '''
    Executes a feap job.
    Arguements:
    
    - *jobdir* working directory for the job
    - *jobname* name of the job
    - *feapex* feap executable file
    - *parfeap* flag for a parfeap job. If true, feap is called twice, the first time to generate the parallel input files, and the second time to run the parallel job.
    - *PETScLinearSolverOptions* Additional command line options to pass to feap. Typically these are options for petsc, which is the linear solver employed by feap.
    - *NURBS* Flag for copying nurbs specific input file data to the parallel input file 

    .. todo::
        multiprocessor case will need modification to call mpirun
    '''
    os.chdir(jobdir)
    prefx    = ["I","O","R","R","P"]
    name     = "".join([i+jobname+"\n" for i in prefx])
    try:
        with open('feapname') as f:
            comm = "n\n"+name+"y\n"
    except IOError as e:
        comm = name+"y\n"

    pfeap = (feapex+" "+PETScLinearSolverOptions).split()
    proc = subprocess.Popen(pfeap,stdin=subprocess.PIPE)
    proc.communicate(comm)

    if NURBS:
        pinpt = "".join([prefx[0]+jobname+"_0001"])
        f = open(pinpt,'r')
        pinpt_cont = f.readlines()
        f.close()
        for i,line in enumerate(pinpt_cont):
            if line == 'NOPRint ! Do not echo data to output file\n':
                ind = i
        n_name = NURBS.split('/')[-1]
        pinpt_cont[ind+1:ind+1] = ['\n', 'INCLude '+n_name+'\n']
        f = open(pinpt,'w')
        for line in pinpt_cont:
            f.write(line)
        f.close()
    if parfeap:
        pname     = "".join([i+jobname+"_0001\n" for i in prefx])
        pcomm = "n\n"+pname+"y\n"

        proc = subprocess.Popen(pfeap,stdin=subprocess.PIPE)
        proc.communicate(pcomm)
#1}}}
#def GetError{{{1
def GetError(Data,CSVFName,RefIndx,Coord,tol=1.e-11,Ctype="index",retcsv=False,WriteAllTimeSteps=1,rtime="last"):
    DataF = [PVDReader( FileName=D) for D in Data]
    [D.UpdatePipeline() for D in DataF]
    ppf = ProgrammableFilter(Input=DataF)
    script = "from numpy.linalg import norm\n" + \
             "RefIndx = "+str(RefIndx)+"\n" + \
             "ErfIndx = range(len(inputs))\n" + \
             "ErfIndx.pop(RefIndx)\n" + \
             "if "+str(Ctype.lower() == 'index')+":\n" + \
             "    Epoints  = numpy.array("+str(Coord)+")\n" + \
             "names = [inputs[RefIndx].GetPointData().GetArrayName(n) for n in range(inputs[RefIndx].GetPointData().GetNumberOfArrays())]\n" + \
             "err_dict = dict((n,[]) for n in names)\n" + \
             "for i,erf in enumerate(inputs):\n" + \
             "    if i in ErfIndx:\n" + \
             "        enames = [erf.GetPointData().GetArrayName(n) for n in range(erf.GetPointData().GetNumberOfArrays())]\n" + \
             "        for en in enames:\n" + \
             "            if en in names:\n" + \
             "                er    = numpy.array(erf.PointData[en][Epoints])\n" + \
             "                rf    = numpy.array(inputs[RefIndx].PointData[en][Epoints])\n" + \
             "                if numpy.shape(er) == numpy.shape(rf):\n" + \
             "                    ermrf  = er - rf\n" + \
             "                    nermrf = norm(ermrf)\n" + \
             "                    nrf    = norm(rf)\n" + \
             "                    if nrf != 0.:\n" + \
             "                        ernrm = nermrf/nrf\n" + \
             "                    else:\n" + \
             "                        ernrm = nermrf\n" + \
             "                    err_dict[en].append(ernrm)\n" + \
             "for key in err_dict:\n" + \
             "    output.RowData.append(numpy.array(err_dict[key]),'err_'+key) \n"

    ppf.Script = script
    ppf.OutputDataSetType = 'vtkTable'
    ppf.UpdatePipeline()

    if isinstance(rtime,basestring):
        if rtime.lower() == "last":
            rtime = str(len(DataF[RefIndx].TimestepValues)-1)
    #Write to csv
    SetActiveSource(ppf)
    CSVF = CreateWriter(CSVFName,ppf)
    CSVF.WriteAllTimeSteps = WriteAllTimeSteps
    CSVF.UpdatePipeline()
    spl = CSVFName.split('.')
    CSVFName0 = '.'.join([spl[0]+"."+rtime,spl[1]])

    m2 = ""
    if WriteAllTimeSteps == 1:
        m2 = ", all time steps written."
    message   = "Errors computed "+m2
    print message,"\n"

    for f in GetSources().values():
        Delete(f)
    #return the csv data if prompted
    if retcsv:
        return pvcsvin(CSVFName0)
#1}}}
#def PlotFieldDataOverTime{{{1
def PlotFieldDataOverTime(PVD,CSVFName,retcsv=False):
    '''
    Applies the paraview plot selection over time filter to field data and optionally returns the result using pvcsvin. Since plot selection over time expects a point or a cell as the selection source, this function creates a point and copies the desired field data to it, permitting use of the psot filter
    Arguements:
    
    - *PVD* the pvd file taken as input
    - *CSVFName* the csv file to write the filter results
    - *retcsv* flag to return the results of the filter
    '''
    #Open PVD
    PVDF = PVDReader( FileName=PVD )
    PVDF.UpdatePipeline()
    script = "inp = self.GetInput()\n" + \
             "out = self.GetPolyDataOutput()\n" + \
             "newPoint = vtk.vtkPoints()\n" + \
             "newPoint.InsertPoint(0,0,0,0)\n" + \
             "out.SetPoints(newPoint)\n" + \
             "newCell = vtk.vtkCellArray()\n" + \
             "vertex = vtk.vtkVertex()\n" + \
             "vertex.GetPointIds().SetId(0,0)\n" + \
             "newCell.InsertNextCell(vertex)\n" + \
             "out.SetVerts(newCell)\n" + \
             "if not inp.GetFieldData() is None:\n" + \
             "    if inp.GetFieldData().GetNumberOfArrays() > 0:\n" + \
             "        for fieldArrayIdx in range(0,inp.GetFieldData().GetNumberOfArrays()):\n" + \
             "            data = inp.GetFieldData().GetArray(fieldArrayIdx)\n" + \
             "            name = inp.GetFieldData().GetArrayName(fieldArrayIdx)\n" + \
             "            # Use first entry of n-th field data array\n" + \
             "            if not data is None:\n" + \
             "                ncomp = data.GetNumberOfComponents()\n" + \
             "                ntup  = data.GetNumberOfTuples()\n" + \
             "                if ntup > 1:\n" + \
             "                    for i in range(ntup):\n" + \
             "                        newData = vtk.vtkFloatArray()\n" + \
             "                        newData.SetNumberOfComponents(ncomp)\n" + \
             "                        newData.SetNumberOfTuples(1)\n" + \
             "                        newData.SetTuple(0,data.GetTuple(i))\n" + \
             "                        newData.SetName(name+str(i))\n" + \
             "                        out.GetPointData().AddArray(newData)\n" + \
             "                else:\n" + \
             "                    newData = vtk.vtkFloatArray()\n" + \
             "                    newData.SetNumberOfComponents(ncomp)\n" + \
             "                    newData.SetNumberOfTuples(1)\n" + \
             "                    newData.SetTuple(0,data.GetTuple(0))\n" + \
             "                    newData.SetName(name)\n" + \
             "                    out.GetPointData().AddArray(newData)"

    ppf = ProgrammableFilter()
    ppf.Script = script
    ppf.OutputDataSetType = 0
    ppf.UpdatePipeline()
    #Apply plot selection over time filter
    PSOT = PlotSelectionOverTime()
    selection_source = IDSelectionSource( ContainingCells=0, InsideOut=0, FieldType='POINT', IDs=[-1L, 0L] )
    PSOT.Selection = selection_source
    
    #Write to csv
    SetActiveSource(PSOT)
    CSVF = CSVWriter()
    CSVF.FileName = CSVFName
    CSVF.UpdatePipeline()

    #rename csv back to original name since paraview appends a 0
    spl = CSVFName.split('.')
    CSVFName0 = '.'.join([spl[0]+"0",spl[1]])
    os.rename(CSVFName0,CSVFName)
    
    message   = "Plot selection over time created for Field Quantities"
    print message,"\n"
    
    for f in GetSources().values():
        Delete(f)
    #return the csv data if prompted
    if retcsv:
        return pvcsvin(CSVFName)
#1}}}
#def xmlPFDOT{{{1
def xmlPFDOT(vtu):
    '''
    .. todo::
        anyone know what this if for?
    '''
    tree = ET.parse(vtu)
    unsg = tree.getroot().find("UnstructuredGrid")
    fd   = unsg.find('FieldData')
    for DataArray in fd:
        print DataArray.tag,DataArray.attrib
        print DataArray.text,"\n"
#1}}}
#def find_value{{{1
def find_value(a,target,FracOf=None,NoneMax="Last"):
    '''
    Finds each instance that array a passes through a given value. Returns a dictionary where key up contains each pair of a[n-1] and a[n] where a[n-1] >= target >= a[n]. Similarly key down contains each pair of a[n-1] and a[n] where a[n-1] <= target <= a[n]. The target value need not be specified by the user, instead target can be set to a fraction, and the FracOf value can be set to min or max, which will result in the target value being the specified fraction of the maximum or minimum value in the array
    Arguements:

    - *a* The input array
    - *target* the threshold value, or the desired fraction of the minimum/maximum of the array to be used as the threshold value
    - *FracOf* Can be None if the target is the actual desired value. Otherwise it can be max or min, and the threshold value will be calculated as target*max(a) or target*min(a)
    - *NoneMax* If there are no instances of the threshold value being crossed, either an empty list will be returned if NoneMax is Empty, or the end of the array is returned if NoneMax is Last
    '''

    a      = np.array(a)  #Array with values at time n
    ar     = np.roll(a,1) #Array with values at time n-1
    ar[0]  = 0.

    if FracOf is not None:
        if FracOf.lower() == "max":
            amax = np.amax(a)
            target = target*amax
            
        if FracOf.lower() == "min":
            amin = np.amin(a)
            target = target*amin


    #return all a[n] for which a[n-1] >= target >= a[n] 
    doc = np.nonzero((target <= ar) & (a <= target))[0] 
    #return all a[n] for which a[n-1] <= target <= a[n]
    upc = np.nonzero((target >= ar) & (a >= target))[0]  
    if doc.shape[0] == 0:
        if NoneMax.lower() == "empty":
            doc = []
        elif NoneMax.lower() == "last":
            lst = np.shape(a)[0]-1
            doc = [[lst,lst]]
    else:
        docout = []
        for d in doc:
            docout.append([d-1,d])
        doc = docout #insert index of a[n-1] for each a[n]
        
    if upc.shape[0] == 0:
        if NoneMax.lower() == "empty":
            upc = []
        elif NoneMax.lower() == "last":
            lst = np.shape(a)[0]-1
            upc = [[lst,lst]]
    else:
        upcout = []
        for u in upc:
            upcout.append([u-1,u])
        upc = upcout #insert index of a[n-1] for each a[n]

    return{'up':upc,'down':doc,'target':target}
#1}}}
#def dict2sorted{{{1
def dict2sorted(dictionary,index,other=0.9):
    '''
    Convert dictionary where a key corresponds to a list into a sorted list with each list item being a key and the value. Useful for sorting the dictionaries returned by get_malloc_log and get_log_summary and plotting the results in a bar or pie chart.  
    Arguements:

    - *dictionary* The dictionary to be converted ino a sorted list
    - *index* index of the list contained in the dictionary to be sorted. For example if the dictionary was produced by get_malloc_log, index = 2 will sort the items by their memory size.
    - *other* used to trim the number of named entries. If a get_malloc_log dictionary is being processed with index = 2, setting other = 0.9 means that functions that take 90% of the memory will keep their name and value, and all smaller names and values will simply be collected into a category named other.
    '''
    
    fullList = [[value,key] for key,value in dictionary.iteritems()]
    #Sort
    fullList = sorted(fullList,key=lambda tup: tup[0][index],reverse=True)
    #Get totals
    total = 0.0
    for l in fullList:
        total += float(l[0][index])

    #Trim 
    ptotal = 0.0
    trimDict = {}
    for l in fullList:
        name = l[-1]
        val  = float(l[0][index])
        frc  = val/total
        ptotal += val
        if ptotal/total >= other:
            otherVal = total-ptotal
            otherFrc = 1.0-ptotal/total
            trimDict[name]    = [val,frc]
            trimDict['other'] = [otherVal,otherFrc]
            break
        else:
            trimDict[name]    = [val,frc]

    
    return trimDict


#1}}}
#def get_malloc_log{{{1
def get_malloc_log(Dir,Name):
    '''
    Parse a file containing the results of petsc -malloc_log. See petsc documentation for more details
    Arguements:

    - *Dir* Job working directory
    - *Name* Name of the malloc_log file
    '''
    malloc_info = {}
    with open(os.path.join(Dir,Name)) as f:
        for i,line in enumerate(f):
            sp = line.split()
            if i == 0:
                indmm = sp.index("PetscMalloc()ed")+1
                indrm = sp.index("process")+1
                malloc_max = int(sp[indmm])
                RSS_max    = int(sp[indrm])
            if len(sp) == 4:
                #strip [ and ] from rank and convert to int
                rank   = int(sp[0].translate(string.maketrans("", "", ), "[]"))
                count  = int(sp[1])
                length = int(sp[2])
                malloc_info[sp[3]] = [rank,count,length]

    return {"malloc_info":malloc_info,"malloc_max":malloc_max,"RSS_max":RSS_max}
#1}}}
#def get_log_summary{{{1
def get_log_summary(Dir,Name):
    '''
    Parse a file containing the results of petsc -log_summary. See petsc documentation for more details
    Arguements:

    - *Dir* Job working directory
    - *Name* Name of the malloc_log file
    '''
    dictOut = {}
    log_summary = {}
    with open(os.path.join(Dir,Name)) as f:
        inStage   = False
        inSummary = False
        cstage = 0
        for i,line in enumerate(f):
            #Identify end of stage
            if inStage and '-'*100 in line:
                inStage = False
                cstage += 1
            #Identify end of Summary
            if inSummary and "Flop counting convention" in line:
                inSummary = False


            if inStage:
                sp = line.split()
                if len(sp) != 0:
                    log_summary[sp[0]] = [sp[1],sp[3],sp[5]]

            if inSummary:
                sp = line.split(':')
                if len(sp) == 2:
                    name = sp[0]
                    val  = sp[1].split()[0]
                    dictOut[name] = val

            #Idetify start of summary
            if "PETSc Performance Summary" in line:
                inSummary = True
            #Identify start of stage
            if "--- Event Stage" in line and str(cstage) in line:
                inStage = True

    dictOut['log_summary'] = log_summary
    return dictOut

#def get_solver_stats{{{1
def get_solver_stats(Dir,Name,PKfile,parallel=True,rtimes="all",silent=False):
    '''
    This function parses FEAP's output files to extract the convergence behavior of the solver. It relies on the structure provided by the *solver_stats*, *TimeStep* and *NonlinearIteration* classes to store and organize all relevant informations.
    Arguements:
    
    - *Dir* directory containing the Ofile to parse
    - *Name* name of the Job that generated this simulation
    - *PKfile* name of a pickle file where the convergence infos will be stored for fast later retrieval.
    - *parallel* flags indicating if the simulation was run with parfeap in which case '_0001' has to be added to the Job name to get the Ofile name.
    - *rtimes* flag indicating which time steps will be retrieved. It can be a list of time steps to get specific time dependent data or it can be set as 'all' in which case the convergence at everytime steps will be extracted.
    - *silent* flag to allow this function to run silently, this is useful to extract the convergence data of multiple simulations run with a FEAPJob object.

    .. todo::
        Add fields
    '''
    if parallel:
        Ofile = os.path.join(Dir,"O"+Name+"_0001")
        Lfile = os.path.join(Dir,"L"+Name+"_0001")
    else:
        Ofile = os.path.join(Dir,"O"+Name)
        Lfile = os.path.join(Dir,"L"+Name)
    
    if isinstance(rtimes,basestring):
        if rtimes.lower() == "all":
            RT = True
        else:
            RT = False

    #Create solver stats object, add name and zeroeth time step
    SS = solver_stats()
    SS.Name = Name
    t0 = TimeStep(0.0)
    SS.TimeSteps.append(t0)

    #Parse Log file
    CPUTime   = 0.0
    Tang      = 0
    Form      = 0
    FinalStep = 0
    with open(Lfile) as f:
        begin = 0
        end   = 0
        for j,line in enumerate(f):
            sp = line.split()
            if "Step" in sp:
                begin = j+2
            if (begin != 0) and not sp:
                break
            elif (begin != 0) and (j >= begin) and sp:
                Tang += int(sp[1])
                Form += int(sp[2])
                CPUTime = float(sp[-2])
                FinalStep += 1

    SS.CPUTime   = CPUTime
    SS.Form      = Form
    SS.Tang      = Tang
    SS.FinalStep = FinalStep

    tsind = 0
    tsct  = 0
    lsflag= False
    lp    = ""
    #Parse output file
    with open(Ofile) as f:
        for line in f:
            #New time step
            if "   Computing solution at time" in line:
                SS.numTimeSteps += 1
                tsct += 1
                if isinstance(rtimes,list):
                    if tsct in rtimes:
                        RT = True
                    else:
                        RT = False
                nsind = -1 
                NL0   = True
                sp = line.split()
                st = float(sp[4][0:-1])
                if RT:
                    tsind += 1 
                    ts = TimeStep(st)
                    SS.TimeSteps.append(ts)
            elif "Maximum memory usage:" in line:
                sp = line.split()
                SS.MaxMemoryUsage = sp[3]
            #New nonlinear iteration
            elif "Command" in line and "utan" in line:
                if "LSBA" in line: 
                    lsflag  = True
                    lsstart = True
                else:
                    lsflag = False

                nsind += 1
                SS.numNonlinearIterations += 1
                if RT:
                    SS.TimeSteps[tsind].numNonlinearIterations += 1
                    NI     = NonlinearIteration()
                    SS.TimeSteps[tsind].NonlinearIterations.append(NI) 
            elif "Norm of step" in line:
                if RT:
                    sp = line.split()
                    SS.TimeSteps[tsind].StepNorms.append(float(sp[4])) 
            elif "Line Search Succesful" in line:
                if RT:
                    sp = lp.split()
                    gnind = sp.index("gnorm")+2
                    SS.TimeSteps[tsind].NonlinearResidual[-1] = float(sp[gnind].replace(',',''))
            #For first nonlinear iteration in the time step
            elif "   Residual norm =" in line and NL0:
                if RT:
                    sp = line.split()
                    SS.TimeSteps[tsind].NonlinearResidual.append(float(sp[3]))
                    NL0 = False
            #For all other nonlinear iterations
            elif "Global residual norm =" in line:
                if RT:
                    sp = line.split()
                    try:
                        nl = float(sp[4])
                    except:
                        sp2 = sp[4].split("+")
                        nl  = float(sp2[0]+"E100")
                    SS.TimeSteps[tsind].NonlinearResidual.append(nl)
            #New linear iterations
            elif "KSP Residual Norm" in line:
                if RT:
                    sp = line.split()
                    spnind = sp.index("Norm")
                    if not sp[0] == "0":
                        SS.numLinearIterations += 1
                        SS.TimeSteps[tsind].numLinearIterations += 1
                        SS.TimeSteps[tsind].NonlinearIterations[nsind].numLinearIterations += 1
                    try:
                        residual = float(sp[spnind+1])
                    except ValueError:
                        if 'Infinity' in sp[spnind+1]:
                            print sp[spnind+1]
                            residual = float('inf')
                    SS.TimeSteps[tsind].NonlinearIterations[nsind].LinearResidual.append(residual)
                    if r"max/min" in sp:
                        spind = sp.index(r"max/min")
                        SS.TimeSteps[tsind].NonlinearIterations[nsind].SVMaxOnMin.append(float(sp[spind+1]))

            lp = line

    with open(PKfile,'wb') as out:
        pickle.dump(SS,out,pickle.HIGHEST_PROTOCOL)
    
    if not silent:
        print "Solver stats collected for "+Name

    return SS
#1}}}
#def load_pickle{{{1
def load_pickle(PKFile):
    '''
    Returns an unpickled structure.
    '''
    
    with open(PKFile,'rb') as inp:
        Unpick = pickle.load(inp)

    return Unpick
#1}}}
#class solver_stats{{{1
class solver_stats:
    #def __init__{{{2
    def __init__(self):
        
        self.numTimeSteps           = 0
        '''Number of time steps during the simulation.'''
        self.numNonlinearIterations = 0
        '''Total number of nonlinear iterations during the simulation.'''
        self.numLinearIterations    = 0
        '''Total number of linear iterations during the simulation.'''
        self.TimeSteps              = []
        '''List of TimeStep objects containing more detailed informations about individual time steps.'''
        self.MaxMemoryUsage         = 0.
        '''Maximum memory used during the simulation.'''
        self.Name                   = ""
        '''Name of the simulation analyzed'''
    #2}}}
    #def __str__{{{
    def __str__(self):
        msg = ''
        msg += 'Simulation: %s\n' %(self.Name)
        msg += '   Time steps: %i\n' %(self.numTimeSteps)
        msg += '   Total nonlinear iterations: %i\n' %(self.numNonlinearIterations)
        msg += '   Total linear iterations: %i\n' %(self.numLinearIterations)
        msg += '   Max memory usage: %s' %(str(self.MaxMemoryUsage))
        return msg
#1}}}
#class TimeStep{{{1
class TimeStep:
    #def __init__{{{2
    def __init__(self,ST):
        self.SolutionTime             = ST
        '''Current solution time in seconds.'''
        self.NonlinearResidual        = []
        '''Norm of the nonlinear residual at each nonlinear iteration of the current time step.'''
        self.numNonlinearIterations   = 0
        '''Number of nonlinear iterations in the current time step.'''
        self.numLinearIterations      = 0
        '''Total number of linear iterations needed in the current time step (sum of the linear iterations at each nonlinear iterations).'''
        self.NonlinearIterations      = []
        '''List of NonlinearIteration objects describing the nonlinear iterations of the current time step.'''
        self.StepNorms                = []
        '''List of the norm of the newton step'''
    #2}}}
    #def __str__{{{2
    def __str__(self):
        msg = ''
        msg += 'Solution time : %s\n' %(str(self.SolutionTime))
        msg += 'Nonlinear iterations: %i\n' %(self.numNonlinearIterations)
        msg += 'Linear iterations: %i' %(self.numLinearIterations)
        return msg
    #2}}}
#1}}}
#class NonlinearIteration{{{1
class NonlinearIteration:
    #def __init__{{{2
    def __init__(self):
        self.numLinearIterations = 0
        '''Number of iterations the linear solver needed to converge during the current nonlinear iteration.'''
        self.LinearResidual      = []
        '''Norm of the linear residual at each linear iterations during the current nonlinear iteration.'''
        self.SVMaxOnMin          = []
        '''Ratio of the max to min singular values of the preconditioned matrix. Activated with command line option -ksp_monitor_singular_value_Ofile'''
        self.LineSearch          = []
        '''Currently not implemented, could be used to show the results of each line search iteration'''
    #2}}}
    #def __str__{{{2
    def __str__(self):
        msg = ''
        msg += 'Linear iterations: %i\n' %(self.numLinearIterations)
        if self.numLinearIterations > 0:
            msg += 'Residual drop: %s --> %s' %(self.LinearResidual[0],self.LinearResidual[-1])
        return msg
    #}}}2
#1}}}
#class Bunch{{{1
class Bunch(object):
    '''This object just stores a bunch of stuff in a dictionary. Classy way to program ; ).'''
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
#1}}}
#class SubJob{{{1
class SubJob(object):
    def __init__(self, Value=None,Name=None,ID=None,APs=[None,None],AFs=None,PLSC="-ksp_type preonly -pc_type lu -pc_factor_mat_solver_package umfpack -ksp_diagonal_scale -malloc_log mlog",run=True,DontCombine=None,Rfile=None,NURBS='',gmopt=None):
        self.Value = Value
        ''' The value to be substituted for the main parameter key'''
        self.Name  = Name
        '''Name of the job.'''
        self.APs   = APs
        '''Additional Parameters to be replace in the input and solver files by the associated value.'''
        self.AFs   = AFs
        '''Additional Files (mesh files mainly) to be copied in the feap job directory.'''
        self.PLSC  = PLSC
        '''Command line options to be passed to feap. Typically these are petsc options which serves as feaps linear solver'''
        self.run   = run
        '''flag indicating if the job should be run or not. Setting run to false is used when a script is run multiple times, perhaps if only certain jobs need to be re run. This can also be used to change different post processing options without re running feap'''
        self.Rfile = Rfile
        '''Flag for the restart file.'''
        self.NURBS = NURBS
        '''Flag for NURBS mesh file.'''
        self.gmopt = gmopt
        '''Options to be passed to gmsh

        .. todo::
            I suppose this is obsolete due to the mesh class?
        '''
        if ID is None:
            self.ID = Name
        else:
            self.ID = ID
            '''Job ID, can be different from the job name if multiple jobs have the same name.'''
        if DontCombine is not None:
            self.DC = [[self.ID,d] for d in DontCombine]
            '''List of names of parameters that the current job should not be combined with. For example if the current parameter is named "A" and DC = ["B","C"] then after all job permutations are computed, all jobs containing "A" and "B" and "A" and "C" will be dropped from the list of jobs'''
        else:
            self.DC = None
#1}}}
#def mkdir_p{{{1
def mkdir_p(path,run):
    '''
    This function is used to create or clear the job directories used by the FEAPJob class. If the directory does not exist, it is created. If the directory exists and the job is set to run, it cleans the directory. Otherwise, nothing is done.
    Arguements:
    
    - *path* the directory to be created or cleared
    - *run* flag to run the job that uses this directory
    '''
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc: # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else: raise
    else:
        if run:
            fils = glob.glob(os.path.join(path,"*"))
            [os.remove(f) if not os.path.isdir(f) else shu.rmtree(f) for f in fils]
#1}}}
#def mvfil {{{1
def mvfil(fil,directory,rename=None):
    '''
    Move and optionally rename a file. Used internally by FEAPJob.CreateJobs()
    Arguements:
    
    - *fil* The original file
    - *directory* The directory to move the file to
    - *rename* If none the file name is unchanged, otherwise rename is the new file name
    '''
    if fil is None:
        return
    filn = os.path.basename(fil)
    dst  = os.path.join(directory,filn)
    if not os.path.exists(dst):
        shu.copyfile(fil,dst)
    if rename is not None:
        dstr = os.path.join(directory,rename)
        os.rename(dst,dstr)
#1}}}
#def ppt_dummy{{{1
def ppt_dummy(*args,**kwargs):
    pass
#1}}}
#def ziptrans{{{1
def ziptrans(lst):
    if isinstance(lst[0],list):
        return zip(*lst)
    else:
        return [(l,) for l in lst]
#def ppfLayout{{{1
def ppfLayout(primaryFunc,newname,SKwargs):

    iv_psot_dict = {'function':iv_psot,
                    'primaryAttrs':["PVDs","PSOT"+newname],
                    'IsOldAttrs':["PVDs","PSOT"+newname],
                    'secondaryFunc':pvcsvin,
                    'secondaryAttrs':["PSOT"+newname],
                    'ext':'csv',
                    'secondaryOpargs':{}}

    view_field_dict = {'function':view_field,
                       'primaryAttrs':["PVDs","PNG"+newname],
                       'IsOldAttrs':["PVDs","PNG"+newname],
                       'secondaryFunc':ppt_dummy,
                       'secondaryAttrs':["PNG"+newname],
                       'ext':'png',
                       'secondaryOpargs':{}}

    get_solver_stats_dict = {'function':get_solver_stats,
                             'primaryAttrs':[["Jobs",0],["Jobs",1],"PKs"+newname],
                             'IsOldAttrs':["PVDs","PKs"+newname],
                             'secondaryFunc':load_pickle,
                             'secondaryAttrs':["PKs"+newname],
                             'ext':'pk',
                             'secondaryOpargs':{}}

    GetArrayRange_dict = {'function':GetArrayRange,
                          'primaryAttrs':["PVDs"],
                          'IsOldAttrs':["PVDs","PVDs"],
                          'secondaryFunc':ppt_dummy,
                          'secondaryAttrs':["PVDs"],
                          'ext':'',
                          'secondaryOpargs':{},
                          'Force':True}

    pol_dict = {'function':pol,
                'primaryAttrs':["PVDs","POL"+newname],
                'IsOldAttrs':["PVDs","POL"+newname],
                'secondaryFunc':pvcsvin,
                'secondaryAttrs':["POL"+newname],
                'ext':'csv',
                'secondaryOpargs':{}}

    funcdict = {'iv_psot'         :iv_psot_dict,
                'view_field'      :view_field_dict,
                'get_solver_stats':get_solver_stats_dict,
                'GetArrayRange'   :GetArrayRange_dict,
                'pol'             :pol_dict}


    return funcdict[primaryFunc]
#class FEAPJob{{{1
class FEAPJob:
    #def __init__{{{2
    def __init__(self,FEAP,home,name,Bin,Bso,mail=None,pponly=False,Parallel=True,Bgeo=None,GMSH=None,Pfdir=''):
        
        #Check if FEAP has executable premission (**x**x**x)
        if os.access(FEAP, os.X_OK):
            self.FEAP     = FEAP
        #Check if FEAP is an empty string
        elif not FEAP:
            FEAP = [val for key, val in os.environ.iteritems() if 'FEAPHOME' in key]
            if len(FEAP) == 0:
                print 'No feap variable detected'
            elif len(FEAP) > 0:
                FEAP = FEAP[0]
        elif os.path.isdir(FEAP):
            if Pfdir:
                Pfdir = os.path.join(Pfdir,'feap')
                self.FEAP = os.path.join(FEAP,Pfdir)
            else:
                self.FEAP = os.path.join(FEAP,'main/feap')
        else:
                    if Parallel:
                        self.FEAP = os.path.join(FEAP,'parfeap/feap')
                    else:
                        self.FEAP = os.path.join(FEAP,'main/feap')
        ''' 
        FEAP executable, if not set it will look up your path for an environment variable containing FEAPHOME and use the default installation path to work from there.
        '''
        if GMSH:
            self.GMSH     = GMSH
        else:
            self.GMSH     = distutils.spawn.find_executable('gmsh')
        ''' 
        gmsh executable

        .. todo::
            Perhaps this is now obsolete due to the mesh class?
        '''
        self.name     = name
        ''' 
        Name of the job collection. Used as the name of the top level directory for the job data
        '''
        self.home     = home
        ''' 
        Home directory for job collection
        '''
        self.mail     = mail
        ''' 
        Email address to recieve alerts that jobs are complete.
        '''
        self.Bin      = Bin
        ''' 
        Base input file to create job input files
        '''
        self.Bso      = Bso
        ''' 
        Base solve file to create job solve files

        .. todo::
            should probably be optional
        '''
        self.Bgeo     = Bgeo
        ''' 
        Base geo file to create job geo files

        .. todo::
            Luc? Did I add this or did you?
            I didn't do that one, but I don't know how we should integrate the mesh class to the FEAPJob class?
        '''
        self.MPs      = []
        ''' 
        List of main parameter information
        '''
        self.pponly   = pponly
        ''' 
        Flag indicating that all jobs have been created and run, and feapjob is just being used to access post processing data
        '''
        self.Parallel = Parallel
        '''
        Flag indicating use of parallel feap

        .. todo::
            Implement multiprocessor case
        '''
        #Sets TimeInfo Attribute to empty
        self.TimeInfo=[]
    #2}}}
    #def mail_alert{{{2
    def mail_alert(self,i,name):
        '''
        Sends an email to the address specified in self.mail after each job is completed. Used internally by FEAPJob.RunJobs
        '''
        t    = self.nJob
        From = "runfeapjob@gmail.com"
        Pasw = "rfjpython"
        Recp = self.mail 
        Date = time.ctime(time.time())
        Subj = "FEAP Job Completed"
        Text = "Job "+name+" completed."+str(i)+"/"+str(t)+" jobs are now complete."
        mMes = ('From: %s\nTo: %s\nDate: %s\nSubject: %s\n%s\n' % (From,Recp,Date,Subj,Text))
        s = smtplib.SMTP('smtp.gmail.com',587)
        s.ehlo()
        s.starttls()
        s.ehlo()
        s.login(From,Pasw)
        rCod = s.sendmail(From,Recp,mMes)

        s.quit()

        if rCod: print 'Error Sending Message'
    #2}}}
    #def AddMainParameter{{{2
    def AddMainParameter(self,Name,Info):
        '''
        Function for user to add a main parameter to the list of main parameters.
        '''
        self.MPs.append(Bunch(name=Name,info=Info))
    #2}}}
    #def ReplaceMainParameter{{{2
    def ReplaceMainParameter(self,Name,Info):
        '''
        Function for user to replace the info belonging to a main parameter
        '''
        newm = Bunch(name=Name,info=Info)
        self.MPs = [newm if m.name == Name else m for m in self.MPs]
    #2}}}
    #def CreateJobs{{{2
    def CreateJobs(self,Restart=False,Bin=None,Bso=None):
        '''
        Function for creating everything needed for the feap jobs. The information contained in the main parameters are unpacked and permuted to create individual jobs. The do not combine list is used to eliminate jobs which contain two parameters which are not to be combined. The names of the parameters are used to create a directory tree as well as the job name. For example if a job contains parameters A, B and C the job will be names ABC and will reside in directory self.home/self.name/A/B/C. For each job, parameters are substituted into base files based on the user supplied keys and values. The resulting job files, along with any other associated files, are moved to the appropriate directory. Finally, the Jobs attribute is set up. This is a list of lists containing all relevant job information: the job directories, names, command line options, and run flags. The Jobs attribute can then be used to iterate over all jobs. This is done by several FEAPJob functions
        '''
        #Create top level job directory
        tdir = os.path.join(self.home,self.name)

        #permute parameters
        names  = [[i.Name  for i in j.info] for j in self.MPs]
        mainv  = [[[j.name,i.Value] for i in j.info] for j in self.MPs]
        APs    = [[i.APs   for i in j.info] for j in self.MPs]
        AFs    = [[i.AFs   for i in j.info] for j in self.MPs]
        plsc   = [[i.PLSC  for i in j.info] for j in self.MPs]
        run    = [[i.run   for i in j.info] for j in self.MPs]
        DC     = [[i.DC    for i in j.info] for j in self.MPs]
        Rfiles = [[i.Rfile for i in j.info] for j in self.MPs]
        NURBS  = [[i.NURBS for i in j.info] for j in self.MPs]
        gmopt  = [[i.gmopt for i in j.info] for j in self.MPs]
        
        names  = list(itt.product(*names))
        mainv  = list(itt.product(*mainv))
        APs    = list(itt.product(*APs))
        AFs    = list(itt.product(*AFs))
        plsc   = list(itt.product(*plsc))
        run    = list(itt.product(*run))
        Rfiles = list(itt.product(*Rfiles))
        NURBS  = list(itt.product(*NURBS))
        gmopt  = list(itt.product(*gmopt))
        
        #Get names indexes of combinations to bypass
        DelInd = []
        i = -1
        DelNames = []
        for subjob in DC:
            for dname in subjob:
                if dname is not None:
                    DelNames += [bn for bn in dname]
        
        for i,name in enumerate(names):
            delck = [bn[0] in name and bn[1] in name for bn in DelNames]
            if np.any(delck):
                DelInd.append(i)
        
        names  = [item for i,item in enumerate(names)  if i not in DelInd]
        mainv  = [item for i,item in enumerate(mainv)  if i not in DelInd]
        APs    = [item for i,item in enumerate(APs)    if i not in DelInd]
        AFs    = [item for i,item in enumerate(AFs)    if i not in DelInd]
        plsc   = [item for i,item in enumerate(plsc)   if i not in DelInd]
        run    = [item for i,item in enumerate(run)    if i not in DelInd]
        Rfiles = [item for i,item in enumerate(Rfiles) if i not in DelInd]
        NURBS  = [item for i,item in enumerate(NURBS)  if i not in DelInd]
        gmopt  = [item for i,item in enumerate(gmopt)  if i not in DelInd]
        
        APs    = [[i if type(s[0]) == list  else s for s in a for i in s] for a in APs]
        AFs    = [[item for sublist in a if sublist is not None for item in sublist] for a in AFs]
        Rfiles = [[item for item in a if item is not None] for a in Rfiles]
        
        #Job Directories and names
        dirs    = [os.path.join(tdir,*n) for n in names]
        names   = [''.join(n) for n in names]
        pls     = [n[0] for n in plsc]
        gmopt   = [n[0] for n in gmopt]
        run     = [False if False in n else True for n in run]
        lst=[]
        for tupl in NURBS:
            #Check if the one of the MPs use a NURBS mesh
            if any(tupl):
                chck=False
                for mesh_name in tupl:
                    if (mesh_name and chck):
                        print 'Too many NURBS meshes in for a single MPs combination!'
                    elif(mesh_name and not chck):
                        lst.append([mesh_name])
                        chck=True
            else:
                lst.append([''])
        NURBS = lst
        
        #For restarted jobs
        if Restart:
            if Bin is not None:
                self.Bin = Bin
            if Bso is not None:
                self.Bso = Bso
            #Add restart directory
            dirs   = [os.path.join(d,"Restart") for d in dirs]
        
        pln = [None for x in dirs]
        if not self.pponly:
            if self.Bgeo: 
                BGEO = read_file(self.Bgeo)
                #list of substitution keywords for base geo file
                sublist = [[BGEO,list(m)+a] for m,a in itt.izip(mainv,APs)]
                #substitute keywords
                Jgeo = sub_files(sublist,1,";")
            else:
                Jgeo = [None for x in names]
            #create job directories, run gmsh, and write mesh files
            ic = 0
            for d,n,a,j,r,R,g in itt.izip(dirs,names,AFs,Jgeo,run,Rfiles,gmopt): 
                mkdir_p(d,r)
                if self.Bgeo:
                    jmn  = os.path.join(d,n)
                    jgeo = jmn+".geo"
                    jmsh = jmn+".msh"
                    jfmsh= jmn+"mesh"
                    write_file(jgeo,j[0])
                    run_gmsh(self.GMSH,jgeo,jmsh,g)
                    pln[ic] = gmsh2feap(jmsh,jfmsh,[2],[6],numid=[1])
                    ic+=1
            
            ic = 0
            mainvr=[]
            APr=[]
            #Find and replace special keywords
            for m,a,n in itt.izip(mainv,APs,names):
                mr = []
                ar = []
                for pair in m:
                    if pair[1] is not None:
                        if "GMNUMNP" in pair[1]:
                            newpair = [pair[0],pair[1].replace("GMNUMNP",str(pln[ic]))]
                        elif "OWNNAME" in pair[1]:
                            newpair = [pair[0],pair[1].replace("OWNNAME",n)]
                        else:
                            newpair = pair
                        mr.append(newpair)
                for pair in a:
                    if pair[1] is not None:
                        if "GMNUMNP" in pair[1]:
                            newpair = [pair[0],pair[1].replace("GMNUMNP",str(pln[ic]))]
                        elif "OWNNAME" in pair[1]:
                            newpair = [pair[0],pair[1].replace("OWNNAME",n)]
                        else:
                            newpair = pair
                        ar.append(newpair)
                        #print pair
                        #print
                mainvr.append(mr)
                APr.append(ar)
                ic +=1
            
            APs = APr
            mainv = mainvr
            
            
            #Read base files
            BI = read_file(self.Bin)
            BS = read_file(self.Bso)
            #create dict of substitution keywords for each job
            sublist = []
            for mp,ap in itt.izip(mainv,APs):
                jobdict = {}
                for m in mp:
                    jobdict[m[0]] = m[1]
                for a in ap:
                    jobdict[a[0]] = a[1]
                
                sublist.append(jobdict)
            
            
            #substitute keywords
            JFIL = [sub_files([BI,BS],sd) for sd in sublist]
            
            #create job directories, write I and .solve files, and move associated files
            for d,n,a,j,r,R,g in itt.izip(dirs,names,AFs,JFIL,run,Rfiles,gmopt): 
                write_file(os.path.join(d,'I'+n),j[0])
                if self.Parallel:
                    write_file(os.path.join(d,'solve.'+n),j[1])
                
                [mvfil(fil,d) for fil in a]
                #Move then rename restart files
                if Restart:
                    rn = "R"+n
                    if self.Parallel:
                        rn = "R"+n+"_0001"
                    mvfil(R[0],d,rename=rn)
        
        self.Jobs  = zip(dirs,names,pls,run)
        self.NURBS = NURBS
        self.Jobs  = [list(a) for a in self.Jobs]
        self.nJob  = len(names)
        self.names = names
    #2}}}
    #def QuickJob{{{2
    def QuickJob(self,Subs=[],Name="Job",ID=None,APs=[None,None],AFs=[None],PLSC="-ksp_type preonly -pc_type lu -pc_factor_mat_solver_package umfpack -ksp_diagonal_scale -malloc_log mlog",run=True,Rfile=None,NURBS=''):
        '''
        Function for creating everything needed for a single feap job. Additionally, the job is run and the pvds are generated. A single job is created based on a list of substitutions with no permutation. In other words, this is like setting up a single subjob. 
        '''
        #Create top level job directory
        tdir = os.path.join(self.home,self.name)

        #Job Directories and names
        jdir    = os.path.join(tdir,Name)
        mkdir_p(jdir,run)
        
        
        #Read base files
        BI = read_file(self.Bin)
        BS = read_file(self.Bso)
        #create dict of substitution keywords for each job
        jobdict = {}
        for s in Subs:
            jobdict[s[0]] = s[1]
            
        #substitute keywords
        JFIL = sub_files([BI,BS],jobdict)
        
        #create job directories, write I and .solve files, and move associated files
        write_file(os.path.join(jdir,'I'+Name),JFIL[0])
        if self.Parallel:
            write_file(os.path.join(jdir,'solve.'+Name),JFIL[1])
        
        [mvfil(fil,jdir) for fil in AFs]
        
        if not self.Parallel:
            p=""

        RunFEAPJob(jdir,Name,self.FEAP,PETScLinearSolverOptions=PLSC,parfeap=self.Parallel,NURBS=NURBS)

        pvdgen(jdir,Name+".pvd")

        self.PVDs = os.path.join(jdir,Name)+'.pvd'

    #2}}}
    #def RunJobs{{{2
    def RunJobs(self):
        '''
        Runs all jobs by iterating over all jobs and calling RunFEAPJob. If the run flag for a job is false, nothing is done for that particular job
        '''
        Jobs  = self.Jobs
        FEAP  = self.FEAP
        mail  = self.mail
        NURBS = self.NURBS
        JobsN = [j+n for j,n in zip(Jobs,NURBS)]
        
        i    = 1
        for d,n,p,r,N in JobsN:
            if r:
                if not self.Parallel:
                    p=""
                RunFEAPJob(d,n,FEAP,PETScLinearSolverOptions=p,parfeap=self.Parallel,NURBS=N)
                if mail is not None:
                    self.mail_alert(i,n)
            i += 1
    #2}}}
    #def CreatePVDs{{{2
    def CreatePVDs(self,force=False):
        '''
        Creates a pvd collection from the vtu data for each job. By default, the pvd is created only if it does not exist, or if it is older than the job's output file as determined by prepost_tools.IsOld. If force is set to True, then the pvd is generated regardless of the age of the files. The PVDs attribute is created containing a list of the pvd file names for each job. The PVDs attribute is separate from the Jobs attribute, but both of these lists are in the same order *Potential Pitfall?* 
        '''
        #create pvd filenames
        Jobs = self.Jobs
        [pvdgen(d,n+".pvd") for d,n,p,r in Jobs if True in [r,force,IsOld(os.path.join(d,"O"+n),os.path.join(d,n+".pvd"))]]

        self.PVDs = [os.path.join(d,n)+'.pvd' for d,n,p,r in Jobs]

        #Create Time Info Attribute
        for pvd in self.PVDs:
            with open(pvd,'r') as f:
               lines = f.read().splitlines()
            nlines=len(lines)
            RealTime=[]
            TimeStep=[]
            for l in lines:
               splt = l.split('timestep=')
               if len(splt)==2:
                  splt=splt[1].split('"')
                  RealTime.append(float(splt[1]))
                  TimeStep.append(int(splt[5].split('_')[1].split('.vtu')[0])-1000000)
            self.TimeInfo.append([TimeStep,RealTime])
    #2}}}
    #def CreatePPF{{{2
    def CreatePPF(self,ext,pname='',ename='',NewDir=None):
        '''
        Creates an attribute of the FEAPJob containing a list of files, typically files that will be given to a post processing function as output. The files created are dir/pname+jobname+ename+'.'+ext. The attribute name is ext + pname + ename, with these variables described below. Optionally, NewDir can be specified if the files are to reside in another location than the job directory.
        Arguements:

        - *ext* The extension of the file
        - *pname* Name to appear before the jobname
        - *ename* name to appear after jobname
        - *NewDir* Optionally specified new directory for files. If NewDir is None (default) then the files will be put in their job's directory. For example, the results of the paraview filter are typically kept in the directory of that job, but it is convienient to have all png files produced by view_field in a single directory.

        .. todo::
            I think that a this function should also run the actual desired filter. Then the user would supply the funtion to be run and everything else is done under the hood. The idea is to eliminate the need for the user to do something like psot = [ppt.iv_psot(a,b,ivflg=True,retcsv=True) if ppt.IsOld(a,b) else ppt.pvcsvin(b) for a,b in zip(T1.PVDs,T1.csvPSOT)]. Instead, the user would just say T1.CreatePPF(iv_psot,params...) where params are optional arguements for iv_psot. Then all iteration and IsOld stuff would be handled under the hood.
        '''
        if NewDir is None:
            setattr(self,ext+pname+ename,[os.path.join(d,pname+n+ename+'.'+ext) for d,n,p,r in self.Jobs])
        else:
            setattr(self,ext+pname+ename,[os.path.join(NewDir,pname+n+ename+'.'+ext) for d,n,p,r in self.Jobs])
    #2}}}
    #def pullAttrs{{{2
    def pullAttrs(self,Attrs,primaryAttrs):

        if Attrs is None:
            Attrs = primaryAttrs
        else:
            for i,attr in enumerate(Attrs):
                Attrs[i] = getattr(self,attr)

            return Attrs
    #2}}}
    #def setupAttrs{{{2
    def setupAttrs(self,Attrs,ext,pname,ename,NewDir,jobList):

        indxs = [i for d,n,p,r,i in jobList]
        #create attributes in primaryAttrs if they do not exist
        for i,attr in enumerate(Attrs):
            #Determine attribute name and index
            if isinstance(attr, basestring):
                attrIndx = None
            else:
                attrIndx = attr[1]
                attr     = attr[0]

            #If attribute already exists, then extract it
            try:
                AttrsFull = getattr(self,attr)
                if len(AttrsFull) == len(self.Jobs): #This is terrible
                    if attrIndx is None:
                        Attrs[i]  = [a for ii,a in enumerate(AttrsFull) if ii in indxs]
                    else:
                        Attrs[i]  = [a[attrIndx] for ii,a in enumerate(AttrsFull) if ii in indxs]
                else:
                    if attrIndx is None:
                        Attrs[i]  = AttrsFull
                    else:
                        Attrs[i]  = [a[attrIndx] for a in AttrsFull]
            #If attribute does not exist, then create it
            except:
                if NewDir is None:
                    setattr(self,attr,[os.path.join(d,pname+n+ename+'.'+ext) for d,n,p,r,ind in jobList])
                else:
                    setattr(self,attr,[os.path.join(NewDir,pname+n+ename+'.'+ext) for d,n,p,r,ind in jobList])

                AttrsFull = getattr(self,attr)
                if attrIndx is None:
                    Attrs[i] = AttrsFull
                else:
                    Attrs[i]  = [a[attrIndx] for a in AttrsFull]


        return ziptrans(Attrs)
#2}}}
    #def CreatePPF2{{{2
    def CreatePPF2(self,primaryFunc,pname='',ename='',primaryOpargs={},NewDir=None,Force=False,AutoLayout=True,SKwargs={},jobContains=None):
        '''
        Applies a prepost_tools function to every job in the FEAPJob object. Typical usage would be to use AutoLayout=True, which lets ppt determine which attributes to extract or create to use as arguements for the function. Most post processing functions are expensive and therefore measures are taken to avoid repetitive calculations that are not needed. Thus if a function is computed only if its output file is older than that same functions input file. Otherwise, the desired information is simply loaded through functions such as pvcsvin or load_pickle
        Arguements:

        - *primaryFunc* String containing the name of the prepost_tools function to be applied
        - *primaryOpargs* Dictionary containing optional arguements for the primary function
        - *pname* Name to appear before the jobname
        - *ename* name to appear after jobname
        - *NewDir* Optionally specified new directory for files. If NewDir is None (default) then the files will be put in their job's directory. For example, the results of the paraview filter are typically kept in the directory of that job, but it is convienient to have all png files produced by view_field in a single directory.
        - *Force* Flag to disregard the results of IsOld and force computation of primary function
        - *AutoLayout* Flag to let prepost_tools automatically populate the SKwargs dictionary based off default values determined by the value of primaryFunc

        SKwargs contains information on which secondary function to use as well as the positional arguements for the primary function, the secondary function, and IsOld. These positional arguements are typically input and output files for prepost functions and thus will be different for each job. Therefore, they are stored as attributes of the FEAPJob instance. Note that for normal use at least one attribute will have to be created for the data created by the prepost function. This is done automatically. If AutoLayout = False then at a minimum SKwargs must contain the key primaryAttrs with an ordered list of strings corresponding to the names of the attributes to be used by primaryFunc.

        ..  todo::
            Handle collections of multiple files corresponding to a time step



        '''

        if jobContains is not None:
            jobList = []
            for i,j in enumerate(self.Jobs):
                if isinstance(jobContains,basestring):
                    if jobContains in j[1]:
                        jobList.append(j+[i])
                else:
                    if all([x in j[1] for x in jobContains]):
                        jobList.append(j+[i])

            if isinstance(jobContains,basestring):
                pname += jobContains
            else:
                pname += "".join(jobContains)

        else:
            jobList = [j+[i] for i,j in enumerate(self.Jobs)]

        names = [n for d,n,p,r,i in jobList]


        #Set default values for optional arguements based on primaryFunc
        if AutoLayout:
            AutoSKwargs     = ppfLayout(primaryFunc,pname+ename,SKwargs)
            primaryFunc     = AutoSKwargs['function']
            primaryAttrs    = AutoSKwargs['primaryAttrs']
            IsOldAttrs      = AutoSKwargs['IsOldAttrs']
            secondaryFunc   = AutoSKwargs['secondaryFunc']
            secondaryAttrs  = AutoSKwargs['secondaryAttrs']
            secondaryOpargs = AutoSKwargs['secondaryOpargs']
            ext             = AutoSKwargs['ext']
            try:
                Force       = AutoSKwargs['Force']
            except:
                pass
        else:
            try:
                primaryAttrs = SKwargs['primaryAttrs']
            except:
                print "Can not proceed with given arguements, AutoLayout=False and primaryAttrs not set in SKwargs" 
                return

            IsOldAttrs      = self.pullAttr(SKwargs.get('IsOldAttrs',None),primaryAttrs)
            secondaryFunc   = SKwargs.get('secondaryFunc',ppt_dummy)
            secondaryAttrs  = self.pullAttr(SKwargs.get('secondaryAttrs',None),primaryAttrs)
            secondaryOpargs = SKwargs.get('secondaryOpargs',{})

        #set up Attrs
        primaryAttrs   = self.setupAttrs(primaryAttrs,ext,pname,ename,NewDir,jobList)
        secondaryAttrs = self.setupAttrs(secondaryAttrs,ext,pname,ename,NewDir,jobList)
        IsOldAttrs     = self.setupAttrs(IsOldAttrs,ext,pname,ename,NewDir,jobList)

        #apply functions
        return [[primaryFunc(*pA,**primaryOpargs) if IsOld(*oA,Coerce=Force) else secondaryFunc(*sA,**secondaryOpargs),n] for pA,oA,sA,n in zip(primaryAttrs,IsOldAttrs,secondaryAttrs,names)]
    #2}}}
    #def CreatePPF3{{{2
    def CreatePPF3(self,FilterList,Fname='filter',pname='',ename='',NewDir=None,jobContains=None,Force=False,ViewTime='last'):
        '''
        Applies a prepost_tools function to every job in the FEAPJob object. Typical usage would be to use AutoLayout=True, which lets ppt determine which attributes to extract or create to use as arguements for the function. Most post processing functions are expensive and therefore measures are taken to avoid repetitive calculations that are not needed. Thus if a function is computed only if its output file is older than that same functions input file. Otherwise, the desired information is simply loaded through functions such as pvcsvin or load_pickle
        Arguements:

        - *FilterList* List of Filters to be applied
        - *pname* Name to appear before the jobname
        - *ename* name to appear after jobname
        - *NewDir* Optionally specified new directory for files. If NewDir is None (default) then the files will be put in their job's directory. For example, the results of the paraview filter are typically kept in the directory of that job, but it is convienient to have all png files produced by view_field in a single directory.
        - *jobContains* string used to filter the jobs on which the ParaView filters will be applied

        ..  todo::
            Handle collections of multiple files corresponding to a time step



        '''
        
        if jobContains is not None:
            jobList = []
            for i,j in enumerate(self.Jobs):
                if isinstance(jobContains,basestring):
                    if jobContains in j[1]:
                        jobList.append(j+[i])
                else:
                    if all([x in j[1] for x in jobContains]):
                        jobList.append(j+[i])

            if isinstance(jobContains,basestring):
                pname += jobContains
            else:
                pname += "".join(jobContains)

        else:
            jobList = [j+[i] for i,j in enumerate(self.Jobs)]
        
        names = [n for d,n,p,r,i in jobList]
        PVDid = [i for d,n,p,r,i in jobList]


        #set up Attrs
        LastFilter     = FilterList[-1]
        FilterDict     = GetFilterDict(LastFilter.Name)
        FileType       = FilterDict['FileType']
        self.CreatePPF(FileType,pname=pname,ename=ename,NewDir=NewDir)
        FilterAttr = getattr(self,FileType+pname+ename)
        PVDtoZip = [PVD for i, PVD in enumerate(self.PVDs) if (i in PVDid)]
        #apply functions
        return [[ApplyFilters(p,o,FilterList,ViewTime=ViewTime) if IsOld(p,o,Coerce=Force) else PPTLoad(o,FileType),n] for p,o,n in zip(PVDtoZip,FilterAttr,names)]
        #return [[ApplyFilters(p,o,FilterList) if IsOld(p,o,Coerce=Force) else PPTLoad(o,FileType),n] for p,o,n in zip(self.PVDs,FilterAttr,names)]
    #2}}}
    #def BackupResults{{{2
    def BackupResults(self,bnam):
        '''
        Creates a backup of results by tarring and zipping the job directory tree
        Arguements:

        - *bnam* Name of the archive
        '''
        home   = self.home
        name   = self.name
        rdir   = os.path.join(home,name)
        tarnam = os.path.join(home,bnam+".tar.gz")

        print "Compressing backup directory"
        tar  = ("tar -zcf "+tarnam+" "+rdir).split()
        proc = subprocess.Popen(tar,stdin=subprocess.PIPE)
        out, err = proc.communicate()

        gzip = ("gzip -f "+tarnam).split()
        proc = subprocess.Popen(gzip,stdin=subprocess.PIPE)
        out, err = proc.communicate()

    #2}}}
    #def GetPPF{{{2
    def GetPPF(self,PPF,rtime="Last"):
        '''
        Retrieves the name of a specific post processing file when there is a series of files corresponding to different time steps. This is the case for the plot over line filter when all time steps are written
        Arguements:

        - *PPF* Name of the attribute containing the list of files
        - *rtime* time to be extracted
        '''

        lPPF    = len(PPF)
        outlist = [0]*lPPF
        if isinstance(rtime,basestring) or isinstance(rtime,int):
            rtm = [rtime for i in range(lPPF)]
        else:
            rtm = rtime

        i = 0
        for p,rt in zip(PPF,rtm):
            #determine directory and extension
            pdir,pnam = os.path.split(p)
            pnam,pext = os.path.splitext(pnam)

            fils = glob.glob(os.path.join(pdir,pnam+'.*'+pext))
            #determine requested time number if specified as last
            if rt == "Last":
                nums = np.array([int(f.split('.')[1]) for f in fils])
                try:
                    r   = nums.argmax()
                except:
                    outlist[i] = None
                    continue
            else:
                r   = rt

            outlist[i]  = fils[r]
            i+=1

        return outlist
    #2}}}
    #def ReshapeJobs{{{2
    def ReshapeJobs(self,matchMP):
        '''
        reshape job list by creating list of list where each row has the same name for main parameter specified in matchMP
        Arguements:

        - *MatchMP* list of indices of the main parameters to match.

        .. todo::
            need a good example for this
        '''

        
        RJobs = []

        matchMP = [self.MPs[i] for i in matchMP]
        names  = [[i.Name  for i in j.info] for j in matchMP]
        names  = list(itt.product(*names))
        for group in names:
            gname = ''.join(gn for gn in group)
            row = [gname]
            for d,n,p,r in self.Jobs:
                if all(substr in n for substr in group):
                    row.append([d,n,p,r])
                    
            RJobs.append(row)

        self.RJobs = RJobs



    #2}}}
    #def GetStep{{{2
    def GetStep(self,Itime,ID='all'):
         '''
         Searches for a particalar time and returns the
         time-step closest to that time. Also returns
         the time associated with that time-step.
         Both ID and Itime can be lists. Itime can be
         passed with the value of 'max' to return only
         the last time-step and time. 

         If multiple times originate the same values for
         the step then that step is only returned once.
         Example: t=[1.2,1.3,1.4] => istp=[32.7,33.2,33.7] =>
                  => rstp=[33,33,34] => [steps]=[33,34]

         Returs the dictionay retval:
         retval[id]=[[steps],[times]]

         Caution: If ID is not a list, the result will still be a dictionary.
                  If Itime is not a list, [steps] and [times] will still be lists.
         '''
         if ID=='all':
            ID=range(self.nJob)
         if type(ID) != type([0]):
            ID=[ID]
         
         retmax=False
         if Itime=='max':
            retmax=True

         if type(Itime) != type([0]):
            Itime=[Itime]
         
         retval=dict()
         for i in ID:
            ts= self.TimeInfo[i][0] #time-steps
            rt= self.TimeInfo[i][1] #real-times
            if not retmax:
               floatts=list(np.interp(Itime,rt,ts))
               ots=list(set([int(round(x)+0.001) for x in floatts]))
               ort=list(np.interp(ots,ts,rt))
            else:
               ots=[ts[-1]] #assumes TimeInfo is ordered
               ort=[rt[-1]]
            retval[i]=[ots,ort]

         return retval
    #2}}}
    #def GetIDs{{{2
    def GetIDs(self,text,combtype='or'):
         '''
         Serches for <text> in all the job names and
         returns a list with the IDs of all jobs that
         match.

         <text> can be a list, in which case the results
         are combined acording to <combtype>.

         <combtype>='or' (default): union of the results of each <text>
         <combtype>='and': intersection of the results of each <text>
         '''
         if type(text)!=type([0]):
            text=[text]
         tsets=[]
         for t in text:
            tsets.append(set([i for i,n in enumerate(self.names) if t in n]))
         if combtype=='or':
            ret= list(set.union(*tsets))
         elif combtype=='and':
            ret= list(set.intersection(*tsets))

         return ret
    #2}}}

#1}}}
#def err_analysis{{{1
def err_analysis(coord_err,time,ref_pvd,ref_csv,comp_pvd,comp_csv,tol=1e-11):
#Check the number of points in the mesh used to compute the error
    nnp_err = coord_err.shape[0]
    if (nnp_err < 1):
        print 'You need at least on point on the error mesh!'
	return
    
#Read the reference solution and constrain it to its values on the error mesh
    ref_csvt = ref_csv.split('.')
    ref_csvt = '.'.join([ref_csvt[0]+"."+str(time),ref_csvt[1]])
    if IsOld(ref_pvd,ref_csvt):
        data = getcsv(ref_pvd,time,ref_csv)
    else :
        data = pvcsvin(ref_csvt)
    
    nodes = np.hstack( (np.array(data['nodes:0']).reshape(-1,1), np.array(data['nodes:1']).reshape(-1,1)) )
    nodes = np.hstack( (nodes, np.array(data['nodes:2']).reshape(-1,1)) )
    u  = np.hstack( (np.array(data['Displacements:0']).reshape(-1,1), np.array(data['Displacements:1']).reshape(-1,1)) )
    u  = np.hstack( (u, np.array(data['Displacements:2']).reshape(-1,1)) )
    T  = np.array(data['Temperature']).reshape(-1,1)
    ep = np.array(data['Plastic Strain']).reshape(-1,1)
    vm = np.array(data['Von Mises']).reshape(-1,1)
    
#    (nodes,u,T,ep,vm) = read_vtu(fi)
    
    i_ref = np.zeros((nnp_err,1),dtype=np.int)
    for i in range(0,nnp_err):
        for j in range(0,nodes.shape[0]):
            if all( abs(nodes[j,0:2]-coord_err[i,:]) < 1e-11 ):
                i_ref[i] = int(j)
                break
    
    
    u_ref = u[i_ref,:]
    T_ref = T[i_ref,:]
    ep_ref = ep[i_ref,:]
    vm_ref = vm[i_ref,:]
    
#Initialize error arrays
    nerrs = len(comp_pvd)
    
    err_u  = np.zeros((nerrs,1))
    err_T  = np.zeros((nerrs,1))
    err_ep = np.zeros((nerrs,1))
    err_vm = np.zeros((nerrs,1))
    
    for k in range(len(comp_pvd)):
        comp_csvt = comp_csv[k].split('.')
        comp_csvt = '.'.join([comp_csvt[0]+"."+str(time),comp_csvt[1]])
        if IsOld(comp_pvd[k],comp_csvt):
            data  = getcsv(comp_pvd[k],time,comp_csv[k])
        else :
            data = pvcsvin(comp_csvt)
        
        nodes = np.hstack( (np.array(data['nodes:0']).reshape(-1,1), np.array(data['nodes:1']).reshape(-1,1)) )
        nodes = np.hstack( (nodes, np.array(data['nodes:2']).reshape(-1,1)) )
        u     = np.hstack( (np.array(data['Displacements:0']).reshape(-1,1), np.array(data['Displacements:1']).reshape(-1,1)) )
        u     = np.hstack( (u, np.array(data['Displacements:2']).reshape(-1,1)) )
        T     = np.array(data['Temperature']).reshape(-1,1)
        ep    = np.array(data['Plastic Strain']).reshape(-1,1)
        vm    = np.array(data['Von Mises']).reshape(-1,1)
        
#        (nodes,u,T,ep,vm) = read_vtu(fi)
	ind = np.zeros((nnp_err,1),dtype=np.int)
	for i in range(0,nnp_err):
            for j in range(0,nodes.shape[0]):
                if all( abs(nodes[j,0:2]-coord_err[i,:]) < 1e-11 ):
                    ind[i] = int(j)
		    break
        
	u = u[ind,:]
	T = T[ind,:]
	ep = ep[ind,:]
	vm = vm[ind,:]
	
	err_u[k,0]  = np.linalg.norm(u-u_ref)/np.linalg.norm(u_ref)
	err_T[k,0]  = np.linalg.norm(T-T_ref)/np.linalg.norm(T_ref)
	err_ep[k,0] = np.linalg.norm(ep-ep_ref)/np.linalg.norm(ep_ref)
	err_vm[k,0] = np.linalg.norm(vm-vm_ref)/np.linalg.norm(vm_ref)
	
    return(err_u,err_T,err_ep,err_vm)

def read_array(fh,ni,nj):
    "This function reads a vtk/vtu DataArray containing ni values with nj components starting at current line of fh."
    out = np.zeros( (ni,nj) )
    for i in xrange(0,ni):
        line = fh.readline().split()
        if nj==1:
            out[i] = float(line[0])
        else:
            for j in xrange(0,nj-1):
                out[i,j] = float(line[j])
    return out;

def read_vtu(f_input):
    f = open(f_input,'r')

#Read some headers and intial VTKFile descriptions
    for i in xrange(0,4):
        f.readline()
        
    line = f.readline().strip('<>\n').replace('"','').replace('=',' ').split()
    numnp  = int(line[2])
    nel    = int(line[4])
#    print "numnp= %i" %numnp
#    print "nel= %i" %nel

#########################
# Read/skip Points info #
#########################
    f.readline() #<Points>
    line = f.readline().strip('<>\n').replace('"','').replace('=',' ').split()
#Store number of dimensions and number of nodes in the mesh
    ndm = int(line[6])
    nodes = np.zeros( (numnp,ndm) )

#Read the nodal coordinate array
    for i in xrange(0,numnp):
        line = f.readline().strip('<>\n').replace('"','').replace('=',' ').split()
        for j in xrange(0,ndm-1):
            nodes[i,j] = float(line[j])
    f.readline()
    
    f.readline() #</Points>

#########################
# Read/skip cells infos #
#########################
    f.readline() #<Cells>
    
#Skip the connectivity array
    line = f.readline().strip('<>\n').replace('"','').replace('=',' ').split()
    for i in xrange(0,nel):
        line = f.readline()
    f.readline()
    
#Skip the offset array
    line = f.readline().strip('<>\n').replace('"','').replace('=',' ').split()
    for i in xrange(0,nel):
        line = f.readline()
    f.readline()
    
#Skip the element type array
    line = f.readline().strip('<>\n').replace('"','').replace('=',' ').split()
    for i in xrange(0,nel):
        line = f.readline()
    f.readline()
    
    f.readline() #</Cells>
    
############################
# Read/skip PointData info #
############################
    f.readline() #<PointData>
    
#Read displacement array
    line = f.readline().strip('<>\n').replace('"','').replace('=',' ').split()
    ndm_u = int(line[6])
    disp = read_array(f,numnp,ndm_u)
    f.readline()
    
#Read temperature array
    line = f.readline().strip('<>\n').replace('"','').replace('=',' ').split()
    ndm_t = int(line[6])
    temp = read_array(f,numnp,ndm_t)
    f.readline()
    
#Read stresses array
    line = f.readline().strip('<>\n').replace('"','').replace('=',' ').split()
    ndm_strs = int(line[6])
    strs = read_array(f,numnp,ndm_strs)
    f.readline()
    
#Read Plastic Strain array
    line = f.readline().strip('<>\n').replace('"','').replace('=',' ').split()
    ndm_ep = int(line[7])
    eqps = read_array(f,numnp,ndm_ep)
    f.readline()
    
#Read von Mises stress array
    line = f.readline().strip('<>\n').replace('"','').replace('=',' ').split()
    ndm_vm = int(line[7])
    vonM = read_array(f,numnp,ndm_vm)
    f.readline()
    
    f.readline() #</PointData>
    f.readline() #</Piece>
    f.readline() #</UnstructuredGrid>
    f.readline() #</VTKFile>
    
    f.close()
    
#    'nodes',nodes
#    'u',disp
#    'T',temp
#    'ep',eqps
#    'vm',vonM
    
    return(nodes,disp,temp,eqps,vonM)
#1}}}
#def getcsv{{{1
def getcsv(PVDFName,time_step,CSVFName):
    #open pvd and 
    PVDF = PVDReader( FileName=PVDFName )
    PVDF.UpdatePipeline()
    
    SetActiveSource(PVDF)
    CSVF = CreateWriter(CSVFName,PVDF)
    CSVF.FieldAssociation = "Points"
    CSVF.WriteAllTimeSteps = 1
    CSVF.UpdatePipeline()
    
    for i in range(len(PVDF.TimestepValues)):
        CSVFi = CSVFName.split('.')
        CSVFi = '.'.join([CSVFi[0]+"."+str(i),CSVFi[1]])
        if i != time_step:
            cmd = "rm "+CSVFi
            os.system(cmd)
        else:
            CSVF0 = CSVFi
    return pvcsvin(CSVF0)
#1}}}
#def get_err{{{1
def get_err(coord_err,ref_file,comp_files,tol=1e-11):
    
#Check the number of points in the mesh used to compute the error
    nnp_err = coord_err.shape[0]
    if (nnp_err < 1):
        print 'You need at least on point on the error mesh!'
	return
    
#Read the reference solution and constrain it to its values on the error mesh
    fi = ref_file
    (nodes,u,T,ep,vm) = read_vtu(fi)
    
    i_ref = np.zeros((nnp_err,1),dtype=np.int)
    for i in range(0,nnp_err):
        for j in range(0,nodes.shape[0]):
            if all( abs(nodes[j,0:2]-coord_err[i,:]) < 1e-11 ):
                i_ref[i] = int(j)
                break
    
    
    u_ref = u[i_ref,:]
    T_ref = T[i_ref,:]
    ep_ref = ep[i_ref,:]
    vm_ref = vm[i_ref,:]
    
#Initialize error arrays
    nerrs = len(comp_files)
    
    err_u  = np.zeros((nerrs,1))
    err_T  = np.zeros((nerrs,1))
    err_ep = np.zeros((nerrs,1))
    err_vm = np.zeros((nerrs,1))
    
    for k,fi in enumerate(comp_files):
        (nodes,u,T,ep,vm) = read_vtu(fi)
	ind = np.zeros((nnp_err,1),dtype=np.int)
	for i in range(0,nnp_err):
            for j in range(0,nodes.shape[0]):
                if all( abs(nodes[j,0:2]-coord_err[i,:]) < 1e-11 ):
                    ind[i] = int(j)
		    break
        
	u = u[ind,:]
	T = T[ind,:]
	ep = ep[ind,:]
	vm = vm[ind,:]
	
	err_u[k,0]  = np.linalg.norm(u-u_ref)/np.linalg.norm(u_ref)
	err_T[k,0]  = np.linalg.norm(T-T_ref)/np.linalg.norm(T_ref)
	err_ep[k,0] = np.linalg.norm(ep-ep_ref)/np.linalg.norm(ep_ref)
	err_vm[k,0] = np.linalg.norm(vm-vm_ref)/np.linalg.norm(vm_ref)
	
    return(err_u,err_T,err_ep,err_vm)
#1}}}
#inp = inputs[0]
##Determine length, area, or volume
#lav = ['Length','Area','Volume']
#if not inp.GetCellData() is None:
#    if inp.GetCellData().GetNumberOfArrays() > 0:
#        for CellArrayIdx in range(0,inp.GetCellData().GetNumberOfArrays()):
#            data = inp.GetCellData().GetArray(CellArrayIdx)
#            name = inp.GetCellData().GetArrayName(CellArrayIdx)
#            if name in lav:
#                LAV = data.GetTuple(0)[0]
##Scale point data by LAV
#npa = inp.GetPointData().GetNumberOfArrays()
#nams= [0]*npa
#if not inp.GetPointData() is None:
#    if inp.GetPointData().GetNumberOfArrays() > 0:
#        for PointArrayIdx in range(0,inp.GetPointData().GetNumberOfArrays()):
#            name = inp.GetPointData().GetArrayName(PointArrayIdx)
#            nams[PointArrayIdx] = name
#for n in nams:
#    data = inp.PointData[n]
#    data = data/LAV
#    output.PointData.append(data,n)
#def view_mesh {{{1
def view_mesh(mesh_file,pdf_file='mesh.pdf',mesh_type='feap',openpdf=True,shownums=False):
    '''
    This function reads mesh files and uses numpy and matplotlib to plot it in a pdf document.
    Arguements:
    
    - *mesh_file* the file to be read
    - *pdf_file* file containing a graphic visualization of the mesh
    - *mesh_type* describes the format of the mesh to chose the good reader option. So far only FEAP is implemented.
    - *openpdf* issues a command to the system to open the pdf file if set to True.
    - *shownums* prints the node numbers and the element numbers
    '''
    if(not os.path.isfile(mesh_file)):
        print "%s, is not a valid file!"%(mesh_file)
        return 1
    else:
        fmesh = open(mesh_file)
        lines = fmesh.readlines()
        coords = np.empty((0,3))
        connec = np.empty((0,4))
        if(mesh_type == "feap"):
            count_coord = 0
            read_coord  = False
            coords      = np.empty((0,3))
            count_elem  = 0
            read_elem   = False
            elems       = np.empty((0,4),dtype='int32')
            for j in range(len(lines)):
                if '!Blank Termination Record' in lines[j]:
                    read_coord  = False
                    read_elem   = False
                
                if read_coord:
                    line   = lines[j].strip().split(',')
                    coords = np.vstack( (coords,np.array(line[2:5],dtype='float32')) )
                    count_coord += 1
                elif read_elem:
                    line  = lines[j].strip().split(',')
                    elem = np.array(line[3:7],dtype='int32')-np.ones((4,),dtype='int32')
                    elems = np.vstack( (elems,elem) )
                    count_elem += 1
                
                if 'COOR' in lines[j]:
                    read_coord  = True
                elif 'ELEM' in lines[j]:
                    read_elem   = True
        fmesh.close()
        print "view_mesh found %i nodes in %i elements while reading the mesh." %(count_coord,count_elem)
        
        #----------------------------------------------#
        #                                              #
        #       Plotting commands and parameters       #
        #                                              #
        #----------------------------------------------#
        #plt.rc('text', usetex=True)
        #plt.rc('font',**{'family':'serif','serif':['Times'],'size':10})
        plt.rc('lines',**{'linewidth':0.1,'markersize':6})
        #plt.rc('legend',**{'numpoints':1,'fontsize':'medium'})
        plt.rc('font',**{'size':8})
        quarterfig = (2.5,1.875)
        halffig    = (5.0,3.75) 
        line=['-o','-s','-p','-*','-^']
        c=['b','r','g','k','y']

        if shownums:
           hsize=np.amax(coords[:,0])-np.amin(coords[:,0])
           vsize=np.amax(coords[:,1])-np.amin(coords[:,1])
           Gsize=max(hsize,vsize)
           Nsize=np.ones(count_coord)
        
        fig = plt.figure(figsize=halffig)
        ax = fig.add_subplot(111)
        for j in range(count_elem):
            sctr = np.hstack((elems[j,:],elems[j,0]))
            x = coords[sctr,0]
            y = coords[sctr,1]
            ax.plot(x,y,'b-',c='b',mec='b')
            if shownums:
               xc=np.mean(x[0:-1])
               yc=np.mean(y[0:-1])
               dl=np.array([((xx-xc)**2+(yy-yc)**2)**0.5/Gsize for xx,yy in zip(x[0:-1],y[0:-1])])
               d=np.amin(dl)
               Nsize[elems[j,:]]=np.minimum(Nsize[elems[j,:]],dl[:])
               ax.text(xc,yc,'E'+str(j+1),fontsize=100*d,verticalalignment='center', horizontalalignment='center')

        if shownums:
           for j in range(count_coord):
              ax.text(coords[j,0],coords[j,1],str(j+1),
                    fontsize=80*Nsize[j])



        ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x,p :"%2.1E"%(x)))
        ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x,p :"%2.1E"%(x)))
        ax.set_aspect('equal')
        fig.tight_layout()
        plt.savefig(os.path.abspath(pdf_file))
        plt.close()
        if openpdf:
            cmd = ' '.join(['xdg-open',os.path.abspath(pdf_file)])
            os.system(cmd)
#1}}}
#def gmsh2feap{{{1
def gmsh2feap(ifile,ofile):
    '''
    This function converts files from the gmsh format to FEAP's format.
    Arguments:
    
    - *ifile* input file in gmsh format
    - *ofile* output file in FEAP's format
    '''
    #get nummat,num
    #Read from input {{{2
    matid = []
    fi = open(ifile,'r')
    fo = open(ofile,'w')
    with open(ifile,'r') as fi:
        with open(ofile,'w') as fo:
            fo.write('COORdinates\n')
            #advance to start of nodes
            while True:
                if fi.readline().startswith("$Nodes"):
                    fi.readline()
                    break

            #get nodes
            while True:
                line = fi.readline()
                if line.startswith("$EndNodes"):
                    fo.write('           !Blank Termination Record\n')
                    break
                else:
                    s = [float(p) for p in line.split()]
                    s = [s[0],0]+s[1:]
                    oline  = "  %d,%d,%E,%E,%E \n" % tuple(s)
                    fo.write(oline)

            fo.write('ELEMents\n')
            #advance to start of elements
            while True:
                if fi.readline().startswith("$Elements"):
                    fi.readline()
                    break

            #get elements
            while True:
                line = fi.readline()
                if line.startswith("$EndElements"):
                    fo.write('           !Blank Termination Record\n')
                    break
                else:
                    l = line.split()
                    #check for material
                    mm = l[3]
                    if mm not in matid:
                        matid.append(mm)
                    ma = matid.index(mm)+1
                    s = [int(p) for p in l]
                    nrec = s[2]
                    esta = 3+nrec
                    s = [s[0],0,ma]+s[esta:]
                    s = ",".join(str(i) for i in s)	
                    s = '  '+s+'\n'

                    fo.write(s)
                    #2}}}
#1}}}
#def run_gmsh{{{1
def run_gmsh(gmsh,geo,msh,options="",print_flg=True):
    gmsh = (gmsh+" "+geo+" -o "+msh+" "+options).split()
    proc = subprocess.Popen(gmsh,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    out, err = proc.communicate()
    if print_flg:
        print out
#1}}}
#def write_mesh {{{1
def write_mesh(coord,elem,file_name):
    '''
    Deprecated function, the mesh class should be used instead of this functions!
    '''
    fo = open(file_name,'w')
    fo.write('COORdinates\n')
    for i in range(coord.shape[0]):
        line = '  %i,0,%E,%E,%E\n' %(i+1,coord[i,0],coord[i,1],coord[i,2])
        fo.write(line)
    fo.write('           !Blank Termination Record\n')
    fo.write('ELEMents\n')
    for i in range(elem.shape[0]):
        lst  = np.hstack( ([i+1,0,1],elem[i,:]) )
        line = ','.join([str(int(word)) for word in lst]+['\n'])
        line = '  '+line
        fo.write(line)
    fo.write('           !Blank Termination Record\n')
    fo.close()	
    
    print '\n','Mesh Data Written to',file_name,'\n'
    print 'Number of Nodes = ',str(coord.shape[0]),'\n'
    print 'Number of Elements = ',str(elem.shape[0]),'\n'
#1}}}
#Class mesh {{{1
class mesh():
    '''
    The mesh class allow to instantiate mesh objects, check their initialization with the setup() method and actually generate a mesh with the generate() method.
    Arguements:
    
    - *imesh*       the input file containing the geometry to mesh (this can be an empty path in certain cases)
    - *omesh*       the ouput file containing the generated geometry
    - *generator*   the program used to generate the mesh. Currently the following generators are supported:
    
      + 'auto'  uses a python script to generate the nodes and elements of the mesh (in this case, *imesh=' '*)
      + 'gmsh'  uses gmsh to generate the mesh.
    
    - *gen_loc*     path to the generator executable
    - *ofmt*        format of the output mesh, currently only FEAP format is supported
    - *geom*        description of the geometries implemented ['square', 'rectangle', 'shell'], only 'shell' requires an input file
    - *\*\*kwargs*  dictionary describing the different geometries availables.
    '''
    #def __init__{{{2
    def __init__(self,imesh,omesh,generator,gen_loc='',ofmt='feap',geom='',**kwargs):
        self.imesh   = imesh
        self.omesh   = omesh
        self.gen     = generator
        self.gen_loc = gen_loc
        self.fmt     = ofmt
        self.geom    = geom
        self.state   = 0
        self.stats   = {'numnp':0, 'numel':0, 'matid':[]}
        self.kwargs  = kwargs
    #2}}}
    #def __str__{{{2
    def __str__(self):
        msg = ''
        if self.state < 2:
            msg = msg + 'No mesh has been created yet, only initialization informations are available:\n' + \
                        '    input geometry file:        %s\n' %self.imesh + \
                        '    mesh generator:             %s\n' %self.gen + \
                        '    path to the mesh generator: %s\n' %self.gen_loc
            if (self.gen == 'gmsh') and (self.geom == 'shell'):
                if 'geop' in self.kwargs:
                    msg = msg + '        Geometric parameters to be substituted:\n'
                    for param in self.kwargs['geop']:
                        msg = msg + '          %s: %s\n'%(param[0],param[1])
            msg = msg + '    ouput mesh file:            %s\n' %self.omesh + \
                        '    output format of the mesh:  %s\n' %self.fmt
            if self.state < 1:
                msg = msg + '\n---------------------------------------------------------------------------\n' + \
                              'The mesh has not been setup, it is unsage to call generate() at this point!\n' + \
                              '---------------------------------------------------------------------------\n'
            else:
                msg = msg + 'The mesh has been setup, it is safe to use generate()'
        else:
            msg = msg + '    Ouput mesh file:            %s\n' %self.omesh + \
                        '    output format of the mesh:  %s\n' %self.fmt + \
                        '\n' + \
                        '    mesh statistics\n' + \
                        '    ---------------\n' + \
                        '        number of nodes:         %s\n' %self.stats['numnp'] + \
                        '        number of elements:      %s\n' %self.stats['numel'] + \
                        '        number of material sets: %s\n' %str(len(self.stats['matid']))
        return msg
    #2}}}
    #def setup(){{{2
    def setup(self):
        '''
        Checks the inputs for consistency, apply default values to assumed variables and quit if required values are missing.
        '''
        gen_list  = ['auto', 'gmsh']
        fmt_list  = ['feap']
        geom_list = ['square', 'rectangle', 'line', 'shell']
        if(not self.gen in gen_list):
            print 'Generator: %s is not recognized.\n' %self.gen
            sys.exit()
        
        if(not self.gen == 'auto') and (self.gen_loc == ''):
            print 'If gen=%s, gen_loc needs to be specified.\n' %self.gen
            sys.exit()
        
        #Check that the output mesh format and file name are valid
        if(not self.fmt in fmt_list):
            print 'Output format: %s is not recognized.\n' %self.fmt
            self.fmt = fmt_list[0]
            print 'The output format is now set to: %s\n' %self.fmt
        opath = os.path.split(self.omesh)[0]
        oname = os.path.split(self.omesh)[1]
        path_msg = ''
        if(not os.path.isdir(opath)):
            opath = os.getcwd()
            path_msg = path_msg + 'Path: %s is not valid for outputing the mesh.\n' %os.path.split(self.omesh)[0]
        if(not oname):
            oname = 'feapmsh'
            path_msg = path_msg + 'The output file name is not specified.\n'
        if(not self.omesh == os.path.join(opath,oname)):
            self.omesh = os.path.join(opath,oname)
            path_msg = path_msg + 'The mesh will be saved in: %s' %self.omesh
            print path_msg
        
        #Check that the input mesh is valid
        if(self.geom == 'shell') and (not os.path.isfile(self.imesh)):
            print 'File: %s is not a valid file path.' %self.imesh
            sys.exit()
            
        if(not self.geom in geom_list):
            print 'Geom: %s is not supported.\n' %self.geom
            sys.exit()
        
        if (self.geom in ['square', 'rectangle']):
            if (not 'Lx' in self.kwargs.keys()) or (not 'nx' in self.kwargs.keys()):
                print 'With geom=(square, rectangle) you need to specify a value for Lx and nx in **kwargs.\n'
                sys.exit()
            if (self.geom == 'rectangle') and ( (not 'Ly' in self.kwargs.keys()) or (not 'ny' in self.kwargs.keys()) ):
                print 'With geom=rectangle if Ly resp. ny **kwargs is empty, it is set equal to Ly resp. ny.\n'
                self.kwargs['Ly'] = self.kwargs['Lx'] if (not 'Ly' in self.kwargs.keys()) else self.kwargs['Ly']
                self.kwargs['ny'] = self.kwargs['nx'] if (not 'ny' in self.kwargs.keys()) else self.kwargs['ny']
            if self.geom=='square':
                self.kwargs['Ly']=self.kwargs['Lx']
                self.kwargs['ny']=self.kwargs['nx']
        
        
        self.state = 1
    #2}}}
    #def generate(){{{2
    def generate(self):
        '''
        This function calls an input format specific reader to read the nodes and elements line by line.
        Each line is formated by FmtLine to and written into the output mesh file.
        '''
        if self.state < 1:
            print 'You need to call mesh.setup() before mesh.generate().\n'
            sys.exit()
        
        if self.gen == 'auto':
            robj = self.auto_gen(self.kwargs)
            
        if self.gen == 'gmsh':
            robj = self.gmsh_gen(self.kwargs)
        
        flgs = {'nodes':False, 'elems':False, 'ifmt':self.gen, 'ofmt':self.fmt, 'eof':False, 'matid':[]}
        fo = open(self.omesh,'w')
        while not flgs['eof']:
            line = self.FmtLine(robj,flgs)
            fo.write(line)
        fo.close()
        self.stats['matid'] = flgs['matid']
        self.state = 2
    #2}}}
    #def auto_gen(){{{2
    def auto_gen(self,geom):
        '''
        This function generates nodes and elements using python and stores them in the output dictionary: *{'coords':coords, 'elems':elems}*. Since the mesh is not stored in an input file it is passed through the arguement *geom* which is a dictionary with the following form: geom={'Lx':Lx, 'Ly':Ly, 'nx':nx, 'ny':ny}.
        '''
        if self.geom in ['square', 'rectangle']:
            Lx = float(geom['Lx'])
            Ly = float(geom['Ly'])
            nx = int(geom['nx'])
            ny = int(geom['ny'])
            
            x = np.linspace(0,Lx,nx)
            y = np.linspace(0,Ly,ny)
            xv, yv = np.meshgrid(x, y)
            xv = xv.reshape( (nx*ny,1) )
            yv = yv.reshape( (nx*ny,1) )
            
            coords = np.hstack( (xv,yv) )
            coords = np.hstack(( coords, np.zeros(( nx*ny,1 )) ))

            
            #Corner nodes go 1,2,3,4
            corners=[0,nx-1,nx*ny-1,nx*ny-nx+1-1]
            #Swap array
            sw=range(nx*ny)
            for i,c in enumerate(corners):
                sw[i]=c
                sw[c]=i
         
            #Swap node coordinates
            ci=np.zeros(len(coords[0]))            
            for i,c in enumerate(corners):
                ci[:]=coords[i,:]
                coords[i,:]=coords[c,:]
                coords[c,:]=ci[:]
                
            #Elements
            elems = np.zeros(( (nx-1)*(ny-1),4 ))
            for i in range(ny-1):
                for j in range(nx-1):
                    elems[i*(nx-1)+j,:] = [sw[i*nx+j]+1,sw[i*nx+j+1]+1,sw[(i+1)*nx+j+1]+1,sw[(i+1)*nx+j]+1]           
            
        else:
            print 'line option not implemented yet.\n'
        
        self.stats['numnp'] = coords.shape[0]
        self.stats['numel'] = elems.shape[0]
        self.stats['matid'] = [1]
        mesh_info = {'coords':coords, 'elems':elems}
        
        return mesh_info
    #2}}}
    #def gmsh_gen(){{{2
    def gmsh_gen(self,geom):
        rm_geo = False
        if self.geom in ['square', 'rectangle']:
            rm_geo = True
            Lx = float(geom['Lx'])
            Ly = float(geom['Ly'])
            nx = int(geom['nx'])
            ny = int(geom['ny'])
            geo_file = ''.join([self.omesh,'.geo'])
            fgeo = open(geo_file,'w')
            geo_script = "General.Light0X = -0.70;\n" + \
                         "General.Light0Y = -0.70;\n" + \
                         "General.Light0Z = 0;\n" + \
                         "\n" + \
                         "Mesh.RecombineAll=1; \n" + \
                         "Mesh.Algorithm=8; \n" + \
                         "Mesh.SubdivisionAlgorithm=1;\n" + \
                         "Mesh.ColorCarousel=0;\n" + \
                         "Mesh.Smoothing = 10; // Elliptic smoother\n" + \
                         "\n" + \
                         "Lx  = %E; // Plate length in x direction\n" %(Lx) + \
                         "Ly  = %E; // Plate length in y direction\n" %(Ly) + \
                         "xx  = %i;\n" %(nx) + \
                         "nx  = xx/2+1; // Number of nodes on each side of the square\n" + \
                         "yy  = %i;\n" %(ny) + \
                         "ny  = yy/2+1; // Number of nodes on each side of the square\n" + \
                         "pr  = 1.0;    // Progression of element sizes \n" + \
                         "clx = pr/nx;  // Mesh size - Not used in structured\n" + \
                         "cly = pr/ny;  // Mesh size - Not used in structured\n" + \
                         "\n" + \
                         "Point(1) = {0, 0, 0, clx};\n" + \
                         "Point(2) = {Lx, 0, 0, cly};\n" + \
                         "Point(3) = {Lx, Ly, 0, clx};\n" + \
                         "Point(4) = {0, Ly, 0, cly};\n" + \
                         "\n" + \
                         "Line(1) = {1, 2};\n" + \
                         "Line(2) = {2, 3};\n" + \
                         "Line(3) = {3, 4};\n" + \
                         "Line(4) = {4, 1};\n" + \
                         "\n" + \
                         "Line Loop(6) = { 1, 2, 3, 4};\n" + \
                         "Ruled Surface(7) = {6};\n" + \
                         "\n"
            transfinite= "Transfinite Line {1, 3} = nx Using Progression pr;\n" + \
                         "Transfinite Line {2, 4} = ny Using Progression 1/pr;\n" + \
                         "Transfinite Surface{7} = {1,2,3,4};\n" + \
                         "\n"
            if self.kwargs['unstr']:
                geo_script = geo_script + "Physical Surface(1) = {7};"
            else:
                geo_script = geo_script + transfinite + "Physical Surface(1) = {7};"
            
            fgeo.write(geo_script)
            fgeo.close()
        elif self.geom == 'shell':
            rm_geo = False
            geo_file = self.imesh
            if 'geop' in self.kwargs:
                os.rename(geo_file,geo_file.replace('.geo','_temp.geo'))
                fgeo_in  = open(geo_file.replace('.geo','_temp.geo'),'r')
                fgeo_out = open(geo_file,'w')
                for line in fgeo_in:
                    for param in self.kwargs['geop']:
                        line = line.replace(param[0],str(param[1]))
                    fgeo_out.write(line)
                fgeo_in.close()
                fgeo_out.close()
        
        msh_file=''.join([self.omesh,'.msh'])
        run_gmsh(self.gen_loc,geo_file,msh_file,options=" -2 -nopopup",print_flg=False)
        if (self.geom == 'shell') and 'geop' in self.kwargs:
            os.remove(geo_file)
            os.rename(geo_file.replace('.geo','_temp.geo'),geo_file)
        if rm_geo:
            os.remove(geo_file)
        fi = open(msh_file)
        return fi
    #2}}}
    #def FmtLine(){{{2
    def FmtLine(self,robj,flgs):
        '''
        Depending on the input format, output format and flgs, FmtLine reads the current line in robj and translate it from input to ouput format.
        '''
        #Generate all lines for auto generator
        if flgs['ifmt']=='auto':
            flgs['matid'].append(1)
            lines = 'COORdinates\n'
            for i in range(self.stats['numnp']):
                lines = lines + '  %i,0,%E,%E,%E\n' %(i+1,robj['coords'][i,0],robj['coords'][i,1],robj['coords'][i,2])
            lines = lines + '           !Blank Termination Record\n'
            lines = lines + 'ELEMents\n'
            for i in range(self.stats['numel']):
                lst  = np.hstack( ([i+1,0,1],robj['elems'][i,:]) )
                line = ','.join([str(int(word)) for word in lst]+['\n'])
                line = '  '+line
                lines = lines + line
            lines = lines + '           !Blank Termination Record\n'
            flgs['eof'] = True
            return lines
        
        if flgs['ifmt']=='gmsh':
            #if we read for the first time, go to the begining of nodes
            if( (not flgs['nodes']) and (not flgs['elems']) ):
                lines = 'COORdinates\n'
                while True:
                    if robj.readline().startswith("$Nodes"):
                        self.stats['numnp'] = int(robj.readline())
                        flgs['nodes']=True
                        return lines
            
            #if there are still nodes add them, otherwise advance to elements.
            if flgs['nodes']:
                line = robj.readline()
                if not line.startswith("$EndNodes"):
                    s = [float(p) for p in line.split()]
                    s = [s[0],0]+s[1:]
                    lines = "  %d,%d,%E,%E,%E \n" % tuple(s)
                else:
                    flgs['nodes'] = False
                    lines = '           !Blank Termination Record\n'
                    while True:
                        line = robj.readline()
                        if line.startswith('$Elements'):
                            lines = lines + 'ELEMents\n'
                            self.stats['numel'] = int(robj.readline())
                            flgs['elems']=True
                            return lines
            
            if flgs['elems']:
                line = robj.readline()
                if not line.startswith('$EndElements'):
                    l = line.split()
                    #check for material
                    mm = l[3]
                    if mm not in flgs['matid']:
                        flgs['matid'].append(mm)
                    ma = flgs['matid'].index(mm)+1
                    s = [int(p) for p in l]
                    nrec = s[2]
                    esta = 3+nrec
                    s = [s[0],0,ma]+s[esta:]
                    s = ",".join(str(i) for i in s)	
                    s = '  '+s+'\n'
                    
                    lines = s
                else:
                    lines = '           !Blank Termination Record\n'
                    flgs['elems'] = False
                    flgs['eof'] = True
            
            return lines
    #2}}}
#1}}}
#def getFrustumFromBounds{{{1
def getFrustumFromBounds(frustumBounds):
    #Returns a list defining a frustum in paraview
    xmin = frustumBounds[0]
    ymin = frustumBounds[1]
    zmin = frustumBounds[2]

    xmax = frustumBounds[3]
    ymax = frustumBounds[4]
    zmax = frustumBounds[5]

    frustum = [xmin,ymin,zmax,1,
               xmin,ymin,zmin,1,
               xmin,ymax,zmax,1,
               xmin,ymax,zmin,1,
               xmax,ymin,zmax,1,
               xmax,ymin,zmin,1,
               xmax,ymax,zmax,1,
               xmax,ymax,zmin,1]

    return frustum
#1}}}
