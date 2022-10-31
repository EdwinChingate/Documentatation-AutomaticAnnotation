from scipy.stats import t
from scipy import stats
import numpy as np
def WelchTest(Ds1,Ds2,alpha=0.05): 
    #Statistical test to check if two fragments are different
    #Ds1 and Ds2 are vectors that summarize the information on the samples [average,std,size]
    s1=Ds1[1]/np.sqrt(Ds1[2])
    s2=Ds2[1]/np.sqrt(Ds2[2])
    t=abs(Ds1[0]-Ds2[0])/np.sqrt(s1**2+s2**2)
    df=(Ds1[1]**2/Ds1[2]+Ds2[1]**2/Ds2[2])**2/(Ds1[1]**4/((Ds1[2]-1)*Ds1[2]**2)+Ds2[1]**4/((Ds2[2]-1)*Ds2[2]**2))
    tref=stats.t.interval(1-alpha, df)[1]
    p=0 #I need to include the calculation of the p-value
    if t>tref:
        val=True
    else:
        val=False
    return [val, t, tref, p]

import numpy as np
import pandas as pd
def SolveSpace(TargetM,Mat=[],Bot=0,p=0,vec=np.zeros(12),Top='Start',MaxPos=np.ones(12)*5):   
    MassDic={'K':38.9637064875,'Na':22.989769282019,'C13':13.003354835336252,'C':12,'Cl':34.968852694,'S34':33.967867015,'S':31.972071174414,'P':30.97376199867,'F':18.998403227,'O':15.994914619257319,'N':14.003074004251241,'H':1.00782503189814}
    Mass=np.array(list(MassDic.values()),dtype=float)
    #K-Na-C13-C-Cl-S34-S-P-F-O-N-H
    #0-1--2---3-4---5--6-7-8-9-10-11
    if Top=='Start':
        Top=min(int(TargetM/Mass[0]),MaxPos[0])
        Top1=1-Top
    else:
        Top1=500
    vec[p]=Top             
    if p<len(Mass)-1:
        MissingMass=TargetM-sum(Mass[:p+1]*vec[:p+1])
        if p>3:
            Spots=(vec[2]+vec[3])*2+2-vec[4]-vec[8]+vec[10]
        else:
            Spots=500
        TopN=min(int(MissingMass/Mass[p+1])+1,MaxPos[p+1],Top1,Spots)
        if TopN<0:
            TopN=0       
        if p==len(Mass)-2 and TopN>0:
           # TopN=min(int(MissingMass/Mass[p+1])+1,MaxPos[p+1],Top1,Spots)
            Bot2=max(TopN-1,int(Spots/4))
        else:
            Bot2=Bot            
        Mat=SolveSpace(p=p+1,Mat=Mat,Top=TopN,vec=vec.copy(),TargetM=TargetM,Bot=Bot2,MaxPos=MaxPos)
    else:   
        Mat.append(vec.copy())                          
    if Top>Bot:                
        Mat=SolveSpace(p=p,Mat=Mat,Top=Top-1,vec=vec.copy(),TargetM=TargetM,Bot=Bot,MaxPos=MaxPos)    
    return Mat  
    
    
from IPython.display import HTML, display
import tabulate	
def ShowDF(DF,col=''):
    if col=='':
        col=list(DF.columns)
    display(HTML(tabulate.tabulate(DF[col], headers= col,tablefmt='html')))
    
    
import pandas as pd
import numpy as np
from FitFragment import *
def SelfConsistFrag(DF,returnMat=False):
    LM=len(DF['Formula'])
    mat=np.zeros((LM,LM))
    D=pd.DataFrame(mat,columns=DF.index,index=DF.index)
    L=len(DF)
    c=0
    Mat=[]
    MF=list(DF.groupby(['Measured_m/z']).groups.keys())
 #   ShowDF(D)
    for x in np.arange(len(MF)-1):
        for y in np.arange(x+1,len(MF)):
            D=FitFragment(DF,D,MF[x],MF[y],Mat)
    if returnMat:
        return pd.DataFrame(Mat,columns=['K','Na','C13','C','Cl','S43','S','P','F','O','N','H'])
    return D
    
from scipy.stats import t
from scipy import stats
import numpy as np
def PondMZStats(peaks,alpha=0.01):
  #  print('here')
    dimen=np.shape(peaks)
    if dimen[0]<dimen[1]:
        peaks=peaks.copy().T#[:,Spec[1,:]/max(Spec[1,:])>Noise]####
    else:
        peaks=peaks.copy()    
    SumIntens=sum(peaks[:,1])
    #print(SumIntens)
    RelativeInt=peaks[:,1]/SumIntens
    MostInt=max(peaks[:,1])
   # print(MostInt)
    whereMostInt=np.where(peaks[:,1]==MostInt)[0]
    MostIntFrag=peaks[whereMostInt,0][0]
    AverageMZ=sum(peaks[:,0]*RelativeInt)
    l=len(peaks[:,1])    
    Varian=sum(RelativeInt*(peaks[:,0]-AverageMZ)**2)*l/(l-1)
    tref=stats.t.interval(1-alpha, l-1)[1]
    Std=np.sqrt(Varian)
    #print(peaks)
    VecStats=[AverageMZ,Std,l,tref*Std/np.sqrt(l),tref*Std/np.sqrt(l)/AverageMZ*1e6,MostIntFrag,SumIntens]      
    return VecStats
    

import numpy as np
import pandas as pd
from WelchTest import *
from PondMZStats import *
from ShowDF import *
def NewReduceSpecFindPeaks(peaks,MinInt=5e3,NoiseInt=1e2,DischargeT=80,Techo=1000,RelDis=2,MinPoint=3): #Filtering noise and reducing similar peaks    
    #MinDis can be a very delicated parameter, as it defines the threshold in between isotopomers and different substances
    Spec=np.array(peaks)
   # ShowDF(peaks)
    #print(Spec)
    dimen=np.shape(Spec)
    #print(dimen)
    if dimen[0]<dimen[1]:
        peak=Spec.copy().T
    else:
        peak=Spec.copy()    
    NonZero=np.where(peak[:,1]>NoiseInt)[0]    
  #  print(NonZero)
    Peak=np.array(peak[NonZero,:])
    #ShowDF(pd.DataFrame(Peak))
    L=len(Peak[:,1].copy())
   # print(L)
    Peak=np.concatenate([Peak,Peak[-1:,:]])
    NFrag=len(Peak[:,0])
    v0=np.arange(NFrag)
    v1=np.arange(NFrag)+1
    DistancesE=Peak[v1[:-1],0]-Peak[v0[:-1],0]    
    x1=0
    x2=0
    NewSpec=[]
    mzRef=Peak[x1,0]
    MinDis=10#DischargeT/1e6*mzRef
    Npeak=0
    TotalI=1
    while x2<L and mzRef<Techo:
        d=(Peak[x2+1,0]-mzRef)
       # print(d,L)
        #print('d',d,MinDis,x2,x1)
       # print('F',Peak[x2,0],Peak[x1,0])
        if abs(d)>MinDis or x2>L-3:  
            if data[-6]>MinInt:                
                if Npeak>0:                                        
                    #print('\n')
                   # print(d)
                    ##print(MinDis,mzRef,x1,x2)
                    ##print(Npeak)
                    STest=WelchTest(NewSpec[-1],data,alpha=0.01)
                    #print(STest)
                    if STest[0]:                              
                        data[-3]=STest[1]
                        data[-2]=STest[2]
                        data[-1]=STest[3]
                        NewSpec.append(data)
                    else:
                       # print(mzRef)
                        PrevDat=Peak[NewSpec[-1][-5]:NewSpec[-1][-4],:2]     
                        JoinSpec=np.append(PrevDat,PosibleSpec,axis=0)                                              
                        data[:-5]=PondMZStats(JoinSpec)
                        data[-5]=NewSpec[-1][-5]
                        if Npeak>1:
                            STest=WelchTest(NewSpec[-2],data,alpha=0.01)
                            data[-3]=STest[1]
                            data[-2]=STest[2]
                            data[-1]=STest[3]                        
                        NewSpec[-1]=data                        
                        Npeak-=1
                else:
                    NewSpec.append(data)
                    
                Npeak+=1 #I need to check what's happening with this one
               # if 2%Npeak==0:
                  #  print(Npeak,time.time()-t0)
            x1=x2+1            
            x2=x1+1   
            mzRef=Peak[x1,0]          
            MinDis=DischargeT/1e6*mzRef
        else:
            x2+=1
         #   print('here')
            PosibleSpec=Peak[x1:x2+1,:2].copy()
            data=PondMZStats(PosibleSpec)         
            MinDis=3*data[1]
            mzRef=float(data[0])
          #  DistancesE=Peak[:,0]-mzRef
           # ErrorDIds=np.where(DistancesE[x2:]>MinDis)[0]
           # TotalI=data[-6]
          #  print(x1,x2)
          #  if len(ErrorDIds)==0:
             #   xn=L-x2-2
          #  else:
          #      xn=ErrorDIds[0]
                #print(mzRef,MinDis,x1,x2,ErrorDIds[0],xn,L)
                #print(MinDis)
                #print(DistancesE[x2:])
                #print(ErrorDIds)               
            #ErrorDIds=np.append(ErrorDIds,)            
            
           # if xn>10:
         #       xn-=1
          #  elif xn<0:
              #  print(xn)
            #    xn=1
         #   print('here')               
            
           # x2+=xn         
           # PosibleSpec=Peak[x1:x2,:2].copy()
           # data=PondMZStats(PosibleSpec)
            data.append(x1)
            data.append(x2)
            data.append(0)
            data.append(0)
            data.append(0)
            #MinDis=3*data[1]            
          #  mzRef=float(data[0])
            #print('here')
    NewSpec=np.array(NewSpec)  
   # ShowDF(pd.DataFrame(NewSpec))
    Discharge=np.where((NewSpec[:,2]>MinPoint)&(NewSpec[:,4]<DischargeT)&(NewSpec[:,6]/max(NewSpec[:,6])*100>RelDis))[0]      
    NewSpec=NewSpec[Discharge,:]
    Mat=pd.DataFrame(NewSpec,columns=['Mean_m/z','Std_m/z','DataPoints','ConfidenceInterval','ConfidenceInterval(ppm)','MostIntense_m/z','TotalIntensity','MinID','MaxID','t_value','t_ref','p'])
    Mat['RelInt']=Mat['TotalIntensity']/sum(Mat['TotalIntensity'])*100
    return Mat
    
import numpy as np
import pandas as pd
from SolveSpace import *
from Formula import *
from ExactMassCal import *
def MoleculesCand(TargetM,RelInt=0,ExpectedV={'K':1,'Na':1,'C13':1,'C':40,'Cl':1,'S34':1,'S':3,'P':1,'F':1,'O':20,'N':20,'H':100},Tres=10):                          
    MaxVal=list(ExpectedV.values())
   # print('he')
    LotofPos=SolveSpace(TargetM=TargetM,MaxPos=np.array(MaxVal),Mat=[])
    
    MassPoss=np.array(list(map(ExactMassCal,LotofPos)))
    MassDiff=abs(MassPoss-TargetM)/TargetM*1e6
   # print(MassDiff)
    Li=len(MassPoss)
   # for x in np.arange(Li):
   #     print(MassPoss[x],MassDiff[x],TargetM)
    BestM=np.where(MassDiff<Tres)[0]
    if len(BestM)==0:
        return 0
   # print(BestM)
    LotofPosMat=np.array(LotofPos)        
    BestOnes=LotofPosMat[BestM,:].copy()
    BestOnesFancy=pd.DataFrame(BestOnes,columns=['K','Na','C13','C','Cl','S43','S','P','F','O','N','H'])
    # ShowDF(BestOnesFancy)
    Formula(BestOnesFancy)
    BestOnesFancy['Error (ppm)']=MassDiff[BestM]
    BestOnesFancy['Predicted_m/z']=MassPoss[BestM]
    BestOnesFancy['Measured_m/z']=TargetM
    BestOnesFancy['ConfidenceInterval(ppm)']=Tres
    BestOnesFancy['RelInt']=RelInt
    #BestOnesFancy['loc']=BestM
   # ShowDF(BestOnesFancy)
    #print()
    return BestOnesFancy
    
import numpy as np
from FragNetIntRes import *
from ShowDF import *
def MinEdges(DF,Vsum):	
    red=np.zeros(5)
   # ShowDF(DF)
  #  print('Vsum',Vsum)
    for x in np.arange(5):        
        ve=np.where(Vsum>x)[0] #Quite sensible parameter    
        Mat=FragNetIntRes(DF.loc[ve],MinTres=60)
      #  print(len(Mat))
        red[x]=len(Mat)
        print(red)
    sf=np.where(red>0)[0]
   # print(red)
    minC=np.where(red==min(red[sf]))[0]  
    return minC
    
def IntPos(DF):
    L=[]
    MF=list(DF.groupby(['Measured_m/z']).groups.keys())
    for x in MF:
        L.append([0,1])
    return L
    
import numpy as np
def IndexLists(DF):
    L=[]
    MF=list(DF.groupby(['Measured_m/z']).groups.keys())
    for x in MF:
        IFDFloc=DF['Measured_m/z']==x
        IFDF=DF.loc[IFDFloc]
        vecind=np.array(IFDF.index)
      #  vecind=np.append(vecind)
        L.append(vecind)
    return L
    
import numpy as np
def GradeNet(NetF,D,MinGrade=0,MinGradeCut=0): #I should include the explained intensity as well
    #NetF[1]=NetF[1].drop_duplicates()
    #Good time to include penalization
    #This one takes too long, I should change the use of DF for list or array
    #IntVec=np.array(list(DF.groupby(['RelInt']).groups.keys()))
    Dmat=np.array(D)
    Lnet=len(NetF)
    for xL in np.arange(Lnet):
        locNet=np.where(NetF[xL,:]>-1)[0]
        locD=np.array(NetF[xL,locNet],dtype=int)
        Dspec=(Dmat[locD,:].copy())[:,locD]        
        grade=np.sum(Dspec)
        NetF[xL,-1]=grade 
    return NetF
    
import numpy as np
import pandas as pd
from NewReduceSpecFindPeaks import *
def GetMS2forFeature(experiment,MM,RT,error=5,errorT=3):
    c1=0
    sN=0
    while True:
        try:
            for spectrum in experiment:
                
                MSl=spectrum.getMSLevel()
                #print(MSl)
                if MSl==2:
                   # print(abs(spectrum.getPrecursors()[0].getMZ()))
                    if abs(spectrum.getPrecursors()[0].getMZ()-MM)/MM*1e6<error and abs(spectrum.getRT()-RT)<errorT:
                        peaks=np.array(spectrum.get_peaks()).T
                      #  print(abs(spectrum.getRT()-RT))
                        if sN==0:
                            Peak=peaks.copy()
                            
                        else:
                            Peak=np.append(Peak,peaks,axis=0)
                        sN+=1
                c1=c1+1
            if sN<1:
                return 0
            PeakN=pd.DataFrame(Peak,columns=['m/z','Intensity'])
            PeakN=PeakN.sort_values(by='m/z')
           # print('CP')
            P=NewReduceSpecFindPeaks(peaks=PeakN)
           # print('CP2')            
            break
        except:
            print('Error extracting MS2')
            return 0
    return P
    
import numpy as np
from ShowDF import *
#this would be a different commit
def GetIntVec(DF):
    MF=list(DF.groupby(['Measured_m/z']).groups.keys())
   # ShowDF(DF)
    IntL=[]
    for x in MF:
        DFtloc=DF['Measured_m/z']==x
        DFt=DF.loc[DFtloc]
        DFtind=DFt.index[0]
        IntL.append(DF.loc[DFtind]['RelInt'])
    IntVec=np.array(IntL,dtype=float)
    return IntVec
    
from GetMS2forFeature import *
from MoleculesCand import *
import numpy as np
def FragSpacePos(experiment,MM,RT,ExpectedV={'K':1,'Na':1,'C13':1,'C':40,'Cl':1,'S34':1,'S':3,'P':1,'F':1,'O':20,'N':20,'H':100}):
    Mat=GetMS2forFeature(experiment=experiment,MM=MM,RT=RT)
  #  ShowDF(Mat)
    if type(Mat)==type(0):
        return 0
  #  ShowDF(Mat)
    c=0
    L=len(Mat)   
    DF=0
    for ind in Mat.index:
        x=Mat.loc[ind]['Mean_m/z']
        RelInt=Mat.loc[ind]['RelInt']        
        Confidence=Mat.loc[ind]['ConfidenceInterval(ppm)']
        re=MoleculesCand(TargetM=x,RelInt=RelInt,ExpectedV=ExpectedV,Tres=Confidence)
       # ShowDF(re)
       # print(re)
        if type(re)!=type(0):   
            if c==0:
                DF=re
            else:
                DF=DF.append(re)
            c+=1
    if type(DF)==type(0):
        print('error')
        return 0
    DF.index=np.arange(len(DF.index))
    return DF
    
import numpy as np
from GetIntVec import *
from IntPos import *
from FragNet import *
def FragNetIntRes(DF,MinTres=90):
    IntVec=GetIntVec(DF)
   # print(IntVec)
    FragIntFake=IntPos(DF)
    LIntVec=len(IntVec)
    OnesV=np.ones(LIntVec)
    MatInt=np.array(FragNet(FragIntFake,OnesV))
    IntExplained=np.matmul(MatInt[:,:-1],IntVec)
    select=np.where(IntExplained>MinTres)[0]
    return MatInt[select,:]
    
    
import numpy as np
def FragNet(DFind,Lv,p=0,ip=0,vec=[],Mat=[],start=True):
    PosLoc=np.where(Lv==1)[0]
    LMF=len(PosLoc)
    LiMF=len(DFind[PosLoc[p]])    
    #print('p:',p,'ip:',ip)    
    if start:        
        vec=-np.ones(len(Lv)+1)
        Mat=[]       
    vec[PosLoc[p]]=int(DFind[PosLoc[p]][ip])
    #print(vec)
    if p<LMF-1:        
        Mat=FragNet(DFind,Lv,p=p+1,ip=0,vec=vec.copy(),Mat=Mat,start=False)   
    if ip<LiMF-1:                
        Mat=FragNet(DFind,Lv,p=p,ip=ip+1,vec=vec.copy(),Mat=Mat,start=False)  
    if p==LMF-1:
       # print(p,ip,'\n')
        Mat.append(vec)        
    
       # print(vec)
    return Mat     
    
    
#Checking if I keep track of everything
def Formula(DF):
    AllFor=[]
    for x in DF.index:
        For=''
        for y in DF.columns:
            v=DF.loc[x,y]        
            if v>1:
                For+=y+str(int(v))
            elif v>0:
                For+=y
        AllFor.append(For)
   # display(DF)
    #print(AllFor)
    DF['Formula']=AllFor 
    #return AllFor
    
import numpy as np
from MoleculesCand import *
def FitFragment(DF,D,Frag1,Frag2,Mat=[]):
    MT=abs(Frag2-Frag1)
    LocId1=DF['Measured_m/z']==Frag1
    DF1=DF.loc[LocId1]
    LocId2=DF['Measured_m/z']==Frag2
    DF2=DF.loc[LocId2]
   # print(Frag1,Frag2)
    Tre1=DF.loc[DF1.index[0]]['ConfidenceInterval(ppm)']
    Tre2=DF.loc[DF2.index[0]]['ConfidenceInterval(ppm)']
    #Tre2=DF.loc[LocId2,'ConfidenceInterval(ppm)']
  #  print('l',Tre1,Tre2)
   # Tres=max(Tre1,Tre2)
    Tres=10
    #print(MT)
    re= MoleculesCand(TargetM=MT,ExpectedV={'K':0,'Na':0,'C13':0,'C':15,'Cl':0,'S34':0,'S':1,'P':0,'F':0,'O':6,'N':5,'H':15},Tres=Tres)    
    if type(re)==type(0):
        return D
  #  print(MT)
   # ShowDF(re)  
    
    for it1 in DF1.index:
        V1=np.array(DF.loc[it1][['K','Na','C13','C','Cl','S43','S','P','F','O','N','H']])
        for it2 in DF2.index:
            V2=np.array(DF.loc[it2][['K','Na','C13','C','Cl','S43','S','P','F','O','N','H']])            
            for z in re.index:
                Vz=np.array(re.loc[z][['K','Na','C13','C','Cl','S43','S','P','F','O','N','H']])   
                if int(sum(abs(abs(V2-V1)-Vz)))==0:
                    D.loc[it1][it2]=1
                    D.loc[it2][it1]=1
                    Mat.append(Vz)
    return D
    
import numpy as np
def ExactMassCal(X):   
    #K-Cl-S-P-Na-F-O-N-C-H
    MassDic={'K':38.9637064875,'Na':22.989769282019,'C13':13.003354835336252,'C':12,'Cl':34.968852694,'S34':33.967867015,'S':31.972071174414,'P':30.97376199867,'F':18.998403227,'O':15.994914619257319,'N':14.003074004251241,'H':1.00782503189814}   
    Mass=np.array(list(MassDic.values()),dtype=float)
    v=sum(X*Mass)
    return v
    
from FragSpacePos import *
#Test with branch
from SelfConsistFrag import *
from MinEdges import *
from FragNetIntRes import *
from IndexLists import *
from AllNet import *
from GradeNet import *
import numpy as np
from ShowDF import *
import datetime
import os
def AnotateSpec(experiment,MM,RT,name='',Save=True,ExpectedV={'K':1,'Na':1,'C13':1,'C':40,'Cl':1,'S34':1,'S':3,'P':1,'F':1,'O':20,'N':20,'H':100}):
    DF=FragSpacePos(experiment=experiment,MM=MM,RT=RT,ExpectedV=ExpectedV)
    print(os.getcwd())
    #ShowDF(DF)
    if name=='':
    	name=str(datetime.datetime.now())[:19].replace(' ','_')
    #ShowDF(DF)
    if type(DF)==type(0):
        return 0
    D=SelfConsistFrag(DF)
   # ShowDF(D)
    Vsum=np.array(D.sum())
  #  print(Vsum)
    minC=MinEdges(DF,Vsum)
    ve=np.where(Vsum>minC)[0] #Quite sensible parameter   
  #  print(ve)
    Mat=FragNetIntRes(DF.loc[ve],MinTres=80)
    DFind=IndexLists(DF.loc[ve])  
   # print(len(Mat))
    AllPosNet=AllNet(DFind,Mat)
    vt=GradeNet(AllPosNet.copy(),D)
    locF=np.where(vt[:,-1]==max(vt[:,-1]))[0]
    locC=np.where((vt[locF,:-1][0]>-1))
    AnSpec=DF.loc[vt[locF,locC][0]]
    AnSpec.index=AnSpec['Formula']
    if Save:
    	AnSpec.to_csv(name+'.csv')
    return AnSpec
    
import numpy as np
from FragNet import *
def AllNet(DFind,Mat):
    c=True
    for x in Mat:
        Lv=x[:-1]
        if c:
            AllPosNet=np.array(FragNet(DFind,Lv)) 
            c=False
        else:
            AllPosNet=np.append(AllPosNet,np.array(FragNet(DFind,Lv)),axis=0)
    return AllPosNet    
    
                                                                                            
