#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 1 Aug 2019

@author: muratelhuseyni
"""

from gurobipy import *
import numpy as np 
import pandas as pd
import time
import itertools as it

size = 3 #of problem size(small,med,med+)

#to fix problem, fix ranges: Ex:typ range(1,2), probrange(3,4)-> med 1 prob 3
#range(a,b)-> from a to b-1

#for typ in range(size): 
for typ in range(3,4):  
 
 probtime = 0       
 input_file2='C:\\Users\\labuser\\Desktop\\python\\alpha input.xlsx'
   
 remainingalphas = pd.read_excel (input_file2,header = None, sheet_name='alphas')
 alphas=np.array(remainingalphas)
 
 if typ<2:
     inst = 10 #of problems
 elif typ == 2:
     inst = 6
 else:
     inst = 4
 
 output_file3 = 'score' + str(typ) + '_Alternative.csv'
 with open(output_file3,"a") as f:   
           f.write("alp1\t" + "alp2\t" + "alp3\t"+ "score\t"+ "time\t"+ "#master\t"+ "#enum\t"+ "#detect\t"+ "#iter\n")
     
 for ind in range(len(alphas)):
 #for ind in range(1,2):
       
       alpha1 = alphas[ind][0]
       alpha2 = alphas[ind][1]
       alpha3 = alphas[ind][2]
                      
       mastermatrix = {}
       
       if typ == 0:
           output_file2 = 'alp1_' + str(round(alpha1,1)) + 'alp2_' + str(round(alpha2,1)) + 'alp3_' + str(round(alpha3,1)) + '_Alternative_Smallsummary.csv'
       elif typ == 1:
           output_file2 = 'alp1_' + str(round(alpha1,1)) + 'alp2_' + str(round(alpha2,1)) + 'alp3_' + str(round(alpha3,1)) + '_Alternative_Mediumsummary.csv'
       elif typ == 2:
           output_file2 = 'alp1_' + str(round(alpha1,1)) + 'alp2_' + str(round(alpha2,1)) + 'alp3_' + str(round(alpha3,1)) + '_Alternative_Medium+summary.csv'
       else:
           output_file2 = 'alp1_' + str(round(alpha1,1)) + 'alp2_' + str(round(alpha2,1)) + 'alp3_' + str(round(alpha3,1)) + '_Alternative_Bigsummary.csv'
              
       colscore = 0
       tottime = 0
       
       totmaster = 0
       totenum = 0
       totsub = 0
       totiter = 0
         
       with open(output_file2,"a") as f:
            f.write("Case\tTotprof\tPenalty\tRevtrans\tcolpreventstate\tTime\t#master\t#totenum\t#detect\t#iter\n") 
             
       for prob in range(1,inst+1): #1'den inst'e kadar 
       #for prob in range(1,2):
        #for prob in range(2,3): 
            #typ = 0
            #prob = 3 #-> small problem 3
            #input_file='Small_size_case3.xlsx' 
            #output_file='Small_size_case3_output.csv'
            
     #       print(prob)
      #      continue
            start = time.time()  #start timer for each problem
            slack_bus=[3] #ikisinde de ortak, burada olabilir
                    
            if typ == 0:            
                n=5 # of nodes
                ig=[1,2,5] #generator nodes
                mm=len(ig)
                not_ig=[] #non_generator nodes
                for i in range(n):
                  if i+1 not in ig:
                    not_ig.append(i+1)
                
                input_file='C:\\Users\\labuser\\Desktop\\python\\trileveldata\\Small '+str(prob)+'.xlsx'                
                output_file= 'alp1_' + str(round(alpha1,1)) + 'alp2_' + str(round(alpha2,1)) + 'alp3_' + str(round(alpha3,1)) + '_Alternative_SmallTrials.csv'
                               
                with open(output_file,"a") as f:  
                    f.write("Small"+str(prob)+"\n") 
                
                #output_file='Small '+str(prob)+'_output.csv'
            
            #if medium+ comes, apply switch case with function
            elif typ < 3:            
                n=7 # of nodes
                ig=[1,2,5,6] #generator nodes
                not_ig=[] #non_generator nodes
                for i in range(n):
                    if i+1 not in ig:
                        not_ig.append(i+1)
            
                slack_bus=[3]
                mm=len(ig)
                
                if typ == 1:
                    input_file='C:\\Users\\labuser\\Desktop\\python\\trileveldata\\Medium '+str(prob)+'.xlsx'                                
                    output_file= 'alp1_' + str(round(alpha1,1)) + 'alp2_' + str(round(alpha2,1)) + 'alp3_' + str(round(alpha3,1)) + '_Alternative_MediumTrials.csv'
                               
                    with open(output_file,"a") as f:  
                        f.write("Medium"+str(prob)+"\n")  
                else:
                    input_file='C:\\Users\\labuser\\Desktop\\python\\trileveldata\\Medium+ '+str(prob)+'.xlsx'             
                    output_file= 'alp1_' + str(round(alpha1,1)) + 'alp2_' + str(round(alpha2,1)) + 'alp3_' + str(round(alpha3,1)) + '_Alternative_Medium+Trials.csv'
                               
                    with open(output_file,"a") as f:  
                        f.write("Medium+"+str(prob)+"\n") 
                
            else:
                    n=9 # of nodes
                    ig=[1,2,5,6,9] #generator nodes
                    mm=len(ig)
                    not_ig=[] #non_generator nodes
                    for i in range(n):
                      if i+1 not in ig:
                        not_ig.append(i+1)
                    
                    input_file='C:\\Users\\labuser\\Desktop\\python\\trileveldata\\Big Problem '+str(prob)+'.xlsx'                
                    output_file= 'alp1_' + str(round(alpha1,1)) + 'alp2_' + str(round(alpha2,1)) + 'alp3_' + str(round(alpha3,1)) + '_Alternative_BigTrials.csv'
                                   
                    with open(output_file,"a") as f:  
                        f.write("Big"+str(prob)+"\n") 
                    
            
            totprof = -1
            pencost = -1
            transrevenue = -1
            initdeltaF = 0
            colprevstate = 0
            
            #continue
            df = pd.read_excel (input_file,header = None, sheet_name='Cost') #header=0 
            de = pd.read_excel (input_file,header = None, sheet_name='Demand') 
            dw = pd.read_excel (input_file,header = None, sheet_name='Bidset')
            dq = pd.read_excel (input_file,header = None, sheet_name='Pmax')
            dg = pd.read_excel (input_file,header = None, sheet_name='Fmax')
            dr = pd.read_excel (input_file,header = None, sheet_name='Y')
            
            if typ  == 1:
                dcol = pd.read_excel (input_file,header = None, sheet_name='CollusiveStrategies')
            else:
                dcol = pd.read_excel (input_file,header = None, sheet_name='Collusive')
                            
            y=np.array(dr)
            fmax1=np.array(dg)
            bid=np.array(dw)  #bidset
            c = list(df[0]) #cost
            d1=list(de[0])  #demand
            pmax1=list(dq[0])  #pmax
            count=0 
            countmaster=0
            countenum=0
            countsub=0
            #22.11.20
            col = list(np.array(dcol))
                  
            #deltaF = np.zeros([n,n])
            initprofit = np.zeros(mm)
            
            a=len(bid[0])  #for column number
            
            pmax = [a/100 for a in pmax1]
            d = [a/100 for a in d1]
            fmax = fmax1/100            
            ee=1
            
            def initDCOPF(Bid): #6.6.20 
            #        global flow_list
                    m=Model('DC-OPF')
                    LMP=np.zeros(n)
                    Phi = np.zeros(n) 
                    Psineg = np.zeros([n,n])
                    Psipos = np.zeros([n,n])
                    Profit_dcopf = np.zeros(n) 
                    
                    p=m.addVars(n,lb=0,vtype=GRB.CONTINUOUS, name="p")
                    theta=m.addVars(n,lb=-GRB.INFINITY,vtype=GRB.CONTINUOUS, name="theta") 
                    
                    for i in ig:
                        m.addConstr(-p[i-1]+pmax[i-1]>=0, name="powercap["+ str(i-1)+ "]")
                        
                    MarkClears = m.addConstrs(p[i]==quicksum(y[i][j]*(theta[i]-theta[j]) for j in range(n))+d[i] for i in range(n))
                    for i in range(n):
                        for j in range(n): 
                            if y[i][j] >0:
                                m.addConstr(fmax[i][j]-y[i][j]*(theta[i]-theta[j]) >= 0, name="deltaFmaxpos["+ str(i) + "," + str(j) + "]")
                                #m.addConstr(y[i][j]*(theta[i]-theta[j]) <= fmax[i][j], name="deltaFmaxpos["+ str(i) + "," + str(j) + "]")
                    
                    for i in range(n):
                        for j in range(n):
                            if y[i][j] >0:  
                                m.addConstr(y[i][j]*(theta[i]-theta[j])+fmax[i][j]>=0, name="deltaFmaxneg["+ str(i) + "," + str(j) + "]")
                                #m.addConstr(y[i][j]*(theta[i]-theta[j])>= -fmax[i][j], name="deltaFmaxneg["+ str(i) + "," + str(j) + "]")
                                
                    m.addConstrs(theta[i-1]==0 for i in slack_bus )
                    m.addConstrs(p[i-1]==0 for i in not_ig)
                    #m.addConstr(obj=quicksum(Bid[i-1]*p[i-1] for i in ig))
                    m.setObjective(quicksum(Bid[i-1]*p[i-1] for i in ig), GRB.MINIMIZE)
                    m.optimize()
                    
                    #Obj[tuple(Bid)]=m.objVal
                    for i in range(n):
                        LMP[i]=MarkClears[i].pi
          
                    #7.11.20
                    
                    for i in ig:
                        Phi[i-1]= m.getConstrByName("powercap["+ str(i-1)+ "]").pi
                                           
                    for i in range(n):
                        for j in range(n):
                            if y[i][j] >0:
                                Psipos[i][j] = m.getConstrByName("deltaFmaxpos["+ str(i) + "," + str(j) + "]").pi
                                Psineg[i][j] = m.getConstrByName("deltaFmaxneg["+ str(i) + "," + str(j) + "]").pi
                    
                    flow_list=[]
                    congested_list = []
                    
                    for i in range(n):
                        for j in range(i+1,n):
                            if y[i][j]>0:
                                ijflow = 100*y[i][j]*(theta[i].x-theta[j].x)
                                flow_list.append(ijflow)
                                
                                #find congested
                                if (abs(abs(ijflow)-100*fmax[i][j]) <= 0.001):
                                    congested_list.append(1)
                                else:
                                    congested_list.append(0)
                    
                    revtrans=0      
                    for i in range(n):
                        sumic = 0
                        for j in range(n):
                           if y[i][j] >0:
                               sumic += 100*y[i][j]*(theta[i].x-theta[j].x)
                        revtrans +=  LMP[i]*sumic*-1
                    
                    for i in ig:
                        power[i-1]= p[i-1].x*100
                
                    for i in ig:
                        Profit_dcopf[i-1]=(LMP[i-1]-c[i-1])*power[i-1]
            
                    return (flow_list, congested_list, revtrans,LMP,Phi,Psineg,Psipos,Profit_dcopf,power)
            
            def DCOPF(Bid, DCOPF_matrix, Power_matrix): #Bid'i tuple olarak gir 
            #        global flow_list
                    m=Model('DC-OPF')
                    LMP=np.zeros(n)
                    Bid=list(Bid) 
                    
                    Profit_dcopf=np.zeros(n)
                    #DCOPF_matrix={}
                    #Power_matrix={}
                    
                    p=m.addVars(n,lb=0,vtype=GRB.CONTINUOUS, name="p")
                    theta=m.addVars(n,lb=-GRB.INFINITY,vtype=GRB.CONTINUOUS, name="theta") 
                    m.addConstrs(-p[i-1]+pmax[i-1]>=0 for i in ig)
                    MarkClears = m.addConstrs(p[i]==quicksum(y[i][j]*(theta[i]-theta[j]) for j in range(n))+d[i] for i in range(n))
                    for i in range(n):
                        for j in range(n): 
                            if y[i][j] >0:
                                m.addConstr(y[i][j]*(theta[i]-theta[j]) <= fmax[i][j], name="deltaFmaxpos["+ str(i) + "," + str(j) + "]")
                    
                    for i in range(n):
                        for j in range(n):
                            if y[i][j] >0:  
                                m.addConstr(y[i][j]*(theta[i]-theta[j])>= -fmax[i][j], name="deltaFmaxneg["+ str(i) + "," + str(j) + "]")
                    m.addConstrs(theta[i-1]==0 for i in slack_bus )
                    m.addConstrs(p[i-1]==0 for i in not_ig)
                #    m.addConstr(obj=quicksum(Bid[i-1]*p[i-1] for i in ig))
                    m.setObjective(quicksum(Bid[i-1]*p[i-1] for i in ig), GRB.MINIMIZE)
                    m.optimize()
                    
                    #Obj[tuple(Bid)]=m.objVal
                    for i in range(n):
                        LMP[i]=MarkClears[i].pi
                    
                    for i in ig:
                        power[i-1]= p[i-1].x*100
                
                    for i in ig:
                        Profit_dcopf[i-1]=(LMP[i-1]-c[i-1])*power[i-1]
                    
                    DCOPF_matrix[tuple(Bid)]=[k for k in Profit_dcopf ]
                    Power_matrix[tuple(Bid)]=[k for k in power]
            
                    #return (DCOPF_matrix, Power_matrix, m.objVal)     
            
            def Updatecollusives(collusives,col,lambdaa,DCOPF_matrix,totenumsolve):
                
                if totenumsolve == False:
                    #collusives = []
                    for i in range(len(col)):
                        Bid = collusives[0]
                        if abs(lambdaa-col[i][2*mm]) <= 0.01: #lambda==a için
                            collusives.remove(Bid)
                            continue
                            
                        if lambdaa-0.01 > col[i][2*mm]:
                            break
                else:
                    col = {}
                    for i in range(len(collusives)):
                        Bid = collusives[i]
                        Profit = DCOPF_matrix[tuple(Bid)]
                        minval = 10000
                        for i in ig:
                            if minval > Profit[i-1]:
                               minval = Profit[i-1] 
                            
                        col[tuple(Bid)] = minval
                    
                    col_sorted =sorted(col.items(), key=lambda x: x[1], reverse=True)
                    collusives.clear()
                
                    # reach dictionary key via list
                    col_list = list(col_sorted)
                    
                    for i in range(len(col_sorted)):
                        if abs(lambdaa-col_list[i][1]) <= 0.01: 
                            continue
                        collusives.append(col_list[i][0])
                    
                return collusives    
            
            def fillcol(collusives,DCOPF_matrix):
                
                col = {}
                for i in range(len(collusives)):
                    Bid = collusives[i]
                    Profit = DCOPF_matrix[tuple(Bid)]
                    minval = 10000
                    for i in ig:
                        if minval > Profit[i-1]:
                           minval = Profit[i-1] 
                        
                    lambdaa = minval
                    col[tuple(Bid)] = lambdaa
                    
                col_sorted =sorted(col.items(), key=lambda x: x[1], reverse=True)
                collusives.clear()
                
                # reach dictionary key via list
                col_list = list(col_sorted)
                
                for i in range(len(col_sorted)):
                    collusives.append(col_list[i][0])
                
                return [col,collusives]
                    
            def DetectionAlgorithm():
                                   
                bidset={}
                for i in ig:
                    bidset[i]=[x for x in list(bid[i-1]) if x!=0]
                
                num_col_bidset=bid.shape[1] # of columns
                
                Obj={}
                
                Power_matrix={} 
                                               
                combinations = it.product(*(bidset[xx] for xx in bidset  ))
                strategy_combinations=list(combinations)
                strategy_combinations2=[]
                
                for  Bid in strategy_combinations: ##adding non-generators 0
                    Bid=list(Bid)
                    if len(Bid)<n:
                        for i in range(1,n+1):
                            if i in not_ig:
                               Bid.insert(i-1,0)  
                    strategy_combinations2.append(Bid)
                
                #initsolve = 0
                for Bid in strategy_combinations2:
                    DCOPF(Bid,DCOPF_matrix, Power_matrix)
                    #(DCOPF_matrix, Power_matrix, Obj)=DCOPF(Bid,deltaF)
                
                Nash_set={}
                     
                    
                def Nash(Bid): #Bid'i tuple olarak gir 
                    bool_nash=0
                               
                    Bid_original=tuple(list(Bid).copy())     
                    
                #    if Bid not in Nash_set  :
                    bid_index=np.zeros(n)
                    for i in ig:
                        for j in range(len(bidset[i])):
                            if Bid[i-1]==bidset[i][j] : 
                                bid_index[i-1]=j
                                break
                    for i in ig:
                        bid_index_new=bid_index.copy()
                        for j in range(len(bidset[i])):
                            if j!=bid_index[i-1] : 
                                bid_index_new[i-1]=int(j)
                                Bid=list(Bid)
                                for l in ig:
                                    Bid[l-1]=bidset[l][int(bid_index_new[l-1])]
                                Bid=tuple(Bid)
                                if DCOPF_matrix[Bid_original][i-1] +0.01 < DCOPF_matrix[Bid][i-1]:
                                    return ( Nash_set ,bool_nash)
                        
                    Nash_set[tuple(Bid_original)]=DCOPF_matrix[tuple(Bid_original)]
                    bool_nash=1
                                
                    return ( Nash_set,bool_nash)
                
                for Bid in strategy_combinations2:
                    ( Nash_set,bool_nash)=Nash(Bid)
                
                    
                r_star=[0]*n
                for Bid in Nash_set:
                    for i in ig:
                        r_star[i-1]=max(r_star[i-1],Nash_set[Bid][i-1]) 
                
                
                collusive_set=[]
                
                def collusive(Bid):
                    if tuple(Bid) not in Nash_set:
                        for i in ig:
                            if r_star[i-1]+0.01>DCOPF_matrix[tuple(Bid)][i-1]:
                                return(collusive_set)
                #                continue
                #            else:                
                        collusive_set.append(Bid)
                    return(collusive_set)
                    
                #23.12.19: Added my Murat-to be consistent with Model2_objvlaues.py    
                if len(Nash_set)>0:
                    for Bid in strategy_combinations2:
                        (collusive_set)=collusive(Bid) 
                    
                return collusive_set, DCOPF_matrix
            
            def WriteEnumeration(flow_list, congested_list, revtrans,lmp,Phi,Psineg,Psipos,Profit,power,Bid,fmax):
                
                with open(output_file,"a") as f: 
                        f.write("enumeration\t")
                        for i in ig:            
                            f.write(str(Bid[i-1])+"\t") 
                        for genco in ig:
                            f.write(str(power[genco-1])+"\t")
                        for i in range(n):                          
                            f.write(str(lmp[i])+"\t")
                        
                        for genco in ig:
                                f.write(str(Profit[genco-1])+"\t") 
                                                
#                        for i in range(len(initprofit)):
#                            f.write(str(initprofit[i])+"\t")  
                        
                        minval = 10000
                        for i in ig:
                            if minval > Profit[i-1]:
                                minval = Profit[i-1] 
                        
                        lambdaa = minval
                        f.write(str( lambdaa )+"\t") 
                        
                        #write 6.6.20                         
                        #f.write(str( revtrans )+"\t") 
                        
                        f.write("\t")
                        f.write(str( sum(Profit) )+"\t") #6.6.20
                        f.write(str( max(Profit) )+"\t") #6.6.20 
                        f.write("\t\t") 
                        
                        for i in range(len(flow_list)):
                            f.write(str(flow_list[i])+"\t")
                        
                        for i in range(len(congested_list)):
                            f.write(str(congested_list[i])+"\t")
                        
                        #deltaF
                        for i in range(n):
                            for j in range(i+1,n):  
                                if y[i][j] >0:
                                    f.write("\t")
                        
                         #fmax
                        for i in range(n):
                            for j in range(i+1,n):  
                                if y[i][j] >0:
                                    f.write(str(100*fmax[i][j])+"\t")
                        
                        f.write("\n")
                        
                return lambdaa
                
            def WriteMaster(m_Master,lmp,fmax,deltaF):
            
#-------------write master         
                with open(output_file,"a") as f: 
                    f.write("masterP\t")
                    for i in ig:
                        f.write(str(m_Master.getVarByName("Bid["+str(i-1)+"]").x)+"\t") 
                    for genco in ig:
                        power[genco-1]=m_Master.getVarByName("p["+ str(genco-1) +"]").x*100
                        f.write(str(power[genco-1])+"\t")  
                    for i in range(n):
                        f.write("\t")
            #            lmp[i] = m_Master.get_var_by_name(str("LMP_"+str(i))).solution_value
                    for genco in ig:
                        Profit[genco-1]= (lmp[genco-1]-c[genco-1])*power[genco-1]
                        f.write(str(Profit[genco-1])+"\t")
                    f.write(str( min(Profit) )+"\t")
                    f.write(str(m_Master.objVal)+"\t")       
                    f.write(str( sum(Profit) )+"\t") 
                    f.write(str( max(Profit) )+"\t") 
                    
                    for i in range(n):
                        Theta[i]=m_Master.getVarByName("theta["+ str(i) +"]").x         
                    revtrans=0     
                    for i in range(n):
                        sumic = 0
                        for j in range(n):
                           if y[i][j] >0:
                               sumic += y[i][j]*(Theta[i]-Theta[j])
                        revtrans +=  lmp[i]*sumic    
                    #a = m_Master.objective_value
                                
                    f.write(str(-100*revtrans)+"\t") 
                    
                    #calculated over lower-upper triang matrix
                    penaltycost = 0
                    for i in range(n):
                        for j in range(i+1,n):
                           if y[i][j] >0 and abs(deltaF[i][j] >= 0.00001):
                                       penaltycost+= abs(deltaF[i][j])
                    
                    f.write(str( 100*penaltycost )+"\t")
                    
                    flow_list=[] 
                    congested_list = [] 
                    for i in range(n):
                        for j in range(i+1,n):
                            if y[i][j]>0:
                                ijflow = 100*y[i][j]*(Theta[i]-Theta[j])
                                f.write(str(ijflow)+"\t")
                                flow_list.append(ijflow)                                
                                #check whether two values are equal at float valuees
                                if (abs(abs(ijflow)-100*fmax[i][j]) <= 0.001):
                                    congested_list.append(1)
                                else:
                                    congested_list.append(0)
                    
                    for i in range(len(congested_list)):
                        f.write(str(congested_list[i])+"\t") 
                    
                    ######
                    global totprof 
                    totprof = sum(Profit)                    
                    global transrevenue 
                    transrevenue = -100*revtrans
                    global pencost 
                    pencost = 100*penaltycost
                    #######
                    
                    #deltaF
                    for i in range(n):
                            for j in range(i+1,n):  
                                if y[i][j] >0:
                                    f.write(str(100*deltaF[i][j])+"\t")
                    
                    #fmax
                    for i in range(n):
                            for j in range(i+1,n):  
                                if y[i][j] >0:
                                    f.write(str(100*fmax[i][j])+"\t")
                
                    f.write("\n")
            
            def WriteSubModel(m_sub,fmax):
                
                Profit=np.zeros(n)
                power=np.zeros(n) 
                Theta=np.zeros(n)
                Bid = np.zeros(mm)
                lmp = np.zeros(n)
                Phi = np.zeros(n) 
                Psineg = np.zeros([n,n])
                Psipos = np.zeros([n,n])
                
                #Write submodel
                with open(output_file,"a") as f:       
                    subcount=0
                    f.write("detectionP\t")
                    for i in ig:            
                        Bid[subcount] = m_sub.getVarByName("Bid["+str(i-1)+"]").x
                        #Bid.append(m_sub.get_var_by_name(str("Bid_"+str(i-1))).solution_value)
                        f.write(str(Bid[subcount])+"\t") 
                        subcount += 1
                    for genco in ig:
                        power[genco-1]=m_sub.getVarByName("p["+ str(genco-1) +"]").x*100
                        f.write(str(power[genco-1])+"\t")
                    for i in range(n):
                        lmp[i] = m_sub.getVarByName("LMP["+str(i)+"]").x                            
                        f.write(str(lmp[i])+"\t")
                    #7.11.20
                    for genco in ig:
                        Phi[genco-1] = m_sub.getVarByName("phi["+str(genco-1)+"]").x   #7.11.20-get phi vals          
                        Profit[genco-1]= (lmp[genco-1]-c[genco-1])*power[genco-1]
                        f.write(str(Profit[genco-1])+"\t")        
                        
                    #f.write(str(m_sub.getVarByName(str("lambdaa")).x*100)+"\t")
                    
                    minval = 1000000
                    for i in ig:
                        if minval > Profit[i-1]:
                           minval = Profit[i-1] 
                    
                    f.write(str(minval)+"\t\t")
                    f.write(str( sum(Profit) )+"\t") 
                    f.write(str( max(Profit) )+"\t\t\t")
                    
                    for genco in range(n):
                        Theta[genco]=m_sub.getVarByName("theta["+ str(genco) +"]").x
                    
                    
                    #7.11.20-get psi vals
                    for i in range(n):
                        for j in range(n):
                            if y[i][j]>0:
                                Psineg[i][j] = m_sub.getVarByName("psineg["+ str(i) + "," + str(j) + "]").x # get values
                                Psipos[i][j] = m_sub.getVarByName("psipos["+ str(i) + "," + str(j) + "]").x # get values
                    
                    flow_list=[] 
                    congested_list = []
                    for i in range(n):
                        for j in range(i+1,n):
                            if y[i][j]>0:                                                                                                          
                                ijflow = 100*y[i][j]*(Theta[i]-Theta[j]) 
                                f.write(str(ijflow)+"\t")
                                flow_list.append(ijflow)                                
                                #check whether two values are equal at float valuees
                                if (abs(abs(ijflow)-100*fmax[i][j]) <= 0.001):
                                    congested_list.append(1)
                                else:
                                    congested_list.append(0)
                
                    for i in range(len(congested_list)): 
                        f.write(str(congested_list[i])+"\t")
                    
                    #deltaF
                    for i in range(n):
                        for j in range(i+1,n):  
                                if y[i][j] >0:
                                    f.write("\t")
                                    
                    #fmax
                    for i in range(n):
                            for j in range(i+1,n):  
                                if y[i][j] >0:
                                    f.write(str(100*fmax[i][j])+"\t")
                    
                    f.write("\n")
                 
                return [minval,lmp,Psipos,Psineg,Phi]
                    
            def ModelSub(d, pmax,a,c,n,ig,fmax,m_sub):
                
                lambdaa = m_sub.addVar(vtype=GRB.CONTINUOUS, name="lambdaa")
                theta=m_sub.addVars(n,lb=-GRB.INFINITY,vtype=GRB.CONTINUOUS, name="theta")
                LMP=m_sub.addVars(n,lb=-GRB.INFINITY,vtype=GRB.CONTINUOUS, name="LMP")
                Bid=m_sub.addVars(n,vtype=GRB.INTEGER, name="Bid")
                p=m_sub.addVars(n,lb=0,vtype=GRB.CONTINUOUS, name="p") 
                phi=m_sub.addVars(n,lb=0,vtype=GRB.CONTINUOUS, name="phi")
                zz=m_sub.addVars(n,a,lb=0,vtype=GRB.CONTINUOUS, name="zz")
                psineg=m_sub.addVars(n,n,lb=0,vtype=GRB.CONTINUOUS, name="psineg")
                psipos=m_sub.addVars(n,n,lb=0,vtype=GRB.CONTINUOUS, name="psipos")
                b=m_sub.addVars(n,a,vtype=GRB.BINARY, name="b") #attention to # of columns (ig,k)
                
                m_sub.setObjective(lambdaa, GRB.MAXIMIZE)   
                m_sub.addConstrs((p[i]-d[i]==quicksum(y[i][j]*(theta[i]-theta[j]) for j in range(n)) for i in range(n)), "marketclear")
                
                m_sub.addConstrs(lambdaa <= -p[i-1]*c[i-1]+ quicksum( bid[i-1][k]*zz[i-1,k] for k in range(a)) for i in ig)
                
                m_sub.addConstrs(quicksum(b[i-1,k] for k in range(a))==1 for i in ig)
                m_sub.addConstrs(zz[i-1,k]<= pmax[i-1]*b[i-1,k] for i in ig for k in range(a))
                m_sub.addConstrs(zz[i-1,k]<= p[i-1] for i in ig for k in range(a) )
                m_sub.addConstrs(zz[i-1,k]>= p[i-1]-pmax[i-1]*(1-b[i-1,k]) for i in ig for k in range(a) )
                m_sub.addConstrs(Bid[i-1]==quicksum(bid[i-1][k]*b[i-1,k]for k in range(a)) for i in ig)
                
                m_sub.addConstrs(Bid[i-1]-LMP[i-1]+phi[i-1]>=0 for i in ig)
                
                m_sub.addConstrs(quicksum(y[i][j]*(LMP[j]-LMP[i]) for j in range(n)) 
                +quicksum(y[i][j]*(psineg[i,j]-psipos[i,j])for j in range(n)) 
                + quicksum(y[j][i]*(psipos[j,i]-psineg[j,i]) for j in range(n))==0 for i in range(n))
                
                for i in range(n):
                    for j in range(n): 
                        if y[i][j] >0:
                                m_sub.addConstr(y[i][j]*(theta[i]-theta[j]) <= fmax[i][j], name="deltaFmaxpos["+ str(i) + "," + str(j) + "]")
            
                for i in range(n):
                    for j in range(n):
                        if y[i][j] >0:  
                                m_sub.addConstr(y[i][j]*(theta[i]-theta[j]) >= -fmax[i][j], name="deltaFmaxneg["+ str(i) + "," + str(j) + "]")
            
                m_sub.addConstrs(-p[i-1]+pmax[i-1]>=0 for i in ig)
                m_sub.addConstr(quicksum(quicksum(bid[i-1][k]*zz[i-1,k] for k in range(a)) for i in ig)==quicksum(d[i]*LMP[i] for i in range(n))- quicksum(pmax[i]*phi[i] for i in range(n)) - quicksum(fmax[i][j]*(psipos[i,j]+psineg[i,j])for i in range(n) for j in range(n))  )
                
                m_sub.addConstrs(theta[i-1]==0 for i in slack_bus )
                m_sub.addConstrs(p[i-1]==0 for i in not_ig) 
            
            def ModelMaster(d, pmax,a,c,n,ig,fmax,m_Master,lmp, Phi, Psineg, Psipos): 
                
                theta=m_Master.addVars(n,lb=-GRB.INFINITY,vtype=GRB.CONTINUOUS, name="theta")
                Bid=m_Master.addVars(n,vtype=GRB.INTEGER, name="Bid")
                p=m_Master.addVars(n,lb=0,vtype=GRB.CONTINUOUS, name="p")
                zz=m_Master.addVars(n,a,lb=0,vtype=GRB.CONTINUOUS, name="zz") 
                b=m_Master.addVars(n,a,vtype=GRB.BINARY, name="b") #attention to # of columns (ig,k)
                delta_Fpos = m_Master.addVars(n,n,lb=0,vtype=GRB.CONTINUOUS, name="delta_Fpos")
                delta_Fneg = m_Master.addVars(n,n,lb=0,vtype=GRB.CONTINUOUS, name="delta_Fneg")
                Fmaxnew = m_Master.addVars(n,n,lb=0, vtype=GRB.CONTINUOUS, name="Fmaxnew")
                
                #b=m_Master.addVars(n,a,vtype=GRB.BINARY, name="b") 
                    
                exprflow = LinExpr()
                for i in range(n):
                    for j in range(n):
                        if y[i][j] >0:
                           exprflow.add(delta_Fpos[i,j],1)
                           exprflow.add(delta_Fneg[i,j],1)
                
                m_Master.setObjective(alpha1*quicksum(p[i-1]*(lmp[i-1]-c[i-1]) for i in ig)
                                       +alpha2*exprflow
                                        + alpha3*quicksum(lmp[i]*quicksum(y[i][j]*(theta[i]-theta[j]) for j in range(n))  for i in range(n)), GRB.MINIMIZE )
                    
                for i in range(n):
                    for j in range(n): 
                        if y[i][j] >0:  
                            m_Master.addConstr(Fmaxnew[i,j] == fmax[i][j]+delta_Fpos[i,j]-delta_Fneg[i,j], name="new network capacity["+ str(i) + "," + str(j) + "]" )
                
                
                #9.11.20 #simetri
                for i in range(n):
                        for j in range(n): 
                            if y[i][j] >0 and i<j:
                                 m_Master.addConstr(Fmaxnew[i,j] == Fmaxnew[j,i])
                             
                                
                m_Master.addConstrs(quicksum(b[i-1,k] for k in range(a))==1 for i in ig)
                m_Master.addConstrs(zz[i-1,k]<= pmax[i-1]*b[i-1,k] for i in ig for k in range(a))
                m_Master.addConstrs(zz[i-1,k]<= p[i-1] for i in ig for k in range(a) )
                m_Master.addConstrs(zz[i-1,k]>= p[i-1]-pmax[i-1]*(1-b[i-1,k]) for i in ig for k in range(a) )
                m_Master.addConstrs(Bid[i-1]==quicksum(bid[i-1][k]*b[i-1,k]for k in range(a)) for i in ig)
                
                m_Master.addConstrs(Bid[i-1]-lmp[i-1]+Phi[i-1]>=0 for i in ig)
                
                #m_Master.addConstrs(quicksum(y[i][j]*(lmp[j]-lmp[i]) for j in range(n))
                 #   +quicksum(y[i][j]*(Psineg[i][j]-Psipos[i][j])for j in range(n))  
                  #  + quicksum(y[j][i]*(Psipos[j][i]- Psineg[j][i]) for j in range(n))==0 for i in range(n))
                
                for i in range(n):
                    for j in range(n):  
                        if y[i][j] >0:
                            m_Master.addConstr(y[i][j]*(theta[i]-theta[j]) <= Fmaxnew[i,j], name="deltaFmaxpos["+ str(i) + "," + str(j) + "]")
                
                for i in range(n):
                    for j in range(n): 
                        if y[i][j] >0:  
                            m_Master.addConstr(y[i][j]*(theta[i]-theta[j])>= -Fmaxnew[i,j], name="deltaFmaxneg["+ str(i) + "," + str(j) + "]" )
                      
                m_Master.addConstrs(-p[i-1]+pmax[i-1]>=0 for i in ig)
                m_Master.addConstrs(p[i]-d[i]==quicksum(y[i][j]*(theta[i]-theta[j]) for j in range(n)) for i in range(n))
                                
                m_Master.addConstr(quicksum(quicksum(bid[i-1][k]*zz[i-1,k] for k in range(a)) for i in ig)== quicksum(d[i]*lmp[i] for i in range(n))- quicksum(pmax[i]*Phi[i] for i in range(n)) - quicksum(Fmaxnew[i,j]*(Psipos[i][j]+Psineg[i][j])for i in range(n) for j in range(n)),"strongduality")
                m_Master.addConstrs(theta[i-1]==0 for i in slack_bus )
                m_Master.addConstrs(p[i-1]==0 for i in not_ig)
            
            # sub bids, profits, revtrans, subids, lmps, profits, lambda
            with open(output_file,"a") as f:  
                    f.write("subtype\t")
                    for i in ig:
                        f.write("Bid["+str(i)+"]\t")
                    for i in ig:
                        f.write("Power["+str(i)+"]\t")
                    for i in range(n):
                        f.write("lmp["+str(i+1)+"]\t")
                    for i in ig:
                        f.write("Profit["+str(i)+"]\t")
                    f.write("Lambda\t")
                    
                    f.write("Objective\t")
                    f.write("TotalProfit\t")
                    f.write("MaxProfit\t")
                    f.write("Revtrans\t")
                    f.write("Penaltycost\t")
                    
                    #flow
                    for i in range(n):
                        for j in range(i+1,n):  
                            if y[i][j] >0:
                                    f.write("flow["+str(i+1)+","+str(j+1)+"]\t")
                    
                    #congested
                    for i in range(n):
                        for j in range(i+1,n):  
                            if y[i][j] >0:
                                    f.write("cong["+str(i+1)+","+str(j+1)+"]\t")
                    
                    #deltaF
                    for i in range(n):
                        for j in range(i+1,n):  
                            if y[i][j] >0:
                                    f.write("deltaF["+str(i+1)+","+str(j+1)+"]\t")
                    
                    #fmax
                    for i in range(n):
                        for j in range(i+1,n):  
                            if y[i][j] >0:
                                    f.write("fmax["+str(i+1)+","+str(j+1)+"]\t")
                    
                    f.write("Time\t")
                    f.write("\n")
                                        
            last = 10    
            fijzero = True
            lambdaa=0
            mastersolve = False
            Fchangenotsolve = False
            DCOPF_matrix={}
            totenumsolve = False
            subsolve = False
            
            collusives = np.zeros([len(col),mm])
            for i in range(len(col)):
                for j in range(mm):
                    collusives[i][j] = col[i][j]
                    
            collusives = list(collusives)
            flow_list=[]
            congested_list = []
            deltaF = np.zeros([n,n])
            Profit=np.zeros(n)
            power=np.zeros(n) 
            Theta=np.zeros(n)
            Bid = np.zeros(mm)
                
            lmp = np.zeros(n)
            Phi = np.zeros(n) 
            Psineg = np.zeros([n,n])
            Psipos = np.zeros([n,n])
            
            while count<last:                
                
                if subsolve==False: #burada max collusive'in lmplerını datadan çekersin
                    # bid, profit, lmp written in dcol
                    Bid = collusives[0]                    
                    # 7.11.20
                    Bid=list(Bid)
                    if len(Bid)<n:
                        for i in range(1,n+1):
                            if i in not_ig:
                               Bid.insert(i-1,0)
               
                    #initsolve = 1
                    (flow_list, congested_list, revtrans,lmp,Phi,Psineg,Psipos,Profit,power) = initDCOPF(Bid)
                    lambdaa = WriteEnumeration(flow_list, congested_list, revtrans,lmp,Phi,Psineg,Psipos,Profit,power,Bid,fmax)
                                     
                m_Master=Model("Master")   
                #modify m_Master objective with lmp          
                ModelMaster(d, pmax,a,c,n,ig,fmax,m_Master,lmp, Phi, Psineg, Psipos) 
                #feasib tolerance
                m_Master.Params.FeasibilityTol = 1e-5                
                m_Master.optimize()   
                mastersolve=True
                fijzero = True
                countmaster += 1
                #reset deltaF
                deltaF = np.zeros([n,n])
                                          
                for i in range(n):
                    for j in range(n):
                       if y[i][j] >0:               
                               deltapos = m_Master.getVarByName("delta_Fpos["+ str(i) + "," + str(j) + "]").x
                               deltaneg = m_Master.getVarByName("delta_Fneg["+ str(i) + "," + str(j) + "]").x
                               deltaF[i][j] =  deltapos-deltaneg  
                               if abs(deltaF[i][j]) >= 0.00001:
                                   fijzero = False
                                   fmax[i][j] = deltaF[i][j] + fmax[i][j]   
                                                                
                WriteMaster(m_Master,lmp,fmax,deltaF) 
                #m_Master.end() 
                m_Master.remove(m_Master.getVars())
                m_Master.remove(m_Master.getConstrs())
                
                if fijzero == False:                         
                    
                    count = count + 1                    
                    m_sub=Model("Subproblem")
                    ModelSub(d, pmax,a,c,n,ig,fmax,m_sub) 
                    m_sub.optimize() 
                    subsolve = True
                    countsub += 1
                    
                    status = m_sub.status                
                    if status == 3 or status == 4:
                            break
                                                                             
                    [detectlambda,lmp,Psipos,Psineg,Phi]  = WriteSubModel(m_sub,fmax)
                                                                                                                                      
                    if detectlambda <= 1e-2:
                        colprevstate = 1
                        break
                                  
                    m_sub.remove(m_sub.getVars())
                    m_sub.remove(m_sub.getConstrs())
                    
                                                                                  
                else:#fijzero=true 
                    if subsolve == True:
                        count = count + 1
                        
                        [collusives,DCOPF_matrix] = DetectionAlgorithm() #olması gereken    
                        totenumsolve = True
                        subsolve = False
                        mastersolve = False
                        countenum += 1
                        if len(collusives) == 0: 
                                 colprevstate = 1     
                                 break 
                        #fill col
                        [col,collusives] = fillcol(collusives,DCOPF_matrix)
                    
                    else:                       
                        collusives = Updatecollusives(collusives,col,lambdaa,DCOPF_matrix,totenumsolve) #olması gereken  
                        if len(collusives) == 0:
                                 colprevstate = 0
                                 break

                if count == last:
                    colprevstate = 0
                    #mastermatrix[prob] = [totprof,pencost,transrevenue,initdeltaF,colprevstate]
            
            end = time.time()
            sure = end-start
            with open(output_file,"a") as f:                      
                    f.write("\t")
                    for i in ig:
                        f.write("\t")
                    for i in ig:
                        f.write("\t")
                    for i in range(n):
                        f.write("\t")
                    for i in ig:
                        f.write("\t")
                    for i in range(6):
                        f.write("\t")
                    #flow
                    for i in range(n):
                        for j in range(i+1,n):  
                            if y[i][j] >0:
                                    f.write("\t")                    
                    #congested
                    for i in range(n):
                        for j in range(i+1,n):  
                            if y[i][j] >0:
                                    f.write("\t")   
                    
                    #deltaF
                    for i in range(n):
                        for j in range(i+1,n):  
                            if y[i][j] >0:
                                    f.write("\t")
                    
                    #fmax
                    for i in range(n):
                        for j in range(i+1,n):  
                            if y[i][j] >0:
                                    f.write("\t")
                                                            
                    f.write(str(sure)+"\n")                    
            
            mastermatrix[prob] = [totprof,pencost,transrevenue,colprevstate,sure,countmaster, countenum, countsub,count]
            #if prob == 10:       
            colscore = colscore + colprevstate
            probtime = probtime + sure
            tottime = tottime + sure
            totmaster += countmaster
            totenum += countenum
            totsub += countsub
            totiter += count
                     
            with open(output_file2,"a") as f: 
                f.write("Case"+str(prob) + "\t")
                for j in range(len(mastermatrix[prob])):                   
                    f.write(str(mastermatrix[prob][j]) + "\t")
                f.write("\n")
                        
       with open(output_file3,"a") as f:   
                    f.write(str(round(alpha1,1)) + '\t' + str(round(alpha2,1)) + '\t' + str(round(alpha3,1)) + "\t"+ str(colscore) + "\t"+ str(tottime)+ 
                            "\t"+ str(totmaster)+"\t"+ str(totenum)+ "\t"+ str(totsub)+"\t"+ str(totiter)+"\n")