#Coding the non-linear double component solution : analytical solution and aux
#Mohammad Afzal Shadab
#Date modified: 06/05/21

# clear all 
from IPython import get_ipython
get_ipython().magic('reset -sf') #for clearing everything
get_ipython().run_line_magic('matplotlib', 'qt') #for plotting in separate window
from matplotlib.markers import MarkerStyle

# import python libraries
import numpy as np
import scipy.sparse as sp
import scipy.special as sp_special
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
import matplotlib.animation as animation
from matplotlib.ticker import LinearLocator, FormatStrFormatter
plt.rcParams.update({'font.size': 20})
plt.rcParams.update({'font.family': 'Serif'})
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fsolve, least_squares


import sys
from matplotlib import colors


#Colors
light_gray = [0.85,0.85,0.85]
blue       = [ 30/255 ,144/255 , 255/255 ]
light_red  = [0,0,0]
light_blue = [0.0,0,0]
light_black= [0.5,0.5,0.5]

class Grid:
    def __init__(self):
        self.xmin = []
        self.xmax = []
        self.Nx   = []

class BC:
    def __init__(self):
        self.dof_dir = []
        self.dof_f_dir = []
        self.g = []
        self.dof_neu = []
        self.dof_f_neu = []
        self.qb = []

def h_w(H,C):
    hw = np.zeros_like(H)  #Region 1: ice + gas
    hw[H>0] = 1.0         #Region 2: ice + water + gas
    hw[H>C] = H[H>C]/C[H>C]#Region 3: water + gas
    return hw


def f_H(H,C,m,n):
    fH = np.zeros_like(H)         #Region 1: ice + gas
    fH[H>0] = (1-C[H>0]+H[H>0])**m * (H[H>0]/(1-C[H>0]+H[H>0]))**n  #Region 2: ice + water + gas
    fH[H>C] = H[H>C]*C[H>C]**(n-1)#Region 3: water + gas
    
    fH[C==1] = H[C==1]**m     #Region 4: single component region
    fH[C>1]  = 0.0       #Region 4: outer region             
    return fH

def f_Hm(H,C,m,n):
    fH = np.zeros_like(H)         #Region 1: ice + gas
    fH[H>0] = (1-C[H>0]+H[H>0])**m  #Region 2: ice + water + gas
    fH[H>C] = 1.0#Region 3: water + gas
    
    fH[C==1] = H[C==1]**m     #Region 4: single component region
    fH[C>1]  = 0.0       #Region 4: outer region             
    return fH

def f_Hn(H,C,m,n):
    fH = np.zeros_like(H)         #Region 1: ice + gas
    fH[H>0] = (H[H>0]/(1-C[H>0]+H[H>0]))**n  #Region 2: ice + water + gas
    fH[H>C] = H[H>C]*C[H>C]**(n-1)#Region 3: water + gas
    
    fH[C==1] = 1.0    #Region 4: single component region
    fH[C>1]  = 0.0       #Region 4: outer region             
    return fH


def f_C(H,C,m,n):
    fC = np.zeros_like(H)          #Region 1: ice + gas
    fC[H>0] = (1-C[H>0]+H[H>0])**m * (H[H>0]/(1-C[H>0]+H[H>0]))**n        #Region 2: ice + water
    fC[H>C] = C[H>C]**(n)#Region 3: water + gas

    fC[C==1] = H[C==1]**m            #Region 4: single component region
    fC[C>1]  = 0.0       #Region 4: outer region   
    return fC

def f_Cm(H,C,m,n):
    fC = np.zeros_like(H)          #Region 1: ice + gas
    fC[H>0] = (1-C[H>0]+H[H>0])**m        #Region 2: ice + water
    fC[H>C] = 1.0#Region 3: water + gas

    fC[C==1] = 0.0#H[C==1]**m            #Region 4: single component region
    fC[C>1]  = 0.0       #Region 4: outer region   
    return fC

def f_Cn(H,C,m,n):
    fC = np.zeros_like(H)          #Region 1: ice + gas
    fC[H>0] = (H[H>0]/(1-C[H>0]+H[H>0]))**n        #Region 2: ice + water
    fC[H>C] = C[H>C]**(n)#Region 3: water + gas

    fC[C==1] = 1.0            #Region 4: single component region
    fC[C>1]  = 0.0       #Region 4: outer region   
    return fC

def phi_w(H,C):
    phiw = np.zeros_like(H)#Region 1: all ice
    phiw[H>0] = H[H>0]     #Region 2: ice + water
    phiw[H>=C]= C[H>=C]    #Region 3: all water
    return phiw

def phi_i(H,C):
    phii = np.zeros_like(H)#Region 3: water + gas
    phii[H<=C]= C[H<=C] - H[H<=C] #Region 2: water + ice + gas
    phii[H<=0]= C[H<=0]     #Region 1: ice + water
    return phii

def phi_g(H,C):
    phig = 1 - C
    return phig

def porosity(H,C):
    por = phi_w(H,C) + phi_g(H,C)
    return por

def saturation(H,C):
    Sw = phi_w(H,C)/(1 - phi_i(H,C))
    Sg = phi_g(H,C)/(1 - phi_i(H,C))    
    return Sw, Sg

def T(H, C, Ste, Cpr):
    T = np.zeros_like(H)                
    T[H<0] = H[H<0] / (C[H<0] *Ste * Cpr)        #Region 1: all ice
    T[H>C] = (H[H>C]/C[H>C]-1)/Ste  #Region 2 & 3: ice + water or all water
    return T

def lambda_1(H, C, m, n):
    lambda1 = np.zeros_like(H)                
    lambda1[H>C] = C[H>C]**(n-1)  #Region 3: gas+ water
    return lambda1

def lambda_2(H, C, m, n):
    lambda2 = np.zeros_like(H)                
    lambda2[H>0] = n*H[H>0]**(n-1) * (1-C[H>0]+H[H>0])**(m-n) #Region 2: water + gas +ice
    lambda2[H>C] = n*C[H>C]**(n-1)  #Region 3: gas+ water
    return lambda2

def int_region3_curves_lambda1(u0,H, m, n): # water + gas
    C = u0[0,:]            #calculating the integrating constant
    C = C*np.ones_like(H)  #calculating the integral curves
    return C

def int_region3_curves_lambda2(u0,H, m, n):
    C = u0[0,:]/u0[1,:]  #calculating the integrating constant
    C = C*H              #calculating the integral curves
    return C

def int_curves_lambda1(u0,H, m, n):
    C = u0[1,:]**(n/(m-n))*(u0[0,:]-(1+u0[1,:]))  #calculating the integrating constant
    C = 1 + H + C*H**(-n/(m-n))   #calculating the integral curves
    return C

def int_curves_lambda2(u0,H, m, n):
    C = u0[0,:]-u0[1,:]  #calculating the integrating constant
    C = H + C            #calculating the integral curves
    return C

def plotting(simulation_name,H_plot,C_plot,m,n):
    H = np.linspace(0,1,10000)
    
    norm = colors.Normalize(vmin=0, vmax=1)
    fig = plt.figure(figsize=(7,10), dpi=100)
    ax = fig.add_subplot(111)
    
    #fancy plots
    #light_red  = [1.0,0.5,0.5]
    #light_blue = [0.5,0.5,1.0]
    #light_black= [0.5,0.5,0.5]
    light_red  = [0,0,0]
    light_blue = [0,0,0]

    #Region 2 Three phase region
    C1 = int_curves_lambda2(np.array([[0.05],[0.0]]),H, m, n)   #u0 = [C,H]^T
    C2 = int_curves_lambda2(np.array([[0.2],[0]]),H, m, n) 
    C3 = int_curves_lambda2(np.array([[0.4],[0]]),H, m, n) 
    C4 = int_curves_lambda2(np.array([[0.6],[0]]),H, m, n) 
    C5 = int_curves_lambda2(np.array([[0.8],[0]]),H, m, n) 
    
    C1_2 = int_curves_lambda1(np.array([[0.05],[0.05]]),H, m, n)   #u0 = [C,H]^T
    C2_2 = int_curves_lambda1(np.array([[0.2],[0.2]]),H, m, n)
    C3_2 = int_curves_lambda1(np.array([[0.4],[0.4]]),H, m, n) 
    C4_2 = int_curves_lambda1(np.array([[0.6],[0.6]]),H, m, n) 
    C5_2 = int_curves_lambda1(np.array([[0.8],[0.8]]),H, m, n) 
    
    #removing the curves outside the region
    C1_2[C1_2<H] = np.nan
    C2_2[C2_2<H] = np.nan
    C3_2[C3_2<H] = np.nan
    C4_2[C4_2<H] = np.nan
    C5_2[C5_2<H] = np.nan
    

    plt.plot(C1_2,H,'--',c=light_blue,label=r'Slow path')
    plt.plot(C2_2,H,'--',c=light_blue)
    plt.plot(C3_2,H,'--',c=light_blue)
    plt.plot(C4_2,H,'--',c=light_blue)
    plt.plot(C5_2,H,'--',c=light_blue)
    plt.plot(C1,H,'-',c=light_red,label=r'Fast path')
    plt.plot(C2,H,'-',c=light_red)
    plt.plot(C3,H,'-',c=light_red)
    plt.plot(C4,H,'-',c=light_red)
    plt.plot(C5,H,'-',c=light_red)
    
    ###Ice lens formation region
    pp3 = plt.Polygon([[1, 0],
                   [1, -0.3],
                   [0.7, -0.3]],facecolor='black', alpha=0.1)
  
    # depict illustrations
    ax.add_patch(pp3)
    ###
    
    plt.plot([0,1],[0,1],'k--',alpha=0.3)
    plt.plot([0,1],[0,0],'k--',alpha=0.3)
    plt.plot([1,1],[1,0],'k--',alpha=0.3)
    
    plt.ylim([-0.2,1])
    plt.xlim([0,1.001])
    plt.plot(C_plot[0],H_plot[0],'ko', markersize=12)
    plt.plot(C_plot[len(C_plot)-1],H_plot[len(C_plot)-1],'ko', markersize=12)
    plt.plot(C_plot,H_plot,'k-',linewidth=2,label='Path')
    plt.ylabel(r'Dim-less Enthalpy $\mathcal{H}$')
    plt.xlabel(r'Dim-less Composition $\mathcal{C}$')
    #plt.ylabel(r'$\mathcal{H}$')
    #plt.xlabel(r'$\mathcal{C}$')
    plt.legend(loc='upper left',borderaxespad=0.)


    #adding arrow
    x_pos1 = C_plot[int(len(H_plot)/4)-1]
    y_pos1 = H_plot[int(len(H_plot)/4)-1]
    x_direct1 = C_plot[int(len(H_plot)/4)]-C_plot[int(len(H_plot)/4)-1]
    y_direct1 = H_plot[int(len(H_plot)/4)]-H_plot[int(len(H_plot)/4)-1]
    #plt.arrow(x_pos1, y_pos1, x_direct1, y_direct1,head_width=0.04, head_length=0.04, fc='k', ec='k')
    
    x_pos2 = C_plot[int(len(H_plot)*2/3)-1]
    y_pos2 = H_plot[int(len(H_plot)*2/3)-1]
    x_direct2 = C_plot[int(len(H_plot)*2/3)+1]-C_plot[int(len(H_plot)*2/3)-1]
    y_direct2 = H_plot[int(len(H_plot)*2/3)+1]-H_plot[int(len(H_plot)*2/3)-1]
    #plt.arrow(x_pos2, y_pos2, x_direct2, y_direct2,head_width=0.04, head_length=0.04, fc='k', ec='k')
        

    plt.savefig(f'../Figures/{simulation_name}_integral_curves.pdf')


def analytical(case_no,etaL,etaR,C_L,H_L,C_R,H_R, m, n):
    
    eta  = np.linspace(etaL,etaR,20000)
    
    if case_no==1:
        print('Case 1: Contact Discontinuity only')
        C = C_L*np.ones_like(eta)  
        H = H_L*np.ones_like(eta)       
        C[eta >0.0] = C_R
        H[eta >0.0] = H_R   
        
        H_plot = np.linspace(H_L,H_R,20000)
        C_plot = int_curves_lambda1(np.array([[C_L],[H_L]]),H_plot, m, n)        
    
    elif case_no==2:
        print('Case 2: Rarefaction only')
        C = C_L*np.ones_like(eta)  
        H = H_L*np.ones_like(eta)             
        
        lambda_2L = lambda_2(np.array([H_L]),np.array([C_L]), m, n)[0] #left state
        lambda_2R = lambda_2(np.array([H_R]),np.array([C_R]), m, n)[0] #right state
        
        #H[eta<lambda_2L] = (H_L**(n-1) + (eta[eta<lambda_2L]-lambda_2L)/(n*(1-C_L+H_L)**(m-n)))**(1/(n-1))
        H[eta>=lambda_2L] = (eta[eta>=lambda_2L]/(n*(1-C_L+H_L)**(m-n)))**(1/(n-1))
        C[eta>=lambda_2L] = C_L - H_L + H[eta>=lambda_2L]
        
        H[eta>=lambda_2R] = H_R
        C[eta>=lambda_2R] = C_R  

        print(lambda_2L,lambda_2R)

        H_plot = np.linspace(H_L,H_R,20000)
        C_plot = int_curves_lambda2(np.array([[C_L],[H_L]]),H_plot, m, n) 

    elif case_no==3:
        print('Case 3: Shock only')
        C = C_L*np.ones_like(eta)  
        H = H_L*np.ones_like(eta)             
        
        s = (f_C(np.array([H_L]),np.array([C_L]),m,n)[0]-f_C(np.array([H_R]),np.array([C_R]),m,n)[0])/(C_L - C_R)#Shock speed
        
        H[eta>=s] = H_R
        C[eta>=s] = C_R
        
        H_plot = np.linspace(H_L,H_R,20000)
        C_plot = int_curves_lambda2(np.array([[C_L],[H_L]]),H_plot, m, n) 
        
    elif case_no==4:
        print('Case 4: 1-Contact Discontinuity, 2-Rarefaction')
        C = C_L*np.ones_like(eta)  
        H = H_L*np.ones_like(eta)             
        
        phi_L = porosity(np.array([H_L]),np.array([C_L]))[0]
        phi_R = porosity(np.array([H_R]),np.array([C_R]))[0]
        
        func = lambda H_I: (phi_R - H_I**(-1/(n-1))*H_L**(n/(n-1))*(phi_L/phi_R)**((m-n)/(n-1))+H_I-phi_L*(H_I/H_L)**(-n/(m-n)))
        #H_I  = fsolve(func,(H_L+H_R)/2)
        H_I  = least_squares(func,(H_L+H_R)/2,jac='2-point',bounds=(0, np.inf)).x[0]
        C_I  = 1 + H_I - (1 + H_L - C_L)*(H_I/H_L)**(-n/(m-n))

        H[eta>=0] = H_I
        C[eta>=0] = C_I
        
        lambda_2I = lambda_2(np.array([H_I]),np.array([C_I]), m, n)[0] #intermediate state
        lambda_2R = lambda_2(np.array([H_R]),np.array([C_R]), m, n)[0] #right state
        
        H[eta>=lambda_2I] = (eta[eta>=lambda_2I]/(n*(1-C_R+H_R)**(m-n)))**(1/(n-1))
        C[eta>=lambda_2I] = C_R - H_R + H[eta>=lambda_2I]
        
        H[eta>=lambda_2R] = H_R
        C[eta>=lambda_2R] = C_R        

        H_plot1 = np.linspace(H_L,H_I,10000)
        C_plot1 = int_curves_lambda1(np.array([[C_L],[H_L]]),H_plot1, m, n) 
        H_plot2 = np.linspace(H_I,H_R,10000)
        C_plot2 = int_curves_lambda2(np.array([[C_R],[H_R]]),H_plot2, m, n) 
        C_plot  = np.append(C_plot1,C_plot2)
        H_plot  = np.append(H_plot1,H_plot2)

    elif case_no==5:
        print('Case 5: 1-Contact Discontinuity, 2-Shock')
        C = C_L*np.ones_like(eta)  
        H = H_L*np.ones_like(eta)             
        
        H_I = H_L*((1+H_L-C_L)/(1+H_R-C_R))**((m-n)/n)
        C_I = H_I + (C_R-H_R)
        
        H[eta>=0] = H_I
        C[eta>=0] = C_I
        
        s = (f_C(np.array([H_I]),np.array([C_I]),m,n)[0]-f_C(np.array([H_R]),np.array([C_R]),m,n)[0])/(C_I - C_R)#Shock speed
        
        H[eta>=s] = H_R
        C[eta>=s] = C_R
        
        H_plot1 = np.linspace(H_L,H_I,10000)
        C_plot1 = int_curves_lambda1(np.array([[C_L],[H_L]]),H_plot1, m, n) 
        H_plot2 = np.linspace(H_I,H_R,10000)
        C_plot2 = int_curves_lambda2(np.array([[C_R],[H_R]]),H_plot2, m, n) 
        C_plot  = np.append(C_plot1,C_plot2)
        H_plot  = np.append(H_plot1,H_plot2)


    elif case_no==6:
        print('Case 6: Contact Discontinuity only (Region 1: ice + gas)')
        C = C_L*np.ones_like(eta)  
        H = H_L*np.ones_like(eta)       
        C[eta >0.0] = C_R
        H[eta >0.0] = H_R   
        
        H_plot = np.linspace(H_L,H_R,20000)
        C_plot = C_L+ (C_L-C_R)/(H_L-H_R)*(H_plot - H_L)  
        
        
    elif case_no==7:
        print('Case 7: Region 3 (Water + gas) - Contact Discontinuity only')
        C = C_L*np.ones_like(eta)  
        H = H_L*np.ones_like(eta) 
        
        lambda1 = lambda_1(np.array([H_L]),np.array([C_L]), m, n)[0]
        
        C[eta >lambda1] = C_R
        H[eta >lambda1] = H_R   
        
        H_plot = np.linspace(H_L,H_R,20000)
        C_plot = int_region3_curves_lambda1(np.array([[C_L],[H_L]]),H_plot, m, n)        
    
    elif case_no==8:
        print('Case 8: Region 3 (Water + gas) - Rarefaction only')
        C = C_L*np.ones_like(eta)  
        H = H_L*np.ones_like(eta)             
        
        
        lambda_2L = lambda_2(np.array([H_L]),np.array([C_L]), m, n)[0] #left state
        lambda_2R = lambda_2(np.array([H_R]),np.array([C_R]), m, n)[0] #right state
        
        C[eta>=lambda_2L] = (eta[eta>=lambda_2L]/n)**(1/(n-1))
        H[eta>=lambda_2L] = H_R/C_R*C[eta>=lambda_2L]
        
        H[eta>=lambda_2R] = H_R
        C[eta>=lambda_2R] = C_R  

        H_plot = np.linspace(H_L,H_R,20000)
        C_plot = int_region3_curves_lambda2(np.array([[C_L],[H_L]]),H_plot, m, n) 
        
    elif case_no==9:
        print('Case 9: Region 3 (Water + gas) - Shock only')
        C = C_L*np.ones_like(eta)  
        H = H_L*np.ones_like(eta)             
        
        s = (f_C(np.array([H_L]),np.array([C_L]),m,n)[0]-f_C(np.array([H_R]),np.array([C_R]),m,n)[0])/(C_L - C_R)#Shock speed
        
        H[eta>=s] = H_R
        C[eta>=s] = C_R
        
        H_plot = np.linspace(H_L,H_R,20000)
        C_plot = int_region3_curves_lambda2(np.array([[C_L],[H_L]]),H_plot, m, n) 
        
        
    elif case_no==10:
        print('Case 10: Region 3 (Water + gas) - 1-Contact Discontinuity, 2-Rarefaction')
        C = C_L*np.ones_like(eta)  
        H = H_L*np.ones_like(eta)             
        
        C_I  = C_L
        H_I  = H_R*C_I/C_R

        lambda_2L = lambda_1(np.array([H_L]),np.array([C_L]), m, n)[0] #left state
        lambda_2I = lambda_2(np.array([H_I]),np.array([C_I]), m, n)[0] #intermediate state
        lambda_2R = lambda_2(np.array([H_R]),np.array([C_R]), m, n)[0] #right state

        H[eta>=lambda_2L] = H_I
        C[eta>=lambda_2L] = C_I
    
        
        C[eta>=lambda_2I] = (eta[eta>=lambda_2I]/n)**(1/(n-1))
        H[eta>=lambda_2I] = H_R/C_R*C[eta>=lambda_2I]
        
        H[eta>=lambda_2R] = H_R
        C[eta>=lambda_2R] = C_R        

        H_plot1 = np.linspace(H_L,H_I,10000)
        C_plot1 = int_region3_curves_lambda1(np.array([[C_L],[H_L]]),H_plot1, m, n) 
        H_plot2 = np.linspace(H_I,H_R,10000)
        C_plot2 = int_region3_curves_lambda2(np.array([[C_R],[H_R]]),H_plot2, m, n) 
        C_plot  = np.append(C_plot1,C_plot2)
        H_plot  = np.append(H_plot1,H_plot2)

    elif case_no==11:
        print('Case 11: Region 3 (Water + gas) - 1-Contact Discontinuity, 2-Shock')
        C = C_L*np.ones_like(eta)  
        H = H_L*np.ones_like(eta)   

        lambda_2L = lambda_1(np.array([H_L]),np.array([C_L]), m, n)[0] #left state

        s = (f_C(np.array([H_L]),np.array([C_L]),m,n)[0]-f_C(np.array([H_R]),np.array([C_R]),m,n)[0])/(C_L - C_R)#Shock speed
        
        C_I  = C_L
        H_I  = H_R*(C_R**(n-1)-s)/(C_I**(n-1)-s)

        H[eta>=lambda_2L] = H_I
        C[eta>=lambda_2L] = C_I

        H[eta>=s] = H_R
        C[eta>=s] = C_R
        
        H_plot1 = np.linspace(H_L,H_I,10000)
        C_plot1 = int_region3_curves_lambda1(np.array([[C_L],[H_L]]),H_plot1, m, n) 
        H_plot2 = np.linspace(H_I,H_R,10000)
        C_plot2 = int_region3_curves_lambda2(np.array([[C_R],[H_R]]),H_plot2, m, n) 
        C_plot  = np.append(C_plot1,C_plot2)
        H_plot  = np.append(H_plot1,H_plot2)
        
    elif case_no==12:
        print('Case 12: Contact Discontinuity + Rarefaction (Region 1 to 2)')
        
        C = C_L*np.ones_like(eta)  
        H = H_L*np.ones_like(eta)             
        
        H_I  = 0.0
        C_I  = H_I + C_R - H_R

        H[eta>=0] = H_I
        C[eta>=0] = C_I
        print(C_I,H_I)
        lambda_2I = lambda_2(np.array([H_I]),np.array([C_I]), m, n)[0] #intermediate state
        lambda_2R = lambda_2(np.array([H_R]),np.array([C_R]), m, n)[0] #right state
        
        H[eta>=lambda_2I] = (eta[eta>=lambda_2I]/(n*(1-C_R+H_R)**(m-n)))**(1/(n-1))
        C[eta>=lambda_2I] = C_R - H_R + H[eta>=lambda_2I]
        
        H[eta>=lambda_2R] = H_R
        C[eta>=lambda_2R] = C_R        

        H_plot1 = np.linspace(H_L,H_I,10000)
        C_plot1 = C_L+ (C_L-C_I)/(H_L-H_I)*(H_plot1 - H_L) 
        H_plot2 = np.linspace(H_I,H_R,10000)
        C_plot2 = int_curves_lambda2(np.array([[C_R],[H_R]]),H_plot2, m, n) 
        C_plot  = np.append(C_plot1,C_plot2)
        H_plot  = np.append(H_plot1,H_plot2)

    elif case_no==13:
        print('Case 13: Mixed R1 to R2: Single Shock only')
        C = C_L*np.ones_like(eta)  
        H = H_L*np.ones_like(eta)             
        
        s = (f_C(np.array([H_L]),np.array([C_L]),m,n)[0]-f_C(np.array([H_R]),np.array([C_R]),m,n)[0])/(C_L - C_R)#Shock speed
        
        H[eta>=s] = H_R
        C[eta>=s] = C_R
        
        H_I  = 0.0
        C_I  = H_I + C_L - H_L

        H_plot1 = np.linspace(H_L,H_I,10000)
        C_plot1 = int_curves_lambda2(np.array([[C_L],[H_L]]),H_plot1, m, n) 
        H_plot2 = np.linspace(H_I,H_R,10000)
        C_plot2 = C_R+ (C_R-C_I)/(H_R-H_I)*(H_plot2 - H_R) 
        C_plot  = np.append(C_plot1,C_plot2)
        H_plot  = np.append(H_plot1,H_plot2)
        
    elif case_no==14:
        print('Case 14: Contact Discontinuity + Shock (Region 2 to 1)')
        
        C = C_L*np.ones_like(eta)  
        H = H_L*np.ones_like(eta)             

        H_I = H_L*((1+H_L-C_L)/(1+H_R-C_R))**((m-n)/n)
        C_I = H_I + (C_R-H_R)

        H[eta>=0] = H_I
        C[eta>=0] = C_I
        
        s = (f_C(np.array([H_I]),np.array([C_I]),m,n)[0]-f_C(np.array([H_R]),np.array([C_R]),m,n)[0])/(C_I - C_R)#Shock speed
        
        H[eta>=s] = H_R
        C[eta>=s] = C_R        

        H_plot1 = np.linspace(H_L,H_I,10000)
        C_plot1 = int_curves_lambda1(np.array([[C_L],[H_L]]),H_plot1, m, n) 
        H_plot2 = np.linspace(H_I,H_R,10000)
        C_plot2 = C_R+ (C_R-C_I)/(H_R-H_I)*(H_plot2 - H_R) 
        C_plot  = np.append(C_plot1,C_plot2)
        H_plot  = np.append(H_plot1,H_plot2)
    
    elif case_no==15:
        print('Case 15: Contact Discontinuity + Shock (Region 1 to 2)')
        
        C = C_L*np.ones_like(eta)  
        H = H_L*np.ones_like(eta)             

        C_I = 1.0
        H_I = H_L**(n/m)*(1-C_L+H_L)**((m-n)/m)

        H[eta>=0] = H_I
        C[eta>=0] = C_I
        
        s = (f_H(np.array([H_I]),np.array([C_I]),m,n)[0]-f_H(np.array([H_R]),np.array([C_R]),m,n)[0])/(H_I - H_R)#Shock speed
        
        H[eta>=s] = H_R
        C[eta>=s] = C_R        

        H_plot1 = np.linspace(H_L,H_I,10000)
        C_plot1 = int_curves_lambda1(np.array([[C_L],[H_L]]),H_plot1, m, n) 
        H_plot2 = np.linspace(H_I,H_R,10000)
        C_plot2 = C_R+ (C_R-C_I)/(H_R-H_I)*(H_plot2 - H_R) 
        C_plot  = np.append(C_plot1,C_plot2)
        H_plot  = np.append(H_plot1,H_plot2)
        
        ''' 
    elif case_no==16:
        print('Case 16: Shock (Region 1 to 2) Single component case')
    
        C = C_L*np.ones_like(eta)  
        H = H_L*np.ones_like(eta)             
        
        s = (f_H(np.array([H_L]),np.array([C_L]),m,n)[0]-f_H(np.array([H_R]),np.array([C_R]),m,n)[0])/(H_L - H_R)#Shock speed
        
        H[eta>=s] = H_R
        C[eta>=s] = C_R
        
        H_plot = np.linspace(H_L,H_R,20000)
        C_plot = C_R + (C_R-C_L)/(H_R-H_L)*(H_plot - H_R) 
        '''
    
    elif case_no==18:
        print('Case 18: Mixed Region 2 to 3 - 1,2-Contact Discontinuity, 3-Rarefaction')
        C = C_L*np.ones_like(eta)  
        H = H_L*np.ones_like(eta)   

        lambda_2L = lambda_1(np.array([H_L]),np.array([C_L]), m, n)[0] #left state
        lambda_2R = lambda_2(np.array([H_R]),np.array([C_R]), m, n)[0] #right state
        
        H_I  = H_L*(1-C_L+H_L)**((m-n)/n)
        C_I  = H_I

        H[eta>=lambda_2L] = H_I
        C[eta>=lambda_2L] = C_I   
   
        C_II  = C_I
        H_II  = H_R/C_R*C_II

        lambda_1I = lambda_1(np.array([H_II]),np.array([C_II]), m, n)[0] #intermediate state     

        H[eta>=lambda_1I] = H_II
        C[eta>=lambda_1I] = C_II
        
        lambda_2II = lambda_2(np.array([H_II]),np.array([C_II]), m, n)[0] #intermediate state        
   
        C[eta>=lambda_2II] = (eta[eta>=lambda_2II]/(n))**(1/(n-1))
        H[eta>=lambda_2II] = H_R/C_R*C[eta>=lambda_2II]
    
    
        H[eta>=lambda_2R] = H_R
        C[eta>=lambda_2R] = C_R
        
        H_plot1 = np.linspace(H_L,H_I,8000)
        C_plot1 = int_curves_lambda1(np.array([[C_L],[H_L]]),H_plot1, m, n) 
        H_plot2 = np.linspace(H_I,H_II,8000)
        C_plot2 = int_region3_curves_lambda1(np.array([[C_I],[H_I]]),H_plot2, m, n) 
        H_plot3 = np.linspace(H_II,H_R,8000)
        C_plot3 = int_region3_curves_lambda2(np.array([[C_R],[H_R]]),H_plot3, m, n) 
        C_plot  = np.append(C_plot1,C_plot2)
        C_plot  = np.append(C_plot,C_plot3)
        H_plot  = np.append(H_plot1,H_plot2)
        H_plot  = np.append(H_plot,H_plot3)
        
    elif case_no==19:
        print('Case 19: Mixed Region 2 to 3 - 1,2-Contact Discontinuity, 3-Shock')
        C = C_L*np.ones_like(eta)  
        H = H_L*np.ones_like(eta)   

        lambda_2L = lambda_1(np.array([H_L]),np.array([C_L]), m, n)[0] #left state
        lambda_2R = lambda_2(np.array([H_R]),np.array([C_R]), m, n)[0] #right state
        
        H_I  = H_L*(1-C_L+H_L)**((m-n)/n)
        C_I  = H_I

        H[eta>=lambda_2L] = H_I
        C[eta>=lambda_2L] = C_I   
   
        C_II  = C_I
        H_II  = H_R/C_R*C_II

        lambda_1I = lambda_1(np.array([H_II]),np.array([C_II]), m, n)[0] #intermediate state     

        H[eta>=lambda_1I] = H_II
        C[eta>=lambda_1I] = C_II
        
        s   =   (C_R**n - C_II**n) / (C_R - C_II)    
    
    
        H[eta>=s] = H_R
        C[eta>=s] = C_R
        
        H_plot1 = np.linspace(H_L,H_I,8000)
        C_plot1 = int_curves_lambda1(np.array([[C_L],[H_L]]),H_plot1, m, n) 
        H_plot2 = np.linspace(H_I,H_II,8000)
        C_plot2 = int_region3_curves_lambda1(np.array([[C_I],[H_I]]),H_plot2, m, n) 
        H_plot3 = np.linspace(H_II,H_R,8000)
        C_plot3 = int_region3_curves_lambda2(np.array([[C_R],[H_R]]),H_plot3, m, n) 
        C_plot  = np.append(C_plot1,C_plot2)
        C_plot  = np.append(C_plot,C_plot3)
        H_plot  = np.append(H_plot1,H_plot2)
        H_plot  = np.append(H_plot,H_plot3)
        
        
    elif case_no==22:
        print('Case 22: Mixed Region 3 to 2 - 1-Shock, 2-Rarefaction')
        C = C_L*np.ones_like(eta)  
        H = H_L*np.ones_like(eta)   

        C_I  = lambda H_I: C_R + (H_I-H_R)
        func = lambda H_I: (H_L*C_L**(n-1)-(1-C_I(H_I)+H_I)**(m-n)*H_I**n)*(C_L - C_I(H_I)) - (H_L - H_I)*(C_L**n -(1-C_I(H_I)+H_I)**(m-n)*H_I**n)
        #func = lambda H_I: (H_L*C_L**(n-1)-(1-C_R+H_R)**(m-n)*H_I**n)*(C_L - C_R + (H_I-H_R)) - (H_L - H_I)*(C_L**n -(1-C_R+H_R)**(m-n)*H_I**n)
        
        H_I = fsolve(func,(H_L+H_R)/2)[0]
        C_I = C_I(H_I)
        
        s = (H_L*C_L**(n-1)-(1-C_I+H_I)**(m-n)*H_I**n)/(H_L - H_I)

        H[eta>=s] = H_I
        C[eta>=s] = C_I   
   
        lambda_2I = lambda_2(np.array([H_I]),np.array([C_I]), m, n)[0] #intermediate state     
        lambda_2R = lambda_2(np.array([H_R]),np.array([C_R]), m, n)[0] #right state
        

        H[eta>=lambda_2I] = (eta[eta>=lambda_2I]/(n*(1-C_R+H_R)**(m-n)))**(1/(n-1))
        C[eta>=lambda_2I] = C_R - H_R + H[eta>=lambda_2I]
        
        H[eta>=lambda_2R] = H_R
        C[eta>=lambda_2R] = C_R        

        H_plot1 = np.linspace(H_L,H_I,10000)
        C_plot1 = C_L+ (C_L-C_I)/(H_L-H_I)*(H_plot1 - H_L) 
        H_plot2 = np.linspace(H_I,H_R,10000)
        C_plot2 = int_curves_lambda2(np.array([[C_R],[H_R]]),H_plot2, m, n) 
        C_plot  = np.append(C_plot1,C_plot2)
        H_plot  = np.append(H_plot1,H_plot2)
        
    elif case_no==23:
        print('Case 23: Mixed Region 3 to 2 - 1-Shock, 2-Shock')
        C = C_L*np.ones_like(eta)  
        H = H_L*np.ones_like(eta)   

        C_I  = lambda H_I: C_R + (H_I-H_R)
        func = lambda H_I: (H_L*C_L**(n-1)-(1-C_I(H_I)+H_I)**(m-n)*H_I**n)*(C_L - C_I(H_I)) - (H_L - H_I)*(C_L**n -(1-C_I(H_I)+H_I)**(m-n)*H_I**n)
        #func = lambda H_I: (H_L*C_L**(n-1)-(1-C_R+H_R)**(m-n)*H_I**n)*(C_L - C_R + (H_I-H_R)) - (H_L - H_I)*(C_L**n -(1-C_R+H_R)**(m-n)*H_I**n)
        
        H_I = fsolve(func,(H_L+H_R)/2)[0]
        C_I = C_I(H_I)
        
        s = (H_L*C_L**(n-1)-(1-C_I+H_I)**(m-n)*H_I**n)/(H_L - H_I)

        H[eta>=s] = H_I
        C[eta>=s] = C_I   
   
        lambda_2I = lambda_2(np.array([H_I]),np.array([C_I]), m, n)[0] #intermediate state     
        lambda_2R = lambda_2(np.array([H_R]),np.array([C_R]), m, n)[0] #right state
        

        s2 = (f_H(np.array([H_I]),np.array([C_I]),m,n)[0]-f_H(np.array([H_R]),np.array([C_R]),m,n)[0])/(H_I - H_R)#Shock speed
        

        H[eta>=s2] = H_R
        C[eta>=s2] = C_R        

        H_plot1 = np.linspace(H_L,H_I,10000)
        C_plot1 = C_L+ (C_L-C_I)/(H_L-H_I)*(H_plot1 - H_L) 
        H_plot2 = np.linspace(H_I,H_R,10000)
        C_plot2 = int_curves_lambda2(np.array([[C_R],[H_R]]),H_plot2, m, n) 
        C_plot  = np.append(C_plot1,C_plot2)
        H_plot  = np.append(H_plot1,H_plot2)
        
        
    elif case_no==24:
        print('Case 24: Mixed Region 1 to 3 - 1-Contact, 2-Rarefaction')
        C = C_L*np.ones_like(eta)  
        H = H_L*np.ones_like(eta)   


        C_I  = 0.0
        H_I  = 0.0

        lambda_2I = lambda_2(np.array([H_I]),np.array([C_I]), m, n)[0] #intermediate state
        lambda_2R = lambda_2(np.array([H_R]),np.array([C_R]), m, n)[0] #right state

        H[eta>=0] = H_I
        C[eta>=0] = C_I
    
        
        C[eta>=lambda_2I] = (eta[eta>=lambda_2I]/n)**(1/(n-1))
        H[eta>=lambda_2I] = H_R/C_R*C[eta>=lambda_2I]
        
        H[eta>=lambda_2R] = H_R
        C[eta>=lambda_2R] = C_R        

        H_plot1 = np.linspace(H_L,H_I,10000)
        C_plot1 = C_L+ (C_L-C_I)/(H_L-H_I)*(H_plot1 - H_L) 
        H_plot2 = np.linspace(H_I,H_R,10000)
        C_plot2 = int_region3_curves_lambda2(np.array([[C_R],[H_R]]),H_plot2, m, n) 
        C_plot  = np.append(C_plot1,C_plot2)
        H_plot  = np.append(H_plot1,H_plot2)
 
    elif case_no==29:
        print('Case 29: Mixed Region 3 to 1 - 1-Shock, 2-Shock')
        C = C_L*np.ones_like(eta)  
        H = H_L*np.ones_like(eta)   

        C_I  = lambda H_I: C_R + (H_I-H_R)
        func = lambda H_I: (H_L*C_L**(n-1)-(1-C_I(H_I)+H_I)**(m-n)*H_I**n)*(C_L - C_I(H_I)) - (H_L - H_I)*(C_L**n -(1-C_I(H_I)+H_I)**(m-n)*H_I**n)
        #func = lambda H_I: (H_L*C_L**(n-1)-(1-C_R+H_R)**(m-n)*H_I**n)*(C_L - C_R + (H_I-H_R)) - (H_L - H_I)*(C_L**n -(1-C_R+H_R)**(m-n)*H_I**n)
        
        H_I = fsolve(func,(H_L+H_R)/2)[0]
        C_I = C_I(H_I)
        
        s = (H_L*C_L**(n-1)-(1-C_I+H_I)**(m-n)*H_I**n)/(H_L - H_I)

        H[eta>=s] = H_I
        C[eta>=s] = C_I   
   
        lambda_2I = lambda_2(np.array([H_I]),np.array([C_I]), m, n)[0] #intermediate state     
        lambda_2R = lambda_2(np.array([H_R]),np.array([C_R]), m, n)[0] #right state
        

        s2 = (f_H(np.array([H_I]),np.array([C_I]),m,n)[0]-f_H(np.array([H_R]),np.array([C_R]),m,n)[0])/(H_I - H_R)#Shock speed
        

        H[eta>=s2] = H_R
        C[eta>=s2] = C_R        

        H_plot1 = np.linspace(H_L,H_I,10000)
        C_plot1 = C_L+ (C_L-C_I)/(H_L-H_I)*(H_plot1 - H_L) 
        H_plot2 = np.linspace(H_I,H_R,10000)
        C_plot2 = C_R+ (C_R-C_I)/(H_R-H_I)*(H_plot2 - H_R) 
        C_plot  = np.append(C_plot1,C_plot2)
        H_plot  = np.append(H_plot1,H_plot2)
        
    elif case_no==31:
        print('Case 31: Infiltration into Temperate firn with saturated region - 1-Shock, 2-Contact, 3-Shock')
        print('Wrong numeric except hodograph plane diagram')
        C = C_L*np.ones_like(eta)  
        H = H_L*np.ones_like(eta)   
 
        C_I1 = 1; C_I2 = 1
        H_I1 = 1 - C_L + H_L 
        H_I2 = 1 - C_R + H_R 
    
        #Analytical solution
        
        a = (1-C_L)/(1-C_R)*(1 - ((1-C_R+H_R)/(1-C_L+H_L))**m * (H_R/(1-C_R+H_R))**n )
        b =-(1-C_L)/(1-C_R)*(1- (H_R/(1-C_R+H_R))**n ) + (H_L/(1-C_L+H_L))**n - 1
        c = 1 - ((1-C_L+H_L) / (1-C_R+H_R))**m * (H_L/(1-C_L+H_L))**n

        Ratio = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)

        q_s_dimless = (Ratio - 1)/(Ratio/(1-C_L+H_L)**m - 1/(1-C_R+H_R)**m)
        
        
        s1 = (q_s_dimless - f_H(np.array([H_L]),np.array([C_L]),m,n)[0])/(H_I1 - H_L)#Backfilling shock speed      
        s3 = (q_s_dimless-f_H(np.array([H_R]),np.array([C_R]),m,n)[0])/(H_I2 - H_R)#Shock speed

        print(Ratio,q_s_dimless,s1,s3)

        H[eta>=s1] = H_I1
        C[eta>=s1] = C_I1  

        H[eta>=0] = H_I2
        C[eta>=0] = C_I2 
    
        H[eta>=s3] = H_R
        C[eta>=s3] = C_R        
 
        H_plot1 = np.linspace(H_L,H_I1,6666)
        C_plot1 = C_L+ (C_L-C_I1)/(H_L-H_I1)*(H_plot1 - H_L) 
        H_plot2 = np.linspace(H_I1,H_I2,6667)
        C_plot2 = C_I1*np.ones_like(H_plot2)
        
        H_plot3 = np.linspace(H_I2,H_R,6667)
        C_plot3 = C_R+ (C_R-C_I2)/(H_R-H_I2)*(H_plot3 - H_R) 
        C_plot  = np.append(C_plot1,C_plot2)
        C_plot  = np.append(C_plot,C_plot3)
        H_plot  = np.append(H_plot1,H_plot2)
        H_plot  = np.append(H_plot,H_plot3)

    elif case_no==32:
        print('Case 32: Infiltration into Cold firn with saturated region - 1-Shock, 2-Contact, 3-Shock')
        print('Wrong numeric except hodograph plane diagram')
        C = C_L*np.ones_like(eta)  
        H = H_L*np.ones_like(eta)   
 
        C_I1 = 1; C_I2 = 1
        H_I1 = 1 - C_L + H_L 
        H_I2 = 1 - C_R + H_R 
    
        
        
        #Analytical solution
        
        a = (1-C_L)/(1-C_R)
        b =-(1-C_L)/(1-C_R) + (H_L/(1-C_L+H_L))**n - 1
        c = 1 - ((1-C_L+H_L) / (1-C_R+H_R))**m * (H_L/(1-C_L+H_L))**n

        Ratio = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)

        q_s_dimless = (Ratio - 1)/(Ratio/(1-C_L+H_L)**m - 1/(1-C_R+H_R)**m)
        
        
        s1 = (q_s_dimless - f_H(np.array([H_L]),np.array([C_L]),m,n)[0])/(H_I1 - H_L)#Backfilling shock speed      
        s3 = (q_s_dimless-f_H(np.array([H_R]),np.array([C_R]),m,n)[0])/(H_I2 - H_R)#Shock speed

        print(Ratio,q_s_dimless,s1,s3)

        H[eta>=s1] = H_I1
        C[eta>=s1] = C_I1  

        H[eta>=0] = H_I2
        C[eta>=0] = C_I2 
    
        H[eta>=s3] = H_R
        C[eta>=s3] = C_R        
 
        H_plot1 = np.linspace(H_L,H_I1,6666)
        C_plot1 = C_L+ (C_L-C_I1)/(H_L-H_I1)*(H_plot1 - H_L) 
        H_plot2 = np.linspace(H_I1,H_I2,6667)
        C_plot2 = C_I1*np.ones_like(H_plot2)
        
        H_plot3 = np.linspace(H_I2,H_R,6667)
        C_plot3 = C_R+ (C_R-C_I2)/(H_R-H_I2)*(H_plot3 - H_R) 
        C_plot  = np.append(C_plot1,C_plot2)
        C_plot  = np.append(C_plot,C_plot3)
        H_plot  = np.append(H_plot1,H_plot2)
        H_plot  = np.append(H_plot,H_plot3)

    elif case_no==33:
        print('Case 33: Ice lens formation')
        print('Wrong analytic except hodograph plane diagram')
        C = C_L*np.ones_like(eta)  
        H = H_L*np.ones_like(eta)   
 
        C_I1 = 1; C_I2 = 1
        H_I1 = 1 - C_L + H_L 
        H_I2 = 0  
    
        s1 = (f_H(np.array([H_L]),np.array([C_L]),m,n)[0])/(C_L - 1)#Backfilling shock speed       
 
        print(s1)
        H[eta>=s1] = H_I1
        C[eta>=s1] = C_I1  

        H[eta>=0] = H_I2
        C[eta>=0] = C_I2 
    
        H[eta>=0.05] = H_R
        C[eta>=0.05] = C_R        
 
    
        H_plot1 = np.linspace(H_L,H_I1,6666)
        C_plot1 = C_L+ (C_L-C_I1)/(H_L-H_I1)*(H_plot1 - H_L) 
        H_plot2 = np.linspace(H_I1,H_I2,6667)
        C_plot2 = C_I1*np.ones_like(H_plot2)
        
        H_plot3 = np.linspace(H_I2,H_R,6667)
        C_plot3 = C_R+ (C_R-C_I2)/(H_R-H_I2)*(H_plot3 - H_R) 
        C_plot  = np.append(C_plot1,C_plot2)
        C_plot  = np.append(C_plot,C_plot3)
        H_plot  = np.append(H_plot1,H_plot2)
        H_plot  = np.append(H_plot,H_plot3)
        

    else: #Not plotting
        C = np.nan*np.ones_like(eta)
        H = np.nan*np.ones_like(eta)
        C_plot = np.nan*np.ones_like(eta)
        H_plot = np.nan*np.ones_like(eta)
                
    return eta,C,H,C_plot,H_plot

def plotting_singularAGU(simulation_name,C_case_plot1,H_case_plot1,m,n):
    H = np.linspace(0,1,10000)
    
    norm = colors.Normalize(vmin=0, vmax=1)
    fig = plt.figure(figsize=(7,10), dpi=100)
    
    
    #fancy plots
#    light_red  = [1.0,0.5,0.5]
    #light_blue = [0.5,0.5,1.0]
   # light_black= [0.5,0.5,0.5]

    #Region 2 Three phase region
    C1 = int_curves_lambda2(np.array([[0.05],[0.0]]),H, m, n)   #u0 = [C,H]^T
    C2 = int_curves_lambda2(np.array([[0.2],[0]]),H, m, n) 
    C3 = int_curves_lambda2(np.array([[0.4],[0]]),H, m, n) 
    C4 = int_curves_lambda2(np.array([[0.6],[0]]),H, m, n) 
    C5 = int_curves_lambda2(np.array([[0.8],[0]]),H, m, n) 
    
    C1_2 = int_curves_lambda1(np.array([[0.05],[0.05]]),H, m, n)   #u0 = [C,H]^T
    C2_2 = int_curves_lambda1(np.array([[0.2],[0.2]]),H, m, n)
    C3_2 = int_curves_lambda1(np.array([[0.4],[0.4]]),H, m, n) 
    C4_2 = int_curves_lambda1(np.array([[0.6],[0.6]]),H, m, n) 
    C5_2 = int_curves_lambda1(np.array([[0.8],[0.8]]),H, m, n) 
    
    #removing the curves outside the region
    C1_2[C1_2<H] = np.nan
    C2_2[C2_2<H] = np.nan
    C3_2[C3_2<H] = np.nan
    C4_2[C4_2<H] = np.nan
    C5_2[C5_2<H] = np.nan
    

    plt.plot(C1_2,H,'--',c=light_blue,label=r'Slow path, $\bf{r}_1$')
    plt.plot(C2_2,H,'--',c=light_blue)
    plt.plot(C3_2,H,'--',c=light_blue)
    plt.plot(C4_2,H,'--',c=light_blue)
    plt.plot(C5_2,H,'--',c=light_blue)
    plt.plot(C1,H,'-',c=light_red,label=r'Fast path, $\bf{r}_2$')
    plt.plot(C2,H,'-',c=light_red)
    plt.plot(C3,H,'-',c=light_red)
    plt.plot(C4,H,'-',c=light_red)
    plt.plot(C5,H,'-',c=light_red)
    
    
    plt.plot([0,1],[0,1],'k--',alpha=0.3)
    plt.plot([0,1],[0,0],'k--',alpha=0.3)
    plt.plot([1,1],[1,0],'k--',alpha=0.3)
    
    
    plt.plot(C_case_plot1,H_case_plot1,'b-',linewidth=3)
    plt.plot(C_case_plot1,H_case_plot1,'r--',linewidth=3)

    plt.plot(C_case_plot1[0],H_case_plot1[0],'kv', markersize=20,markerfacecolor='white',markeredgewidth=3)
    
    plt.plot(C_case_plot1[-1],H_case_plot1[-1],'ks', markersize=20,markerfacecolor='white',markeredgewidth=3)
    
    x_pos1 = C_case_plot1[int(len(C_case_plot1)/2)-1]
    y_pos1 = H_case_plot1[int(len(H_case_plot1)/2)-1]
    x_direct1 = C_case_plot1[int(len(C_case_plot1)/2)]-C_case_plot1[int(len(C_case_plot1)/2)-1]
    y_direct1 = H_case_plot1[int(len(H_case_plot1)/2)]-H_case_plot1[int(len(H_case_plot1)/2)-1]
    #plt.arrow(x_pos1, y_pos1, x_direct1, y_direct1,head_width=0.04, head_length=0.04, fc='k', ec='k')

    
    plt.ylabel(r'Dim-less Enthalpy $\mathcal{H}$')
    plt.xlabel(r'Dim-less Composition $\mathcal{C}$')
    #plt.ylabel(r'$\mathcal{H}$')
    #plt.xlabel(r'$\mathcal{C}$')
    plt.legend(loc='upper left',borderaxespad=0.)
    #plt.axis('scaled')
    plt.ylim([-0.2,1])
    plt.xlim([0,1])
    plt.savefig(f'../Figures/{simulation_name}_integral_curves_combined.pdf')
    


def plotting_combined(simulation_name,C_case_plot1,H_case_plot1,C_case_plot2,H_case_plot2,C_case_plot3,H_case_plot3,m,n):
    H = np.linspace(0,1,10000)
    
    norm = colors.Normalize(vmin=0, vmax=1)
    fig = plt.figure(figsize=(7,10), dpi=100)
    
    
    #fancy plots
#    light_red  = [1.0,0.5,0.5]
    #light_blue = [0.5,0.5,1.0]
   # light_black= [0.5,0.5,0.5]

    #Region 2 Three phase region
    C1 = int_curves_lambda2(np.array([[0.1],[0.0]]),H, m, n)   #u0 = [C,H]^T
    C2 = int_curves_lambda2(np.array([[0.3],[0]]),H, m, n) 
    C3 = int_curves_lambda2(np.array([[0.5],[0]]),H, m, n) 
    C4 = int_curves_lambda2(np.array([[0.7],[0]]),H, m, n) 
    C5 = int_curves_lambda2(np.array([[0.9],[0]]),H, m, n) 
    
    C1_2 = int_curves_lambda1(np.array([[0.1],[0.1]]),H, m, n)   #u0 = [C,H]^T
    C2_2 = int_curves_lambda1(np.array([[0.5],[0.5]]),H, m, n)
    C4_2 = int_curves_lambda1(np.array([[0.7],[0.4]]),H, m, n) 
    C3_2 = int_curves_lambda1(np.array([[0.7],[0.7]]),H, m, n) 
    C5_2 = int_curves_lambda1(np.array([[0.9],[0.9]]),H, m, n) 
    
    #removing the curves outside the region
    C1_2[C1_2<H] = np.nan
    C2_2[C2_2<H] = np.nan
    C3_2[C3_2<H] = np.nan
    C4_2[C4_2<H] = np.nan
    C5_2[C5_2<H] = np.nan
    

    plt.plot(C1_2,H,'--',c=light_blue,label=r'Slow path, $\bf{r}_1$')
    plt.plot(C2_2,H,'--',c=light_blue)
    plt.plot(C3_2,H,'--',c=light_blue)
    plt.plot(C4_2,H,'--',c=light_blue)
    plt.plot(C5_2,H,'--',c=light_blue)
    plt.plot(C1,H,'-',c=light_red,label=r'Fast path, $\bf{r}_2$')
    plt.plot(C2,H,'-',c=light_red)
    plt.plot(C3,H,'-',c=light_red)
    plt.plot(C4,H,'-',c=light_red)
    plt.plot(C5,H,'-',c=light_red)
    
    
    plt.plot([0,1],[0,1],'k--',alpha=0.3)
    plt.plot([0,1],[0,0],'k--',alpha=0.3)
    plt.plot([1,1],[1,0],'k--',alpha=0.3)
    
    
    plt.plot(C_case_plot1,H_case_plot1,'g-',linewidth=3,label='Contact Discontinuity')
    plt.plot(C_case_plot2,H_case_plot2,'b-',linewidth=3,label='Rarefaction')
    plt.plot(C_case_plot3,H_case_plot3,'r-',linewidth=3,label='Shock')


    plt.plot(C_case_plot1[0],H_case_plot1[0],'kv', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot2[0],H_case_plot2[0],'kv', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot3[0],H_case_plot3[0],'kv', markersize=20,markerfacecolor='white',markeredgewidth=3)
    
    plt.plot(C_case_plot1[-1],H_case_plot1[-1],'go', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot2[-1],H_case_plot2[-1],'b^', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot3[-1],H_case_plot3[-1],'rs', markersize=20,markerfacecolor='white',markeredgewidth=3)
    
    x_pos1 = C_case_plot1[int(len(C_case_plot1)/2)-1]
    y_pos1 = H_case_plot1[int(len(H_case_plot1)/2)-1]
    x_direct1 = C_case_plot1[int(len(C_case_plot1)/2)]-C_case_plot1[int(len(C_case_plot1)/2)-1]
    y_direct1 = H_case_plot1[int(len(H_case_plot1)/2)]-H_case_plot1[int(len(H_case_plot1)/2)-1]
    #plt.arrow(x_pos1, y_pos1, x_direct1, y_direct1,head_width=0.04, head_length=0.04, fc='g', ec='g')
    
    x_pos1 = C_case_plot2[int(len(C_case_plot2)/2)-1]
    y_pos1 = H_case_plot2[int(len(H_case_plot2)/2)-1]
    x_direct1 = C_case_plot2[int(len(C_case_plot2)/2)]-C_case_plot2[int(len(C_case_plot2)/2)-1]
    y_direct1 = H_case_plot2[int(len(H_case_plot2)/2)]-H_case_plot2[int(len(H_case_plot2)/2)-1]
    #plt.arrow(x_pos1, y_pos1, x_direct1, y_direct1,head_width=0.04, head_length=0.04, fc='b', ec='b')
    
    x_pos1 = C_case_plot3[int(len(C_case_plot3)/2)-1]
    y_pos1 = H_case_plot3[int(len(H_case_plot3)/2)-1]
    x_direct1 = C_case_plot3[int(len(C_case_plot3)/2)]-C_case_plot3[int(len(C_case_plot3)/2)-1]
    y_direct1 = H_case_plot3[int(len(H_case_plot3)/2)]-H_case_plot3[int(len(H_case_plot3)/2)-1]
    #plt.arrow(x_pos1, y_pos1, x_direct1, y_direct1,head_width=0.04, head_length=0.04, fc='r', ec='r')

    
    plt.ylabel(r'Dim-less Enthalpy $\mathcal{H}$')
    plt.xlabel(r'Dim-less Composition $\mathcal{C}$')
    #plt.ylabel(r'$\mathcal{H}$')
    #plt.xlabel(r'$\mathcal{C}$')
    plt.legend(loc='upper left',borderaxespad=0., framealpha=0.1)
    plt.axis('scaled')
    plt.ylim([-0.2,1])
    plt.xlim([0,1])
    plt.savefig(f'../Figures/{simulation_name}_integral_curves_combined.pdf')
    
def plotting_combined2(simulation_name,C_case_plot1,H_case_plot1,C_case_plot2,H_case_plot2,m,n):
    H = np.linspace(0,1,10000)
    
    norm = colors.Normalize(vmin=0, vmax=1)
    fig = plt.figure(figsize=(7,10), dpi=100)
    
    
    #fancy plots
    #  light_red  = [1.0,0.5,0.5]
    #light_blue = [0.5,0.5,1.0]
   # light_black= [0.5,0.5,0.5]

    #Region 2 Three phase region
    C1 = int_curves_lambda2(np.array([[0.05],[0.0]]),H, m, n)   #u0 = [C,H]^T
    C2 = int_curves_lambda2(np.array([[0.2],[0]]),H, m, n) 
    C3 = int_curves_lambda2(np.array([[0.948],[0.528]]),H, m, n) 
    C4 = int_curves_lambda2(np.array([[0.6],[0]]),H, m, n) 
    C5 = int_curves_lambda2(np.array([[0.8],[0]]),H, m, n) 
    
    C1_2 = int_curves_lambda1(np.array([[0.534],[0.116]]),H, m, n)   #u0 = [C,H]^T
    C2_2 = int_curves_lambda1(np.array([[0.2],[0.2]]),H, m, n)
    C3_2 = int_curves_lambda1(np.array([[0.4],[0.4]]),H, m, n) 
    C4_2 = int_curves_lambda1(np.array([[0.6],[0.6]]),H, m, n) 
    C5_2 = int_curves_lambda1(np.array([[0.8],[0.8]]),H, m, n) 
    
    #removing the curves outside the region
    C1_2[C1_2<H] = np.nan
    C2_2[C2_2<H] = np.nan
    C3_2[C3_2<H] = np.nan
    C4_2[C4_2<H] = np.nan
    C5_2[C5_2<H] = np.nan
    

    plt.plot(C1_2,H,'--',c=light_blue,label=r'Slow path, $\bf{r}_1$')
    plt.plot(C2_2,H,'--',c=light_blue)
    plt.plot(C3_2,H,'--',c=light_blue)
    plt.plot(C4_2,H,'--',c=light_blue)
    plt.plot(C5_2,H,'--',c=light_blue)
    plt.plot(C1,H,'-',c=light_red,label=r'Fast path, $\bf{r}_2$')
    plt.plot(C2,H,'-',c=light_red)
    plt.plot(C3,H,'-',c=light_red)
    plt.plot(C4,H,'-',c=light_red)
    plt.plot(C5,H,'-',c=light_red)
    
    
    plt.plot([0,1],[0,1],'k--',alpha=0.3)
    plt.plot([0,1],[0,0],'k--',alpha=0.3)
    plt.plot([1,1],[1,0],'k--',alpha=0.3)
    
    plt.ylim([-0.2,1])
    plt.xlim([0,1])
    plt.plot(C_case_plot1,H_case_plot1,'b-',linewidth=3,label='$\mathcal{C}_1\mathcal{R}_2$')
    plt.plot(C_case_plot2,H_case_plot2,'r-',linewidth=3,label='$\mathcal{C}_1\mathcal{S}_2$')


    plt.plot(C_case_plot1[0],H_case_plot1[0],'bv', markersize=20,markerfacecolor='none',markeredgewidth=3)
    plt.plot(C_case_plot1[-1],H_case_plot1[-1],'bo', markersize=20,markerfacecolor='none',markeredgewidth=3) 
    plt.plot(C_case_plot2[0],H_case_plot2[0],'rv', markersize=20,markerfacecolor='none',markeredgewidth=3)
    
    plt.plot(C_case_plot1[int(10000-1)],H_case_plot1[int(10000-1)],'bd', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot2[int(10000-1)],H_case_plot2[int(10000-1)],'rs', markersize=20,markerfacecolor='white',markeredgewidth=3)
    
    plt.plot(C_case_plot2[-1],H_case_plot2[-1],'ro', markersize=20,markerfacecolor='none',markeredgewidth=3)

    x_pos1 = C_case_plot1[int(len(C_case_plot1)/4)-1]
    y_pos1 = H_case_plot1[int(len(H_case_plot1)/4)-1]
    x_direct1 = C_case_plot1[int(len(C_case_plot1)/4)]-C_case_plot1[int(len(C_case_plot1)/4)-1]
    y_direct1 = H_case_plot1[int(len(H_case_plot1)/4)]-H_case_plot1[int(len(H_case_plot1)/4)-1]
    #plt.arrow(x_pos1, y_pos1, x_direct1, y_direct1,head_width=0.04, head_length=0.04, fc='b', ec='b')

    x_pos1 = C_case_plot1[int(3*len(C_case_plot1)/4)-1]
    y_pos1 = H_case_plot1[int(3*len(H_case_plot1)/4)-1]
    x_direct1 = C_case_plot1[int(3*len(C_case_plot1)/4)]-C_case_plot1[int(3*len(C_case_plot1)/4)-1]
    y_direct1 = H_case_plot1[int(3*len(H_case_plot1)/4)]-H_case_plot1[int(3*len(H_case_plot1)/4)-1]
    #plt.arrow(x_pos1, y_pos1, x_direct1, y_direct1,head_width=0.04, head_length=0.04, fc='b', ec='b')
    
    x_pos1 = C_case_plot2[int(len(C_case_plot2)/4)-1]
    y_pos1 = H_case_plot2[int(len(H_case_plot2)/4)-1]
    x_direct1 = C_case_plot2[int(len(C_case_plot2)/4)]-C_case_plot2[int(len(C_case_plot2)/4)-1]
    y_direct1 = H_case_plot2[int(len(H_case_plot2)/4)]-H_case_plot2[int(len(H_case_plot2)/4)-1]
    #plt.arrow(x_pos1, y_pos1, x_direct1, y_direct1,head_width=0.04, head_length=0.04, fc='r', ec='r')

    x_pos1 = C_case_plot2[int(3*len(C_case_plot2)/4)-1]
    y_pos1 = H_case_plot2[int(3*len(H_case_plot2)/4)-1]
    x_direct1 = C_case_plot2[int(3*len(C_case_plot2)/4)]-C_case_plot2[int(3*len(C_case_plot2)/4)-1]
    y_direct1 = H_case_plot2[int(3*len(H_case_plot2)/4)]-H_case_plot2[int(3*len(H_case_plot2)/4)-1]
    #plt.arrow(x_pos1, y_pos1, x_direct1, y_direct1,head_width=0.04, head_length=0.04, fc='r', ec='r')
    
    plt.ylabel(r'Dim-less Enthalpy $\mathcal{H}$')
    plt.xlabel(r'Dim-less Composition $\mathcal{C}$')
    #plt.ylabel(r'$\mathcal{H}$')
    #plt.xlabel(r'$\mathcal{C}$')
    plt.axis('scaled')
    plt.legend(loc='upper left',borderaxespad=0.)
    plt.ylim([-0.2,1])
    plt.xlim([0,1])
    plt.savefig(f'../Figures/{simulation_name}_integral_curves_combined2.pdf')
    
    
def plotting_combined3(simulation_name,C_case_plot1,H_case_plot1,C_case_plot2,H_case_plot2,m,n): #left is region 1
    H = np.linspace(0,1,10000)
    
    norm = colors.Normalize(vmin=0, vmax=1)
    fig = plt.figure(figsize=(7,10), dpi=100)
    
    #fancy plots
#    light_red  = [1.0,0.5,0.5]
    #light_blue = [0.5,0.5,1.0]
   # light_black= [0.5,0.5,0.5]

    #Region 2 Three phase region
    C1 = int_curves_lambda2(np.array([[0.05],[0.0]]),H, m, n)   #u0 = [C,H]^T
    C2 = int_curves_lambda2(np.array([[0.2],[0]]),H, m, n) 
    C3 = int_curves_lambda2(np.array([[0.4],[0]]),H, m, n) 
    C4 = int_curves_lambda2(np.array([[0.6],[0]]),H, m, n) 
    C5 = int_curves_lambda2(np.array([[0.8],[0]]),H, m, n) 
    
    C1_2 = int_curves_lambda1(np.array([[0.05],[0.05]]),H, m, n)   #u0 = [C,H]^T
    C2_2 = int_curves_lambda1(np.array([[0.2],[0.2]]),H, m, n)
    C3_2 = int_curves_lambda1(np.array([[0.4],[0.4]]),H, m, n) 
    C4_2 = int_curves_lambda1(np.array([[0.6],[0.6]]),H, m, n) 
    C5_2 = int_curves_lambda1(np.array([[0.8],[0.8]]),H, m, n) 
    
    #removing the curves outside the region
    C1_2[C1_2<H] = np.nan
    C2_2[C2_2<H] = np.nan
    C3_2[C3_2<H] = np.nan
    C4_2[C4_2<H] = np.nan
    C5_2[C5_2<H] = np.nan
    

    plt.plot(C1_2,H,'--',c=light_blue,label=r'Slow path, $\bf{r}_1$')
    plt.plot(C2_2,H,'--',c=light_blue)
    plt.plot(C3_2,H,'--',c=light_blue)
    plt.plot(C4_2,H,'--',c=light_blue)
    plt.plot(C5_2,H,'--',c=light_blue)
    plt.plot(C1,H,'-',c=light_red,label=r'Fast path, $\bf{r}_2$')
    plt.plot(C2,H,'-',c=light_red)
    plt.plot(C3,H,'-',c=light_red)
    plt.plot(C4,H,'-',c=light_red)
    plt.plot(C5,H,'-',c=light_red)
    
    
    plt.plot([0,1],[0,1],'k--',alpha=0.3)
    plt.plot([0,1],[0,0],'k--',alpha=0.3)
    plt.plot([1,1],[1,0],'k--',alpha=0.3)
    
    plt.plot(C_case_plot1,H_case_plot1,'b-',linewidth=3,label='$\mathcal{C}$')
    plt.plot(C_case_plot2,H_case_plot2,'r-',linewidth=3,label='$\mathcal{C}_1\mathcal{R}_2$')


    plt.plot(C_case_plot1[0],H_case_plot1[0],'bv', markersize=20,markerfacecolor='none',markeredgewidth=3)
    plt.plot(C_case_plot1[-1],H_case_plot1[-1],'bo', markersize=20,markerfacecolor='none',markeredgewidth=3) 
    plt.plot(C_case_plot2[0],H_case_plot2[0],'rv', markersize=20,markerfacecolor='none',markeredgewidth=3)
    
    plt.plot(C_case_plot2[int(10000-1)],H_case_plot2[int(10000-1)],'rs', markersize=20,markerfacecolor='white',markeredgewidth=3)
    
    plt.plot(C_case_plot2[-1],H_case_plot2[-1],'ro', markersize=20,markerfacecolor='none',markeredgewidth=3)

    x_pos1 = C_case_plot1[int(len(C_case_plot1)/4)-1]
    y_pos1 = H_case_plot1[int(len(H_case_plot1)/4)-1]
    x_direct1 = C_case_plot1[int(len(C_case_plot1)/4)]-C_case_plot1[int(len(C_case_plot1)/4)-1]
    y_direct1 = H_case_plot1[int(len(H_case_plot1)/4)]-H_case_plot1[int(len(H_case_plot1)/4)-1]
    #plt.arrow(x_pos1, y_pos1, x_direct1, y_direct1,head_width=0.04, head_length=0.04, fc='b', ec='b')

    x_pos1 = C_case_plot1[int(3*len(C_case_plot1)/4)-1]
    y_pos1 = H_case_plot1[int(3*len(H_case_plot1)/4)-1]
    x_direct1 = C_case_plot1[int(3*len(C_case_plot1)/4)]-C_case_plot1[int(3*len(C_case_plot1)/4)-1]
    y_direct1 = H_case_plot1[int(3*len(H_case_plot1)/4)]-H_case_plot1[int(3*len(H_case_plot1)/4)-1]
    #plt.arrow(x_pos1, y_pos1, x_direct1, y_direct1,head_width=0.04, head_length=0.04, fc='b', ec='b')
    
    x_pos1 = C_case_plot2[int(len(C_case_plot2)/4)-1]
    y_pos1 = H_case_plot2[int(len(H_case_plot2)/4)-1]
    x_direct1 = C_case_plot2[int(len(C_case_plot2)/4)]-C_case_plot2[int(len(C_case_plot2)/4)-1]
    y_direct1 = H_case_plot2[int(len(H_case_plot2)/4)]-H_case_plot2[int(len(H_case_plot2)/4)-1]
    #plt.arrow(x_pos1, y_pos1, x_direct1, y_direct1,head_width=0.04, head_length=0.04, fc='r', ec='r')

    x_pos1 = C_case_plot2[int(3*len(C_case_plot2)/4)-1]
    y_pos1 = H_case_plot2[int(3*len(H_case_plot2)/4)-1]
    x_direct1 = C_case_plot2[int(3*len(C_case_plot2)/4)]-C_case_plot2[int(3*len(C_case_plot2)/4)-1]
    y_direct1 = H_case_plot2[int(3*len(H_case_plot2)/4)]-H_case_plot2[int(3*len(H_case_plot2)/4)-1]
    #plt.arrow(x_pos1, y_pos1, x_direct1, y_direct1,head_width=0.04, head_length=0.04, fc='r', ec='r')
    
    plt.ylabel(r'Dim-less Enthalpy $\mathcal{H}$')
    plt.xlabel(r'Dim-less Composition $\mathcal{C}$')
    #plt.ylabel(r'$\mathcal{H}$')
    #plt.xlabel(r'$\mathcal{C}$')
    plt.axis('scaled')
    plt.legend(loc='upper left',borderaxespad=0.)
    plt.ylim([-0.2,1])
    plt.xlim([0,1])
    plt.savefig(f'../Figures/{simulation_name}_integral_curves_combined3.pdf')
    
    
def plotting_combined4(simulation_name,C_case_plot1,H_case_plot1,C_case_plot2,H_case_plot2,m,n): #R2 to R1
    H = np.linspace(0,1,10000)
    
    norm = colors.Normalize(vmin=0, vmax=1)
    fig = plt.figure(figsize=(7,10), dpi=100)
    
    
    #fancy plots
#    light_red  = [1.0,0.5,0.5]
    #light_blue = [0.5,0.5,1.0]
   # light_black= [0.5,0.5,0.5]

    #Region 2 Three phase region
    C1 = int_curves_lambda2(np.array([[0.05],[0.0]]),H, m, n)   #u0 = [C,H]^T
    C2 = int_curves_lambda2(np.array([[0.2],[0]]),H, m, n) 
    C3 = int_curves_lambda2(np.array([[0.4],[0]]),H, m, n) 
    C4 = int_curves_lambda2(np.array([[0.6],[0]]),H, m, n) 
    C5 = int_curves_lambda2(np.array([[0.8],[0]]),H, m, n) 
    
    C1_2 = int_curves_lambda1(np.array([[0.05],[0.05]]),H, m, n)   #u0 = [C,H]^T
    C2_2 = int_curves_lambda1(np.array([[0.2],[0.2]]),H, m, n)
    C3_2 = int_curves_lambda1(np.array([[0.4],[0.4]]),H, m, n) 
    C4_2 = int_curves_lambda1(np.array([[0.6],[0.6]]),H, m, n) 
    C5_2 = int_curves_lambda1(np.array([[0.8],[0.8]]),H, m, n) 
    
    #removing the curves outside the region
    C1_2[C1_2<H] = np.nan
    C2_2[C2_2<H] = np.nan
    C3_2[C3_2<H] = np.nan
    C4_2[C4_2<H] = np.nan
    C5_2[C5_2<H] = np.nan
    

    plt.plot(C1_2,H,'--',c=light_blue,label=r'Slow path, $\bf{r}_1$')
    plt.plot(C2_2,H,'--',c=light_blue)
    plt.plot(C3_2,H,'--',c=light_blue)
    plt.plot(C4_2,H,'--',c=light_blue)
    plt.plot(C5_2,H,'--',c=light_blue)
    plt.plot(C1,H,'-',c=light_red,label=r'Fast path, $\bf{r}_2$')
    plt.plot(C2,H,'-',c=light_red)
    plt.plot(C3,H,'-',c=light_red)
    plt.plot(C4,H,'-',c=light_red)
    plt.plot(C5,H,'-',c=light_red)
    
    
    plt.plot([0,1],[0,1],'k--',alpha=0.3)
    plt.plot([0,1],[0,0],'k--',alpha=0.3)
    plt.plot([1,1],[1,0],'k--',alpha=0.3)
    
    plt.ylim([-0.2,1])
    plt.xlim([0,1])
    plt.plot(C_case_plot1,H_case_plot1,'b-',linewidth=3,label='$\mathcal{S}$')
    plt.plot(C_case_plot2,H_case_plot2,'r-',linewidth=3,label='$\mathcal{C}_1\mathcal{S}_2$')


    plt.plot(C_case_plot1[0],H_case_plot1[0],'bv', markersize=20,markerfacecolor='none',markeredgewidth=3)
    plt.plot(C_case_plot1[-1],H_case_plot1[-1],'bo', markersize=20,markerfacecolor='none',markeredgewidth=3) 
    plt.plot(C_case_plot2[0],H_case_plot2[0],'rv', markersize=20,markerfacecolor='none',markeredgewidth=3)
    
    plt.plot(C_case_plot2[int(10000-1)],H_case_plot2[int(10000-1)],'rs', markersize=20,markerfacecolor='white',markeredgewidth=3)
    
    plt.plot(C_case_plot2[-1],H_case_plot2[-1],'ro', markersize=20,markerfacecolor='none',markeredgewidth=3)

    x_pos1 = C_case_plot1[int(len(C_case_plot1)/4)-1]
    y_pos1 = H_case_plot1[int(len(H_case_plot1)/4)-1]
    x_direct1 = C_case_plot1[int(len(C_case_plot1)/4)]-C_case_plot1[int(len(C_case_plot1)/4)-1]
    y_direct1 = H_case_plot1[int(len(H_case_plot1)/4)]-H_case_plot1[int(len(H_case_plot1)/4)-1]
    #plt.arrow(x_pos1, y_pos1, x_direct1, y_direct1,head_width=0.04, head_length=0.04, fc='b', ec='b')

    x_pos1 = C_case_plot1[int(3*len(C_case_plot1)/4)-1]
    y_pos1 = H_case_plot1[int(3*len(H_case_plot1)/4)-1]
    x_direct1 = C_case_plot1[int(3*len(C_case_plot1)/4)]-C_case_plot1[int(3*len(C_case_plot1)/4)-1]
    y_direct1 = H_case_plot1[int(3*len(H_case_plot1)/4)]-H_case_plot1[int(3*len(H_case_plot1)/4)-1]
    #plt.arrow(x_pos1, y_pos1, x_direct1, y_direct1,head_width=0.04, head_length=0.04, fc='b', ec='b')
    
    x_pos1 = C_case_plot2[int(len(C_case_plot2)/4)-1]
    y_pos1 = H_case_plot2[int(len(H_case_plot2)/4)-1]
    x_direct1 = C_case_plot2[int(len(C_case_plot2)/4)]-C_case_plot2[int(len(C_case_plot2)/4)-1]
    y_direct1 = H_case_plot2[int(len(H_case_plot2)/4)]-H_case_plot2[int(len(H_case_plot2)/4)-1]
    #plt.arrow(x_pos1, y_pos1, x_direct1, y_direct1,head_width=0.04, head_length=0.04, fc='r', ec='r')

    x_pos1 = C_case_plot2[int(3*len(C_case_plot2)/4)-1]
    y_pos1 = H_case_plot2[int(3*len(H_case_plot2)/4)-1]
    x_direct1 = C_case_plot2[int(3*len(C_case_plot2)/4)]-C_case_plot2[int(3*len(C_case_plot2)/4)-1]
    y_direct1 = H_case_plot2[int(3*len(H_case_plot2)/4)]-H_case_plot2[int(3*len(H_case_plot2)/4)-1]
    #plt.arrow(x_pos1, y_pos1, x_direct1, y_direct1,head_width=0.04, head_length=0.04, fc='r', ec='r')
    
    plt.ylabel(r'Dim-less Enthalpy $\mathcal{H}$')
    plt.xlabel(r'Dim-less Composition $\mathcal{C}$')
    #plt.ylabel(r'$\mathcal{H}$')
    #plt.xlabel(r'$\mathcal{C}$')
    plt.axis('scaled')
    plt.legend(loc='upper left',borderaxespad=0.)
    plt.ylim([-0.2,1])
    plt.xlim([0,1])
    plt.legend(loc='upper left',borderaxespad=0.)

    plt.savefig(f'../Figures/{simulation_name}_integral_curves_combined4.pdf')
    
    
def plotting_combined5(simulation_name,C_case_plot1,H_case_plot1,C_case_plot2,H_case_plot2,m,n): #Saturated
    H = np.linspace(0,1,10000)
    
    norm = colors.Normalize(vmin=0, vmax=1)
    fig = plt.figure(figsize=(7,10), dpi=100)
    ax = fig.add_subplot(111)
    
    #fancy plots
#    light_red  = [1.0,0.5,0.5]
    #light_blue = [0.5,0.5,1.0]
   # light_black= [0.5,0.5,0.5]

    #Region 2 Three phase region
    C1 = int_curves_lambda2(np.array([[0.05],[0.0]]),H, m, n)   #u0 = [C,H]^T
    C2 = int_curves_lambda2(np.array([[0.2],[0]]),H, m, n) 
    C3 = int_curves_lambda2(np.array([[0.4],[0]]),H, m, n) 
    C4 = int_curves_lambda2(np.array([[0.6],[0]]),H, m, n) 
    C5 = int_curves_lambda2(np.array([[0.8],[0]]),H, m, n) 
    
    C1_2 = int_curves_lambda1(np.array([[0.05],[0.05]]),H, m, n)   #u0 = [C,H]^T
    C2_2 = int_curves_lambda1(np.array([[0.2],[0.2]]),H, m, n)
    C3_2 = int_curves_lambda1(np.array([[0.4],[0.4]]),H, m, n) 
    C4_2 = int_curves_lambda1(np.array([[0.6],[0.6]]),H, m, n) 
    C5_2 = int_curves_lambda1(np.array([[0.8],[0.8]]),H, m, n) 
    
    #removing the curves outside the region
    C1_2[C1_2<H] = np.nan
    C2_2[C2_2<H] = np.nan
    C3_2[C3_2<H] = np.nan
    C4_2[C4_2<H] = np.nan
    C5_2[C5_2<H] = np.nan
    

    C1[C1>1] = np.nan
    C2[C2>1] = np.nan
    C3[C3>1] = np.nan
    C4[C4>1] = np.nan
    C5[C5>1] = np.nan
    
    C1_2[C1_2>1] = np.nan
    C2_2[C2_2>1] = np.nan
    C3_2[C3_2>1] = np.nan
    C4_2[C4_2>1] = np.nan
    C5_2[C5_2>1] = np.nan
    

    plt.plot(C1_2,H,'--',c=light_blue,label=r'Slow path, $\bf{r}_1$')
    plt.plot(C2_2,H,'--',c=light_blue)
    plt.plot(C3_2,H,'--',c=light_blue)
    plt.plot(C4_2,H,'--',c=light_blue)
    plt.plot(C5_2,H,'--',c=light_blue)
    plt.plot(C1,H,'-',c=light_red,label=r'Fast path, $\bf{r}_2$')
    plt.plot(C2,H,'-',c=light_red)
    plt.plot(C3,H,'-',c=light_red)
    plt.plot(C4,H,'-',c=light_red)
    plt.plot(C5,H,'-',c=light_red)
    
    
    plt.plot([0,1],[0,1],'k--',alpha=0.3)
    plt.plot([0,1],[0,0],'k--',alpha=0.3)
    plt.plot([1,1],[1,0],'k--',alpha=0.3)
    
    plt.ylim([-0.2,1])
    plt.xlim([0,1])
    plt.plot(C_case_plot1,H_case_plot1,'b-',linewidth=3)


    plt.plot(C_case_plot1[0],H_case_plot1[0],'bv', markersize=20,markerfacecolor='none',markeredgewidth=3)
    plt.plot(C_case_plot1[6666],H_case_plot1[6666],'bd', markersize=20,markerfacecolor='none',markeredgewidth=3)
    plt.plot(C_case_plot1[13332],H_case_plot1[13332],'bs', markersize=20,markerfacecolor='none',markeredgewidth=3)
    plt.plot(C_case_plot1[-1],H_case_plot1[-1],'bo', markersize=20,markerfacecolor='none',markeredgewidth=3) 
    #plt.plot(C_case_plot2[0],H_case_plot2[0],'rv', markersize=20,markerfacecolor='none',markeredgewidth=3)
    
    #plt.plot(C_case_plot2[int(10000-1)],H_case_plot2[int(10000-1)],'rs', markersize=20,markerfacecolor='white',markeredgewidth=3)

    x_pos1 = C_case_plot1[1111-1]
    y_pos1 = H_case_plot1[1111-1]
    x_direct1 = C_case_plot1[1111]-C_case_plot1[1111-1]
    y_direct1 = H_case_plot1[1111]-H_case_plot1[1111-1]
    #plt.arrow(x_pos1, y_pos1, x_direct1, y_direct1,head_width=0.04, head_length=0.04, fc='b', ec='b')

    x_pos1 = C_case_plot1[9999-1]
    y_pos1 = H_case_plot1[9999-1]
    x_direct1 = C_case_plot1[9999]-C_case_plot1[9999-1]
    y_direct1 = H_case_plot1[9999]-H_case_plot1[9999-1]
    #plt.arrow(x_pos1, y_pos1, x_direct1, y_direct1,head_width=0.04, head_length=0.04, fc='b', ec='b')
    
    x_pos1 = C_case_plot1[16666-1]
    y_pos1 = H_case_plot1[16666-1]
    x_direct1 = C_case_plot1[16666]-C_case_plot1[16666-1]
    y_direct1 = H_case_plot1[16666]-H_case_plot1[16666-1]
    #plt.arrow(x_pos1, y_pos1, x_direct1, y_direct1,head_width=0.04, head_length=0.04, fc='b', ec='b')    

    
    plt.ylabel(r'Dim-less Enthalpy $\mathcal{H}$')
    plt.xlabel(r'Dim-less Composition $\mathcal{C}$')
    #plt.ylabel(r'$\mathcal{H}$')
    #plt.xlabel(r'$\mathcal{C}$')
    plt.axis('scaled')
    plt.legend(loc='upper left',borderaxespad=0.)
    plt.ylim([-0.2,1])
    plt.xlim([0,1.02])

    plt.savefig(f'../Figures/{simulation_name}_integral_curves_combined5.pdf')
    
    
def plotting_combined5AGU(simulation_name,C_case_plot1,H_case_plot1,C_case_plot2,H_case_plot2,m,n): #Saturated
    H = np.linspace(0,1,10000)
    
    norm = colors.Normalize(vmin=0, vmax=1)
    fig = plt.figure(figsize=(7,10), dpi=100)
    ax = fig.add_subplot(111)
    
    #fancy plots
#    light_red  = [1.0,0.5,0.5]
    #light_blue = [0.5,0.5,1.0]
   # light_black= [0.5,0.5,0.5]

    #Region 2 Three phase region
    C1 = int_curves_lambda2(np.array([[0.05],[0.0]]),H, m, n)   #u0 = [C,H]^T
    C2 = int_curves_lambda2(np.array([[0.2],[0]]),H, m, n) 
    C3 = int_curves_lambda2(np.array([[0.4],[0]]),H, m, n) 
    C4 = int_curves_lambda2(np.array([[0.6],[0]]),H, m, n) 
    C5 = int_curves_lambda2(np.array([[0.8],[0]]),H, m, n) 
    
    C1_2 = int_curves_lambda1(np.array([[0.05],[0.05]]),H, m, n)   #u0 = [C,H]^T
    C2_2 = int_curves_lambda1(np.array([[0.2],[0.2]]),H, m, n)
    C3_2 = int_curves_lambda1(np.array([[0.4],[0.4]]),H, m, n) 
    C4_2 = int_curves_lambda1(np.array([[0.6],[0.6]]),H, m, n) 
    C5_2 = int_curves_lambda1(np.array([[0.8],[0.8]]),H, m, n) 
    
    #removing the curves outside the region
    C1_2[C1_2<H] = np.nan
    C2_2[C2_2<H] = np.nan
    C3_2[C3_2<H] = np.nan
    C4_2[C4_2<H] = np.nan
    C5_2[C5_2<H] = np.nan
    

    C1[C1>1] = np.nan
    C2[C2>1] = np.nan
    C3[C3>1] = np.nan
    C4[C4>1] = np.nan
    C5[C5>1] = np.nan
    
    C1_2[C1_2>1] = np.nan
    C2_2[C2_2>1] = np.nan
    C3_2[C3_2>1] = np.nan
    C4_2[C4_2>1] = np.nan
    C5_2[C5_2>1] = np.nan
    

    plt.plot(C1_2,H,'--',c=light_blue,label=r'Slow path, $\bf{r}_1$')
    plt.plot(C2_2,H,'--',c=light_blue)
    plt.plot(C3_2,H,'--',c=light_blue)
    plt.plot(C4_2,H,'--',c=light_blue)
    plt.plot(C5_2,H,'--',c=light_blue)
    plt.plot(C1,H,'-',c=light_red,label=r'Fast path, $\bf{r}_2$')
    plt.plot(C2,H,'-',c=light_red)
    plt.plot(C3,H,'-',c=light_red)
    plt.plot(C4,H,'-',c=light_red)
    plt.plot(C5,H,'-',c=light_red)
    
    
    plt.plot([0,1],[0,1],'k--',alpha=0.3)
    plt.plot([0,1],[0,0],'k--',alpha=0.3)
    plt.plot([1,1],[1,0],'k--',alpha=0.3)
    
    plt.ylim([-0.2,1])
    plt.xlim([0,1])
    plt.plot(C_case_plot1,H_case_plot1,'b-',linewidth=3)
    plt.plot(C_case_plot1,H_case_plot1,'r--',linewidth=3)

    plt.plot(C_case_plot1[0],H_case_plot1[0],'bv', markersize=20,markerfacecolor='none',markeredgewidth=3)
    plt.plot(C_case_plot1[6666],H_case_plot1[6666],'bd', markersize=20,markerfacecolor='none',markeredgewidth=3)
    plt.plot(C_case_plot1[13332],H_case_plot1[13332],'bs', markersize=20,markerfacecolor='none',markeredgewidth=3)
    plt.plot(C_case_plot1[-1],H_case_plot1[-1],'bo', markersize=20,markerfacecolor='none',markeredgewidth=3) 
    #plt.plot(C_case_plot2[0],H_case_plot2[0],'rv', markersize=20,markerfacecolor='none',markeredgewidth=3)
    
    #plt.plot(C_case_plot2[int(10000-1)],H_case_plot2[int(10000-1)],'rs', markersize=20,markerfacecolor='white',markeredgewidth=3)

    x_pos1 = C_case_plot1[1111-1]
    y_pos1 = H_case_plot1[1111-1]
    x_direct1 = C_case_plot1[1111]-C_case_plot1[1111-1]
    y_direct1 = H_case_plot1[1111]-H_case_plot1[1111-1]
    #plt.arrow(x_pos1, y_pos1, x_direct1, y_direct1,head_width=0.04, head_length=0.04, fc='k', ec='k')

    x_pos1 = C_case_plot1[9999-1]
    y_pos1 = H_case_plot1[9999-1]
    x_direct1 = C_case_plot1[9999]-C_case_plot1[9999-1]
    y_direct1 = H_case_plot1[9999]-H_case_plot1[9999-1]
    #plt.arrow(x_pos1, y_pos1, x_direct1, y_direct1,head_width=0.04, head_length=0.04, fc='k', ec='k')
    
    x_pos1 = C_case_plot1[16666-1]
    y_pos1 = H_case_plot1[16666-1]
    x_direct1 = C_case_plot1[16666]-C_case_plot1[16666-1]
    y_direct1 = H_case_plot1[16666]-H_case_plot1[16666-1]
    #plt.arrow(x_pos1, y_pos1, x_direct1, y_direct1,head_width=0.04, head_length=0.04, fc='k', ec='k')    


    ###Ice lens formation region
    pp3 = plt.Polygon([[1, 0],
                   [1, -0.3],
                   [0.7, -0.3]],facecolor='black', alpha=0.1)
  
    # depict illustrations
    ax.add_patch(pp3)   
    
    plt.ylabel(r'Dim-less Enthalpy $\mathcal{H}$')
    plt.xlabel(r'Dim-less Composition $\mathcal{C}$')
    #plt.ylabel(r'$\mathcal{H}$')
    #plt.xlabel(r'$\mathcal{C}$')
    plt.axis('scaled')
    plt.legend(loc='upper left',borderaxespad=0.)
    plt.ylim([-0.2,1])
    plt.xlim([0,1.02])

    plt.savefig(f'../Figures/{simulation_name}_integral_curves_combined5.pdf')
    
    
def plotting_combined6(simulation_name,C_case_plot1,H_case_plot1,C_case_plot2,H_case_plot2,m,n): #Saturated
    H = np.linspace(0,1,10000)
    
    norm = colors.Normalize(vmin=0, vmax=1)
    fig = plt.figure(figsize=(7,10), dpi=100)
    ax = fig.add_subplot(111)
    
    #fancy plots
#    light_red  = [1.0,0.5,0.5]
    #light_blue = [0.5,0.5,1.0]
   # light_black= [0.5,0.5,0.5]

    #Region 2 Three phase region
    C1 = int_curves_lambda2(np.array([[0.05],[0.0]]),H, m, n)   #u0 = [C,H]^T
    C2 = int_curves_lambda2(np.array([[0.2],[0]]),H, m, n) 
    C3 = int_curves_lambda2(np.array([[0.4],[0]]),H, m, n) 
    C4 = int_curves_lambda2(np.array([[0.6],[0]]),H, m, n) 
    C5 = int_curves_lambda2(np.array([[0.8],[0]]),H, m, n) 
    
    C1_2 = int_curves_lambda1(np.array([[0.05],[0.05]]),H, m, n)   #u0 = [C,H]^T
    C2_2 = int_curves_lambda1(np.array([[0.2],[0.2]]),H, m, n)
    C3_2 = int_curves_lambda1(np.array([[0.4],[0.4]]),H, m, n) 
    C4_2 = int_curves_lambda1(np.array([[0.6],[0.6]]),H, m, n) 
    C5_2 = int_curves_lambda1(np.array([[0.8],[0.8]]),H, m, n) 
    
    #removing the curves outside the region
    C1_2[C1_2<H] = np.nan
    C2_2[C2_2<H] = np.nan
    C3_2[C3_2<H] = np.nan
    C4_2[C4_2<H] = np.nan
    C5_2[C5_2<H] = np.nan
    

    C1[C1>1] = np.nan
    C2[C2>1] = np.nan
    C3[C3>1] = np.nan
    C4[C4>1] = np.nan
    C5[C5>1] = np.nan
    
    C1_2[C1_2>1] = np.nan
    C2_2[C2_2>1] = np.nan
    C3_2[C3_2>1] = np.nan
    C4_2[C4_2>1] = np.nan
    C5_2[C5_2>1] = np.nan
    

    plt.plot(C1_2,H,'--',c=light_blue,label=r'Slow path, $\bf{r}_1$')
    plt.plot(C2_2,H,'--',c=light_blue)
    plt.plot(C3_2,H,'--',c=light_blue)
    plt.plot(C4_2,H,'--',c=light_blue)
    plt.plot(C5_2,H,'--',c=light_blue)
    plt.plot(C1,H,'-',c=light_red,label=r'Fast path, $\bf{r}_2$')
    plt.plot(C2,H,'-',c=light_red)
    plt.plot(C3,H,'-',c=light_red)
    plt.plot(C4,H,'-',c=light_red)
    plt.plot(C5,H,'-',c=light_red)
    
    
    plt.plot([0,1],[0,1],'k--',alpha=0.3)
    plt.plot([0,1],[0,0],'k--',alpha=0.3)
    plt.plot([1,1],[1,0],'k--',alpha=0.3)
    
    plt.ylim([-0.2,1])
    plt.xlim([0,1])
    plt.plot(C_case_plot1,H_case_plot1,'b-',linewidth=3)


    plt.plot(C_case_plot1[0],H_case_plot1[0],'bv', markersize=20,markerfacecolor='none',markeredgewidth=3)
    plt.plot(C_case_plot1[6666],H_case_plot1[6666],'bd', markersize=20,markerfacecolor='none',markeredgewidth=3)
    plt.plot(C_case_plot1[13332],H_case_plot1[13332],'bs', markersize=20,markerfacecolor='none',markeredgewidth=3)
    plt.plot(C_case_plot1[-1],H_case_plot1[-1],'bo', markersize=20,markerfacecolor='none',markeredgewidth=3) 
    #plt.plot(C_case_plot2[0],H_case_plot2[0],'rv', markersize=20,markerfacecolor='none',markeredgewidth=3)
    
    #plt.plot(C_case_plot2[int(10000-1)],H_case_plot2[int(10000-1)],'rs', markersize=20,markerfacecolor='white',markeredgewidth=3)

    x_pos1 = C_case_plot1[1111-1]
    y_pos1 = H_case_plot1[1111-1]
    x_direct1 = C_case_plot1[1111]-C_case_plot1[1111-1]
    y_direct1 = H_case_plot1[1111]-H_case_plot1[1111-1]
    #plt.arrow(x_pos1, y_pos1, x_direct1, y_direct1,head_width=0.04, head_length=0.04, fc='b', ec='b')

    x_pos1 = C_case_plot1[9999-1]
    y_pos1 = H_case_plot1[9999-1]
    x_direct1 = C_case_plot1[9999]-C_case_plot1[9999-1]
    y_direct1 = H_case_plot1[9999]-H_case_plot1[9999-1]
    #plt.arrow(x_pos1, y_pos1, x_direct1, y_direct1,head_width=0.04, head_length=0.04, fc='b', ec='b')
    
    x_pos1 = C_case_plot1[16666-1]
    y_pos1 = H_case_plot1[16666-1]
    x_direct1 = C_case_plot1[16666]-C_case_plot1[16666-1]
    y_direct1 = H_case_plot1[16666]-H_case_plot1[16666-1]
    #plt.arrow(x_pos1, y_pos1, x_direct1, y_direct1,head_width=0.04, head_length=0.04, fc='b', ec='b')    

    ###Ice lens formation region
    pp3 = plt.Polygon([[1, 0],
                   [1, -0.3],
                   [0.7, -0.3]],facecolor='black', alpha=0.1)
  
    # depict illustrations
    ax.add_patch(pp3)   
    
    plt.ylabel(r'Dim-less Enthalpy $\mathcal{H}$')
    plt.xlabel(r'Dim-less Composition $\mathcal{C}$')
    #plt.ylabel(r'$\mathcal{H}$')
    #plt.xlabel(r'$\mathcal{C}$')
    plt.axis('scaled')
    plt.legend(loc='upper left',borderaxespad=0.)
    plt.ylim([-0.2,1])
    plt.xlim([0,1.02])

    plt.savefig(f'../Figures/{simulation_name}_integral_curves_combined6.pdf')
    
    
def plotting_combined6AGU(simulation_name,C_case_plot1,H_case_plot1,C_case_plot2,H_case_plot2,m,n): #Saturated
    H = np.linspace(0,1,10000)
    
    norm = colors.Normalize(vmin=0, vmax=1)
    fig = plt.figure(figsize=(7,10), dpi=100)
    ax = fig.add_subplot(111)
    
    #fancy plots
#    light_red  = [1.0,0.5,0.5]
    #light_blue = [0.5,0.5,1.0]
   # light_black= [0.5,0.5,0.5]

    #Region 2 Three phase region
    C1 = int_curves_lambda2(np.array([[0.05],[0.0]]),H, m, n)   #u0 = [C,H]^T
    C2 = int_curves_lambda2(np.array([[0.2],[0]]),H, m, n) 
    C3 = int_curves_lambda2(np.array([[0.4],[0]]),H, m, n) 
    C4 = int_curves_lambda2(np.array([[0.6],[0]]),H, m, n) 
    C5 = int_curves_lambda2(np.array([[0.8],[0]]),H, m, n) 
    
    C1_2 = int_curves_lambda1(np.array([[0.05],[0.05]]),H, m, n)   #u0 = [C,H]^T
    C2_2 = int_curves_lambda1(np.array([[0.2],[0.2]]),H, m, n)
    C3_2 = int_curves_lambda1(np.array([[0.4],[0.4]]),H, m, n) 
    C4_2 = int_curves_lambda1(np.array([[0.6],[0.6]]),H, m, n) 
    C5_2 = int_curves_lambda1(np.array([[0.8],[0.8]]),H, m, n) 
    
    #removing the curves outside the region
    C1_2[C1_2<H] = np.nan
    C2_2[C2_2<H] = np.nan
    C3_2[C3_2<H] = np.nan
    C4_2[C4_2<H] = np.nan
    C5_2[C5_2<H] = np.nan
    

    C1[C1>1] = np.nan
    C2[C2>1] = np.nan
    C3[C3>1] = np.nan
    C4[C4>1] = np.nan
    C5[C5>1] = np.nan
    
    C1_2[C1_2>1] = np.nan
    C2_2[C2_2>1] = np.nan
    C3_2[C3_2>1] = np.nan
    C4_2[C4_2>1] = np.nan
    C5_2[C5_2>1] = np.nan
    

    plt.plot(C1_2,H,'--',c=light_blue,label=r'Slow path, $\bf{r}_1$')
    plt.plot(C2_2,H,'--',c=light_blue)
    plt.plot(C3_2,H,'--',c=light_blue)
    plt.plot(C4_2,H,'--',c=light_blue)
    plt.plot(C5_2,H,'--',c=light_blue)
    plt.plot(C1,H,'-',c=light_red,label=r'Fast path, $\bf{r}_2$')
    plt.plot(C2,H,'-',c=light_red)
    plt.plot(C3,H,'-',c=light_red)
    plt.plot(C4,H,'-',c=light_red)
    plt.plot(C5,H,'-',c=light_red)
    
    
    plt.plot([0,1],[0,1],'k--',alpha=0.3)
    plt.plot([0,1],[0,0],'k--',alpha=0.3)
    plt.plot([1,1],[1,0],'k--',alpha=0.3)
    
    plt.ylim([-0.2,1])
    plt.xlim([0,1])
    plt.plot(C_case_plot1,H_case_plot1,'b-',linewidth=3)
    plt.plot(C_case_plot1,H_case_plot1,'r--',linewidth=3)

    plt.plot(C_case_plot1[0],H_case_plot1[0],'kv', markersize=20,markerfacecolor='none',markeredgewidth=3)
    plt.plot(C_case_plot1[6666],H_case_plot1[6666],'kd', markersize=20,markerfacecolor='none',markeredgewidth=3)
    plt.plot(C_case_plot1[13332],H_case_plot1[13332],'ks', markersize=20,markerfacecolor='none',markeredgewidth=3)
    plt.plot(C_case_plot1[-1],H_case_plot1[-1],'ko', markersize=20,markerfacecolor='none',markeredgewidth=3) 
    #plt.plot(C_case_plot2[0],H_case_plot2[0],'kv', markersize=20,markerfacecolor='none',markeredgewidth=3)
    
    #plt.plot(C_case_plot2[int(10000-1)],H_case_plot2[int(10000-1)],'rs', markersize=20,markerfacecolor='white',markeredgewidth=3)

    x_pos1 = C_case_plot1[1111-1]
    y_pos1 = H_case_plot1[1111-1]
    x_direct1 = C_case_plot1[1111]-C_case_plot1[1111-1]
    y_direct1 = H_case_plot1[1111]-H_case_plot1[1111-1]
    #plt.arrow(x_pos1, y_pos1, x_direct1, y_direct1,head_width=0.04, head_length=0.04, fc='k', ec='k')

    x_pos1 = C_case_plot1[9999-1]
    y_pos1 = H_case_plot1[9999-1]
    x_direct1 = C_case_plot1[9999]-C_case_plot1[9999-1]
    y_direct1 = H_case_plot1[9999]-H_case_plot1[9999-1]
    #plt.arrow(x_pos1, y_pos1, x_direct1, y_direct1,head_width=0.04, head_length=0.04, fc='k', ec='k')
    
    x_pos1 = C_case_plot1[16666-1]
    y_pos1 = H_case_plot1[16666-1]
    x_direct1 = C_case_plot1[16666]-C_case_plot1[16666-1]
    y_direct1 = H_case_plot1[16666]-H_case_plot1[16666-1]
    #plt.arrow(x_pos1, y_pos1, x_direct1, y_direct1,head_width=0.04, head_length=0.04, fc='k', ec='k')    

    ###Ice lens formation region
    pp3 = plt.Polygon([[1, 0],
                   [1, -0.3],
                   [0.7, -0.3]],facecolor='black', alpha=0.1)
  
    # depict illustrations
    ax.add_patch(pp3)   
    
    plt.ylabel(r'Dim-less Enthalpy $\mathcal{H}$')
    plt.xlabel(r'Dim-less Composition $\mathcal{C}$')
    #plt.ylabel(r'$\mathcal{H}$')
    #plt.xlabel(r'$\mathcal{C}$')
    plt.axis('scaled')
    plt.legend(loc='upper left',borderaxespad=0.)
    plt.ylim([-0.2,1])
    plt.xlim([0,1.02])

    plt.savefig(f'../Figures/{simulation_name}_integral_curves_combined6.pdf')
    
    
def plotting_combined7(simulation_name,C_case_plot1,H_case_plot1,C_case_plot2,H_case_plot2,m,n): #Saturated
    H = np.linspace(0,1,10000)
    
    norm = colors.Normalize(vmin=0, vmax=1)
    fig = plt.figure(figsize=(7,10), dpi=100)
    ax = fig.add_subplot(111)
    
    #fancy plots
#    light_red  = [1.0,0.5,0.5]
    #light_blue = [0.5,0.5,1.0]
   # light_black= [0.5,0.5,0.5]

    #Region 2 Three phase region
    C1 = int_curves_lambda2(np.array([[0.05],[0.0]]),H, m, n)   #u0 = [C,H]^T
    C2 = int_curves_lambda2(np.array([[0.2],[0]]),H, m, n) 
    C3 = int_curves_lambda2(np.array([[0.4],[0]]),H, m, n) 
    C4 = int_curves_lambda2(np.array([[0.6],[0]]),H, m, n) 
    C5 = int_curves_lambda2(np.array([[0.8],[0]]),H, m, n) 
    
    C1_2 = int_curves_lambda1(np.array([[0.05],[0.05]]),H, m, n)   #u0 = [C,H]^T
    C2_2 = int_curves_lambda1(np.array([[0.2],[0.2]]),H, m, n)
    C3_2 = int_curves_lambda1(np.array([[0.4],[0.4]]),H, m, n) 
    C4_2 = int_curves_lambda1(np.array([[0.6],[0.6]]),H, m, n) 
    C5_2 = int_curves_lambda1(np.array([[0.8],[0.8]]),H, m, n) 
    
    #removing the curves outside the region
    C1_2[C1_2<H] = np.nan
    C2_2[C2_2<H] = np.nan
    C3_2[C3_2<H] = np.nan
    C4_2[C4_2<H] = np.nan
    C5_2[C5_2<H] = np.nan
    

    C1[C1>1] = np.nan
    C2[C2>1] = np.nan
    C3[C3>1] = np.nan
    C4[C4>1] = np.nan
    C5[C5>1] = np.nan
    
    C1_2[C1_2>1] = np.nan
    C2_2[C2_2>1] = np.nan
    C3_2[C3_2>1] = np.nan
    C4_2[C4_2>1] = np.nan
    C5_2[C5_2>1] = np.nan
    

    plt.plot(C1_2,H,'--',c=light_blue,label=r'Slow path, $\bf{r}_1$')
    plt.plot(C2_2,H,'--',c=light_blue)
    plt.plot(C3_2,H,'--',c=light_blue)
    plt.plot(C4_2,H,'--',c=light_blue)
    plt.plot(C5_2,H,'--',c=light_blue)
    plt.plot(C1,H,'-',c=light_red,label=r'Fast path, $\bf{r}_2$')
    plt.plot(C2,H,'-',c=light_red)
    plt.plot(C3,H,'-',c=light_red)
    plt.plot(C4,H,'-',c=light_red)
    plt.plot(C5,H,'-',c=light_red)
    
    
    plt.plot([0,1],[0,1],'k--',alpha=0.3)
    plt.plot([0,1],[0,0],'k--',alpha=0.3)
    plt.plot([1,1],[1,0],'k--',alpha=0.3)
    
    plt.ylim([-0.2,1])
    plt.xlim([0,1])
    plt.plot(C_case_plot1,H_case_plot1,'b-',linewidth=3)


    plt.plot(C_case_plot1[0],H_case_plot1[0],'bv', markersize=20,markerfacecolor='none',markeredgewidth=3)
    plt.plot(C_case_plot1[6666],H_case_plot1[6666],'bd', markersize=20,markerfacecolor='none',markeredgewidth=3)
    plt.plot(C_case_plot1[13332],H_case_plot1[13332],'bs', markersize=20,markerfacecolor='none',markeredgewidth=3)
    plt.plot(C_case_plot1[-1],H_case_plot1[-1],'bo', markersize=20,markerfacecolor='none',markeredgewidth=3) 
    #plt.plot(C_case_plot2[0],H_case_plot2[0],'rv', markersize=20,markerfacecolor='none',markeredgewidth=3)
    
    #plt.plot(C_case_plot2[int(10000-1)],H_case_plot2[int(10000-1)],'rs', markersize=20,markerfacecolor='white',markeredgewidth=3)

    x_pos1 = C_case_plot1[1111-1]
    y_pos1 = H_case_plot1[1111-1]
    x_direct1 = C_case_plot1[1111]-C_case_plot1[1111-1]
    y_direct1 = H_case_plot1[1111]-H_case_plot1[1111-1]
    #plt.arrow(x_pos1, y_pos1, x_direct1, y_direct1,head_width=0.04, head_length=0.04, fc='b', ec='b')

    x_pos1 = C_case_plot1[9999-1]
    y_pos1 = H_case_plot1[9999-1]
    x_direct1 = C_case_plot1[9999]-C_case_plot1[9999-1]
    y_direct1 = H_case_plot1[9999]-H_case_plot1[9999-1]
    #plt.arrow(x_pos1, y_pos1, x_direct1, y_direct1,head_width=0.04, head_length=0.04, fc='b', ec='b')
    
    x_pos1 = C_case_plot1[16666-1]
    y_pos1 = H_case_plot1[16666-1]
    x_direct1 = C_case_plot1[16666]-C_case_plot1[16666-1]
    y_direct1 = H_case_plot1[16666]-H_case_plot1[16666-1]
    #plt.arrow(x_pos1, y_pos1, x_direct1, y_direct1,head_width=0.04, head_length=0.04, fc='b', ec='b')    
    
    plt.ylabel(r'Dim-less Enthalpy $\mathcal{H}$')
    plt.xlabel(r'Dim-less Composition $\mathcal{C}$')
    #plt.ylabel(r'$\mathcal{H}$')
    #plt.xlabel(r'$\mathcal{C}$')
    plt.axis('scaled')
    plt.ylim([-0.2,1])
    plt.xlim([0,1.02])
    plt.legend(loc='upper left',borderaxespad=0.)

    plt.savefig(f'../Figures/{simulation_name}_integral_curves_combined7.pdf')
    
    
    #More plotting functions
    light_gray = [0.85,0.85,0.85]
    blue   = [ 30/255 ,144/255 , 255/255 ]

def plot_phiw_CH(C_analy1,H_analy1,eta,Ste,Cpr,case_no):
    fig, (ax1, ax2) = plt.subplots(1,2, sharey=True,figsize=(12,7) , dpi=100)
    #fig.suptitle('Vertically stacked subplots')
    
    #manager = plt.get_current_fig_manager()
    #manager.window.showMaximized()
    
    if np.isnan(H_analy1[1]) == False:
        ax1.plot(C_analy1,eta,'-',c=light_blue,label=r'$\mathcal{C}$ ')
        ax1.plot(H_analy1,eta,'-',c=light_black,label=r'$\mathcal{H}$')
        ax1.plot(T(H_analy1,C_analy1,Ste,Cpr),eta,'-',c=light_red,label=r'$\mathcal{T}$')

    ax2.fill_betweenx(eta, phi_i(H_analy1,C_analy1)+phi_w(H_analy1,C_analy1)+phi_g(H_analy1,C_analy1), facecolor='white',label=r'$\phi_g$')
    ax2.fill_betweenx(eta, phi_w(H_analy1,C_analy1)+phi_i(H_analy1,C_analy1), facecolor=blue,label=r'$\phi_w$')
    ax2.fill_betweenx(eta, phi_i(H_analy1,C_analy1), facecolor=light_gray,label=r'$\phi_i$')
    ax2.set_xlim([0, 1])
    ax2.set_ylim([np.max(eta),np.min(eta)])
    #ax3.plot(porosity(H_sol[:,Nt-1],C_sol[:,Nt-1]),Grid.xc/t[Nt-1],'k--',label=r'$\varphi$')
    #ax3.plot(saturation(H_sol[:,Nt-1],C_sol[:,Nt-1])[0],Grid.xc/t[Nt-1],'b--',label=r'$S_w$')
    #ax3.set_xlim([0, 1])
    
    ax1.set_ylabel(r'Dim-less depth / Dim-less time')
    ax1.set_xlabel(r'$\mathcal{C},\mathcal{H}$, Dim-less Temperature $\mathcal{T}$')
    ax2.set_xlabel(r'Volume fractions $\phi$')
    ax1.legend(loc='lower right',borderaxespad=0.)
    ax2.legend(loc='lower left',borderaxespad=0.)
    #ax3.legend(loc='upper right',borderaxespad=0.)
    plt.savefig(f"../Figures/{case_no}_combined.pdf")    


def plot_time_varying_ice_lensAGU(C_analy1,H_analy1,eta,Ste,Cpr,case_no):

    brown  = [181/255 , 101/255, 29/255]
    red    = [255/255 ,255/255 ,255/255 ]
    blue   = [ 30/255 ,144/255 , 255/255 ]
    green  = [  0/255 , 166/255 ,  81/255]
    orange = [247/255 , 148/255 ,  30/255]
    purple = [102/255 ,  45/255 , 145/255]
    brown  = [155/255 ,  118/255 ,  83/255]
    tan    = [199/255 , 178/255 , 153/255]
    gray   = [100/255 , 100/255 , 100/255]    

    # First set up the figure, the axis
    
    fig = plt.figure(figsize=(10,7.5) , dpi=100)
    ax1 = fig.add_subplot(1, 4, 1)
    ax2 = fig.add_subplot(1, 4, 2)
    ax3 = fig.add_subplot(1, 4, 3)
    ax4 = fig.add_subplot(1, 4, 4)
    
    ax1.set_ylabel(r'Dimensionless depth')
    ax1.set_xlim([0,1])
    
    ax2.set_xlim([0,1])
    ax2.axes.yaxis.set_visible(False)
    
    ax3.set_xlim([0,1])
    ax3.axes.yaxis.set_visible(False)

    ax4.set_xlim([0,1])
    ax4.axes.yaxis.set_visible(False)
    
    #manager = plt.get_current_fig_manager()
    #manager.window.showMaximized()
    
    kk = np.array([1e-6,0.08,0.16,0.24,1])  #time stamps
    
    ax1.set_ylim([np.max(eta)*kk[-1], np.min(eta)*kk[-1]])
    ax2.set_ylim([np.max(eta)*kk[-1], np.min(eta)*kk[-1]])
    ax3.set_ylim([np.max(eta)*kk[-1], np.min(eta)*kk[-1]]) 
    ax4.set_ylim([np.max(eta)*kk[-1], np.min(eta)*kk[-1]]) 

    ax1.set_ylim([1+0.1, 1-0.1])
    ax2.set_ylim([1+0.1, 1-0.1])
    ax3.set_ylim([1+0.1, 1-0.1]) 
    ax4.set_ylim([1+0.1, 1-0.1]) 
    
    
    k = kk[0] 
    
    print(kk[-1]*np.min(eta), k*np.min(eta), k*np.max(eta),kk[-1]*np.max(eta))
    left  = np.linspace(kk[-1]*np.min(eta)/(k+1e-10),np.min(eta),10000)
    right = np.linspace(np.max(eta),kk[-1]*np.max(eta)/(k+1e-10),10000)
    
    H_analy_left =np.ones_like(left)*H_analy1[0]
    C_analy_left=np.ones_like(left)*C_analy1[0]
    H_analy_right =np.ones_like(right)*H_analy1[-1]
    C_analy_right=np.ones_like(right)*C_analy1[-1] 
    
    H_analy1 = np.concatenate((H_analy_left,H_analy1 ,H_analy_right),axis=0)
    C_analy1 = np.concatenate((C_analy_left,C_analy1 ,C_analy_right),axis=0)    
    eta = np.concatenate((left,eta,right),axis=0)
    
    ###Add 1+
    ax1.fill_betweenx(1+k*eta, phi_i(H_analy1,C_analy1)+phi_w(H_analy1,C_analy1)+phi_g(H_analy1,C_analy1), facecolor='white',label=r'$\phi_g$')
    ax1.fill_betweenx(1+k*eta, phi_w(H_analy1,C_analy1)+phi_i(H_analy1,C_analy1), facecolor=blue,label=r'$\phi_w$')
    ax1.fill_betweenx(1+k*eta, phi_i(H_analy1,C_analy1), facecolor=light_gray,label=r'$\phi_i$')

    k =  kk[1]
    ax2.fill_betweenx(1+k*eta, phi_i(H_analy1,C_analy1)+phi_w(H_analy1,C_analy1)+phi_g(H_analy1,C_analy1), facecolor='white',label=r'$\phi_g$')
    ax2.fill_betweenx(1+k*eta, phi_w(H_analy1,C_analy1)+phi_i(H_analy1,C_analy1), facecolor=blue,label=r'$\phi_w$')
    ax2.fill_betweenx(1+k*eta, phi_i(H_analy1,C_analy1), facecolor=light_gray,label=r'$\phi_i$')    
    
    k =  kk[2]
    
    ax3.fill_betweenx(1+k*eta, phi_i(H_analy1,C_analy1)+phi_w(H_analy1,C_analy1)+phi_g(H_analy1,C_analy1), facecolor='white',label=r'$\phi_g$')
    ax3.fill_betweenx(1+k*eta, phi_w(H_analy1,C_analy1)+phi_i(H_analy1,C_analy1), facecolor=blue,label=r'$\phi_w$')
    ax3.fill_betweenx(1+k*eta, phi_i(H_analy1,C_analy1), facecolor=light_gray,label=r'$\phi_i$')
    
    k =  kk[3]
    
    ax4.fill_betweenx(1+k*eta, phi_i(H_analy1,C_analy1)+phi_w(H_analy1,C_analy1)+phi_g(H_analy1,C_analy1), facecolor='white',label=r'$\phi_g$')
    ax4.fill_betweenx(1+k*eta, phi_w(H_analy1,C_analy1)+phi_i(H_analy1,C_analy1), facecolor=blue,label=r'$\phi_w$')
    ax4.fill_betweenx(1+k*eta, phi_i(H_analy1,C_analy1), facecolor=light_gray,label=r'$\phi_i$')
    
    ax1.legend(loc='lower left', shadow=False, fontsize='medium')
    plt.subplots_adjust(wspace=0, hspace=0.2)
    ax1.set_title(r'$\tau=$%.2f'%(3.56+kk[0]))
    ax2.set_title(r'%.2f'%(3.56+kk[1]))
    ax3.set_title(r'%.2f'%(3.56+kk[2]))
    ax4.set_title(r'%.2f'%(3.56+kk[3]))
    plt.subplots_adjust(wspace=0.15, hspace=0)
    plt.savefig(f"../Figures/{case_no}_combined.pdf") 

def plot_time_varying_diff_times(C_analy1,H_analy1,eta,Ste,Cpr,case_no,times):
    brown  = [181/255 , 101/255, 29/255]
    red    = [255/255 ,255/255 ,255/255 ]
    blue   = [ 30/255 ,144/255 , 255/255 ]
    green  = [  0/255 , 166/255 ,  81/255]
    orange = [247/255 , 148/255 ,  30/255]
    purple = [102/255 ,  45/255 , 145/255]
    brown  = [155/255 ,  118/255 ,  83/255]
    tan    = [199/255 , 178/255 , 153/255]
    gray   = [100/255 , 100/255 , 100/255]    

    # First set up the figure, the axis
    
    fig = plt.figure(figsize=(10,7.5) , dpi=100)
    ax1 = fig.add_subplot(1, 4, 1)
    ax2 = fig.add_subplot(1, 4, 2)
    ax3 = fig.add_subplot(1, 4, 3)
    ax4 = fig.add_subplot(1, 4, 4)
    
    ax1.set_ylabel(r'Dimensionless depth')
    ax1.set_xlim([0,1])
    
    ax2.set_xlim([0,1])
    ax2.axes.yaxis.set_visible(False)
    
    ax3.set_xlim([0,1])
    ax3.axes.yaxis.set_visible(False)

    ax4.set_xlim([0,1])
    ax4.axes.yaxis.set_visible(False)
    
    #manager = plt.get_current_fig_manager()
    #manager.window.showMaximized()
    
    kk = times  #time stamps
    
    ax1.set_ylim([np.max(eta)*kk[-1], np.min(eta)*kk[-1]])
    ax2.set_ylim([np.max(eta)*kk[-1], np.min(eta)*kk[-1]])
    ax3.set_ylim([np.max(eta)*kk[-1], np.min(eta)*kk[-1]]) 
    ax4.set_ylim([np.max(eta)*kk[-1], np.min(eta)*kk[-1]]) 

    ax1.set_ylim([1.5, -0.1])
    ax2.set_ylim([1.5, -0.1])
    ax3.set_ylim([1.5, -0.1]) 
    ax4.set_ylim([1.5, -0.1]) 
    
    
    k = kk[0] 
    
    print(kk[-1]*np.min(eta), k*np.min(eta), k*np.max(eta),kk[-1]*np.max(eta))
    left  = np.linspace(kk[-1]*np.min(eta)/(k+1e-10),np.min(eta),10000)
    right = np.linspace(np.max(eta),kk[-1]*np.max(eta)/(k+1e-10),10000)
    
    H_analy_left =np.ones_like(left)*H_analy1[0]
    C_analy_left=np.ones_like(left)*C_analy1[0]
    H_analy_right =np.ones_like(right)*H_analy1[-1]
    C_analy_right=np.ones_like(right)*C_analy1[-1] 
    
    H_analy1 = np.concatenate((H_analy_left,H_analy1 ,H_analy_right),axis=0)
    C_analy1 = np.concatenate((C_analy_left,C_analy1 ,C_analy_right),axis=0)    
    eta = np.concatenate((left,eta,right),axis=0)
    
    ###Add 1+
    ax1.fill_betweenx(k*eta, phi_i(H_analy1,C_analy1)+phi_w(H_analy1,C_analy1)+phi_g(H_analy1,C_analy1), facecolor='white',label=r'$\phi_g$')
    ax1.fill_betweenx(k*eta, phi_w(H_analy1,C_analy1)+phi_i(H_analy1,C_analy1), facecolor=blue,label=r'$\phi_w$')
    ax1.fill_betweenx(k*eta, phi_i(H_analy1,C_analy1), facecolor=light_gray,label=r'$\phi_i$')

    k =  kk[1]
    ax2.fill_betweenx(k*eta, phi_i(H_analy1,C_analy1)+phi_w(H_analy1,C_analy1)+phi_g(H_analy1,C_analy1), facecolor='white',label=r'$\phi_g$')
    ax2.fill_betweenx(k*eta, phi_w(H_analy1,C_analy1)+phi_i(H_analy1,C_analy1), facecolor=blue,label=r'$\phi_w$')
    ax2.fill_betweenx(k*eta, phi_i(H_analy1,C_analy1), facecolor=light_gray,label=r'$\phi_i$')    
    
    k =  kk[2]
    
    ax3.fill_betweenx(k*eta, phi_i(H_analy1,C_analy1)+phi_w(H_analy1,C_analy1)+phi_g(H_analy1,C_analy1), facecolor='white',label=r'$\phi_g$')
    ax3.fill_betweenx(k*eta, phi_w(H_analy1,C_analy1)+phi_i(H_analy1,C_analy1), facecolor=blue,label=r'$\phi_w$')
    ax3.fill_betweenx(k*eta, phi_i(H_analy1,C_analy1), facecolor=light_gray,label=r'$\phi_i$')
    
    k =  kk[3]
    
    ax4.fill_betweenx(k*eta, phi_i(H_analy1,C_analy1)+phi_w(H_analy1,C_analy1)+phi_g(H_analy1,C_analy1), facecolor='white',label=r'$\phi_g$')
    ax4.fill_betweenx(k*eta, phi_w(H_analy1,C_analy1)+phi_i(H_analy1,C_analy1), facecolor=blue,label=r'$\phi_w$')
    ax4.fill_betweenx(k*eta, phi_i(H_analy1,C_analy1), facecolor=light_gray,label=r'$\phi_i$')
    
    ax1.legend(loc='center left', shadow=False, fontsize='medium')
    plt.subplots_adjust(wspace=0, hspace=0.2)
    ax1.set_title(r'$\tau=$%.2f'%(kk[0]))
    ax2.set_title(r'%.2f'%(kk[1]))
    ax3.set_title(r'%.2f'%(kk[2]))
    ax4.set_title(r'%.2f'%(kk[3]))
    plt.subplots_adjust(wspace=0.15, hspace=0)
    #plt.savefig(f"../Figures/{case_no}_combined.pdf")  
    #plt.savefig(f"../Figures/{case_no}_combined.svg")      
    

def plot_time_varying(C_analy1,H_analy1,eta,Ste,Cpr,case_no):

    brown  = [181/255 , 101/255, 29/255]
    red    = [255/255 ,255/255 ,255/255 ]
    blue   = [ 30/255 ,144/255 , 255/255 ]
    green  = [  0/255 , 166/255 ,  81/255]
    orange = [247/255 , 148/255 ,  30/255]
    purple = [102/255 ,  45/255 , 145/255]
    brown  = [155/255 ,  118/255 ,  83/255]
    tan    = [199/255 , 178/255 , 153/255]
    gray   = [100/255 , 100/255 , 100/255]    

    # First set up the figure, the axis
    
    fig = plt.figure(figsize=(10,7.5) , dpi=100)
    ax1 = fig.add_subplot(1, 4, 1)
    ax2 = fig.add_subplot(1, 4, 2)
    ax3 = fig.add_subplot(1, 4, 3)
    ax4 = fig.add_subplot(1, 4, 4)
    
    ax1.set_ylabel(r'Dimensionless depth')
    ax1.set_xlim([0,1])
    
    ax2.set_xlim([0,1])
    ax2.axes.yaxis.set_visible(False)
    
    ax3.set_xlim([0,1])
    ax3.axes.yaxis.set_visible(False)

    ax4.set_xlim([0,1])
    ax4.axes.yaxis.set_visible(False)
    
    #manager = plt.get_current_fig_manager()
    #manager.window.showMaximized()
    
    kk = np.array([1e-7,1.1866666666666668,2.3733333333333335,3.56])  #time stamps
    
    ax1.set_ylim([np.max(eta)*kk[-1], np.min(eta)*kk[-1]])
    ax2.set_ylim([np.max(eta)*kk[-1], np.min(eta)*kk[-1]])
    ax3.set_ylim([np.max(eta)*kk[-1], np.min(eta)*kk[-1]]) 
    ax4.set_ylim([np.max(eta)*kk[-1], np.min(eta)*kk[-1]]) 

    ax1.set_ylim([1.5, -0.1])
    ax2.set_ylim([1.5, -0.1])
    ax3.set_ylim([1.5, -0.1]) 
    ax4.set_ylim([1.5, -0.1]) 
    
    
    k = kk[0] 
    
    print(kk[-1]*np.min(eta), k*np.min(eta), k*np.max(eta),kk[-1]*np.max(eta))
    left  = np.linspace(kk[-1]*np.min(eta)/(k+1e-10),np.min(eta),10000)
    right = np.linspace(np.max(eta),kk[-1]*np.max(eta)/(k+1e-10),10000)
    
    H_analy_left =np.ones_like(left)*H_analy1[0]
    C_analy_left=np.ones_like(left)*C_analy1[0]
    H_analy_right =np.ones_like(right)*H_analy1[-1]
    C_analy_right=np.ones_like(right)*C_analy1[-1] 
    
    H_analy1 = np.concatenate((H_analy_left,H_analy1 ,H_analy_right),axis=0)
    C_analy1 = np.concatenate((C_analy_left,C_analy1 ,C_analy_right),axis=0)    
    eta = np.concatenate((left,eta,right),axis=0)
    
    ###Add 1+
    ax1.fill_betweenx(k*eta, phi_i(H_analy1,C_analy1)+phi_w(H_analy1,C_analy1)+phi_g(H_analy1,C_analy1), facecolor='white',label=r'$\phi_g$')
    ax1.fill_betweenx(k*eta, phi_w(H_analy1,C_analy1)+phi_i(H_analy1,C_analy1), facecolor=blue,label=r'$\phi_w$')
    ax1.fill_betweenx(k*eta, phi_i(H_analy1,C_analy1), facecolor=light_gray,label=r'$\phi_i$')

    k =  kk[1]
    ax2.fill_betweenx(k*eta, phi_i(H_analy1,C_analy1)+phi_w(H_analy1,C_analy1)+phi_g(H_analy1,C_analy1), facecolor='white',label=r'$\phi_g$')
    ax2.fill_betweenx(k*eta, phi_w(H_analy1,C_analy1)+phi_i(H_analy1,C_analy1), facecolor=blue,label=r'$\phi_w$')
    ax2.fill_betweenx(k*eta, phi_i(H_analy1,C_analy1), facecolor=light_gray,label=r'$\phi_i$')    
    
    k =  kk[2]
    
    ax3.fill_betweenx(k*eta, phi_i(H_analy1,C_analy1)+phi_w(H_analy1,C_analy1)+phi_g(H_analy1,C_analy1), facecolor='white',label=r'$\phi_g$')
    ax3.fill_betweenx(k*eta, phi_w(H_analy1,C_analy1)+phi_i(H_analy1,C_analy1), facecolor=blue,label=r'$\phi_w$')
    ax3.fill_betweenx(k*eta, phi_i(H_analy1,C_analy1), facecolor=light_gray,label=r'$\phi_i$')
    
    k =  kk[3]
    
    ax4.fill_betweenx(k*eta, phi_i(H_analy1,C_analy1)+phi_w(H_analy1,C_analy1)+phi_g(H_analy1,C_analy1), facecolor='white',label=r'$\phi_g$')
    ax4.fill_betweenx(k*eta, phi_w(H_analy1,C_analy1)+phi_i(H_analy1,C_analy1), facecolor=blue,label=r'$\phi_w$')
    ax4.fill_betweenx(k*eta, phi_i(H_analy1,C_analy1), facecolor=light_gray,label=r'$\phi_i$')
    
    ax1.legend(loc='lower left', shadow=False, fontsize='medium')
    plt.subplots_adjust(wspace=0, hspace=0.2)
    ax1.set_title(r'$\tau=$%.2f'%(kk[0]))
    ax2.set_title(r'%.2f'%(kk[1]))
    ax3.set_title(r'%.2f'%(kk[2]))
    ax4.set_title(r'%.2f'%(kk[3]))
    plt.subplots_adjust(wspace=0.15, hspace=0)
    plt.savefig(f"../Figures/{case_no}_combined.pdf")  
    
    
def plot_time_varying_lower_layer(C_analy1,H_analy1,eta,Ste,Cpr,case_no):  #when in continuity with single front for AGU

    brown  = [181/255 , 101/255, 29/255]
    red    = [255/255 ,255/255 ,255/255 ]
    blue   = [ 30/255 ,144/255 , 255/255 ]
    green  = [  0/255 , 166/255 ,  81/255]
    orange = [247/255 , 148/255 ,  30/255]
    purple = [102/255 ,  45/255 , 145/255]
    brown  = [155/255 ,  118/255 ,  83/255]
    tan    = [199/255 , 178/255 , 153/255]
    gray   = [100/255 , 100/255 , 100/255]    

    # First set up the figure, the axis
    
    fig = plt.figure(figsize=(10,7.5) , dpi=100)
    ax1 = fig.add_subplot(1, 4, 1)
    ax2 = fig.add_subplot(1, 4, 2)
    ax3 = fig.add_subplot(1, 4, 3)
    ax4 = fig.add_subplot(1, 4, 4)
    
    ax1.set_ylabel(r'Dimensionless depth')
    ax1.set_xlim([0,1])
    
    ax2.set_xlim([0,1])
    ax2.axes.yaxis.set_visible(False)
    
    ax3.set_xlim([0,1])
    ax3.axes.yaxis.set_visible(False)

    ax4.set_xlim([0,1])
    ax4.axes.yaxis.set_visible(False)
    
    #manager = plt.get_current_fig_manager()
    #manager.window.showMaximized()
    
    kk = np.array([1e-7,1.1866666666666668,2.3733333333333335,3.716])  #time stamps  #4.08
    
    ax1.set_ylim([np.max(eta)*kk[-1], np.min(eta)*kk[-1]])
    ax2.set_ylim([np.max(eta)*kk[-1], np.min(eta)*kk[-1]])
    ax3.set_ylim([np.max(eta)*kk[-1], np.min(eta)*kk[-1]]) 
    ax4.set_ylim([np.max(eta)*kk[-1], np.min(eta)*kk[-1]]) 
    
        
    ax1.set_ylim([1+0.5,-1.1+1])
    ax2.set_ylim([1+0.5,-1.1+1])
    ax3.set_ylim([1+0.5,-1.1+1]) 
    ax4.set_ylim([1+0.5,-1.1+1])

    k = kk[0] 
    
    print(kk[-1]*np.min(eta), k*np.min(eta), k*np.max(eta),kk[-1]*np.max(eta))
    left  = np.linspace(kk[-1]*np.min(eta)/(k+1e-10),np.min(eta),10000)
    right = np.linspace(np.max(eta),kk[-1]*np.max(eta)/(k+1e-10),10000)
    
    H_analy_left =np.ones_like(left)*H_analy1[0]
    C_analy_left=np.ones_like(left)*C_analy1[0]
    H_analy_right =np.ones_like(right)*H_analy1[-1]
    C_analy_right=np.ones_like(right)*C_analy1[-1] 
    
    H_analy1 = np.concatenate((H_analy_left,H_analy1 ,H_analy_right),axis=0)
    C_analy1 = np.concatenate((C_analy_left,C_analy1 ,C_analy_right),axis=0)    
    eta = np.concatenate((left,eta,right),axis=0)
    
    ###Add 1+
    ax1.fill_betweenx(1+k*eta, phi_i(H_analy1,C_analy1)+phi_w(H_analy1,C_analy1)+phi_g(H_analy1,C_analy1), facecolor='white',label=r'$\phi_g$')
    ax1.fill_betweenx(1+k*eta, phi_w(H_analy1,C_analy1)+phi_i(H_analy1,C_analy1), facecolor=blue,label=r'$\phi_w$')
    ax1.fill_betweenx(1+k*eta, phi_i(H_analy1,C_analy1), facecolor=light_gray,label=r'$\phi_i$')

    k =  kk[1]
    ax2.fill_betweenx(1+k*eta, phi_i(H_analy1,C_analy1)+phi_w(H_analy1,C_analy1)+phi_g(H_analy1,C_analy1), facecolor='white',label=r'$\phi_g$')
    ax2.fill_betweenx(1+k*eta, phi_w(H_analy1,C_analy1)+phi_i(H_analy1,C_analy1), facecolor=blue,label=r'$\phi_w$')
    ax2.fill_betweenx(1+k*eta, phi_i(H_analy1,C_analy1), facecolor=light_gray,label=r'$\phi_i$')    
    
    k =  kk[2]
    
    ax3.fill_betweenx(1+k*eta, phi_i(H_analy1,C_analy1)+phi_w(H_analy1,C_analy1)+phi_g(H_analy1,C_analy1), facecolor='white',label=r'$\phi_g$')
    ax3.fill_betweenx(1+k*eta, phi_w(H_analy1,C_analy1)+phi_i(H_analy1,C_analy1), facecolor=blue,label=r'$\phi_w$')
    ax3.fill_betweenx(1+k*eta, phi_i(H_analy1,C_analy1), facecolor=light_gray,label=r'$\phi_i$')
    
    k =  kk[3]
    
    ax4.fill_betweenx(1+k*eta, phi_i(H_analy1,C_analy1)+phi_w(H_analy1,C_analy1)+phi_g(H_analy1,C_analy1), facecolor='white',label=r'$\phi_g$')
    ax4.fill_betweenx(1+k*eta, phi_w(H_analy1,C_analy1)+phi_i(H_analy1,C_analy1), facecolor=blue,label=r'$\phi_w$')
    ax4.fill_betweenx(1+k*eta, phi_i(H_analy1,C_analy1), facecolor=light_gray,label=r'$\phi_i$')
    
    ax1.legend(loc='lower left', shadow=False, fontsize='medium')
    plt.subplots_adjust(wspace=0, hspace=0.2)
    ax1.set_title(r'$\tau=$%.2f'%(3.56+kk[0]))
    ax2.set_title(r'%.2f'%(3.56+kk[1]))
    ax3.set_title(r'%.2f'%(3.56+kk[2]))
    ax4.set_title(r'%.2f'%(3.56+kk[3]))
    plt.subplots_adjust(wspace=0.15, hspace=0)
    plt.savefig(f"../Figures/{case_no}_combined_later.pdf")  


#Making one huge subplot
light_gray = [0.85,0.85,0.85]
blue   = [ 30/255 ,144/255 , 255/255 ]

def plot_phiw_CH(C_analy1,H_analy1,eta,Ste,Cpr,case_no):
    fig, (ax1, ax2) = plt.subplots(1,2, sharey=True,figsize=(12,7) , dpi=100)
    #fig.suptitle('Vertically stacked subplots')
    
    #manager = plt.get_current_fig_manager()
    #manager.window.showMaximized()
    
    if np.isnan(H_analy1[1]) == False:
        ax1.plot(C_analy1,eta,'-',c=light_blue,label=r'$\mathcal{C}$ ')
        ax1.plot(H_analy1,eta,'-',c=light_black,label=r'$\mathcal{H}$')
        ax1.plot(T(H_analy1,C_analy1,Ste,Cpr),eta,'-',c=light_red,label=r'$\mathcal{T}$')

    ax2.fill_betweenx(eta, phi_i(H_analy1,C_analy1)+phi_w(H_analy1,C_analy1)+phi_g(H_analy1,C_analy1), facecolor='white',label=r'$\phi_g$')
    ax2.fill_betweenx(eta, phi_w(H_analy1,C_analy1)+phi_i(H_analy1,C_analy1), facecolor=blue,label=r'$\phi_w$')
    ax2.fill_betweenx(eta, phi_i(H_analy1,C_analy1), facecolor=light_gray,label=r'$\phi_i$')
    ax2.set_xlim([0, 1])
    ax2.set_ylim([np.max(eta),np.min(eta)])
    #ax3.plot(porosity(H_sol[:,Nt-1],C_sol[:,Nt-1]),Grid.xc/t[Nt-1],'k--',label=r'$\varphi$')
    #ax3.plot(saturation(H_sol[:,Nt-1],C_sol[:,Nt-1])[0],Grid.xc/t[Nt-1],'b--',label=r'$S_w$')
    #ax3.set_xlim([0, 1])
    
    ax1.set_ylabel(r'Dim-less depth / Dim-less time')
    ax1.set_xlabel(r'$\mathcal{C},\mathcal{H}$, Dim-less Temperature $\mathcal{T}$')
    ax2.set_xlabel(r'Volume fractions $\phi$')
    ax1.legend(loc='lower right',borderaxespad=0.)
    ax2.legend(loc='lower left',borderaxespad=0.)
    #ax3.legend(loc='upper right',borderaxespad=0.)
    plt.savefig(f"../Figures/{case_no}_combined.pdf")    


def plot_time_varying_ice_lensAGU(C_analy1,H_analy1,eta,Ste,Cpr,case_no):

    brown  = [181/255 , 101/255, 29/255]
    red    = [255/255 ,255/255 ,255/255 ]
    blue   = [ 30/255 ,144/255 , 255/255 ]
    green  = [  0/255 , 166/255 ,  81/255]
    orange = [247/255 , 148/255 ,  30/255]
    purple = [102/255 ,  45/255 , 145/255]
    brown  = [155/255 ,  118/255 ,  83/255]
    tan    = [199/255 , 178/255 , 153/255]
    gray   = [100/255 , 100/255 , 100/255]    

    # First set up the figure, the axis
    
    fig = plt.figure(figsize=(10,7.5) , dpi=100)
    ax1 = fig.add_subplot(1, 4, 1)
    ax2 = fig.add_subplot(1, 4, 2)
    ax3 = fig.add_subplot(1, 4, 3)
    ax4 = fig.add_subplot(1, 4, 4)
    
    ax1.set_ylabel(r'Dimensionless depth')
    ax1.set_xlim([0,1])
    
    ax2.set_xlim([0,1])
    ax2.axes.yaxis.set_visible(False)
    
    ax3.set_xlim([0,1])
    ax3.axes.yaxis.set_visible(False)

    ax4.set_xlim([0,1])
    ax4.axes.yaxis.set_visible(False)
    
    #manager = plt.get_current_fig_manager()
    #manager.window.showMaximized()
    
    kk = np.array([1e-6,0.08,0.16,0.24,1])  #time stamps
    
    ax1.set_ylim([np.max(eta)*kk[-1], np.min(eta)*kk[-1]])
    ax2.set_ylim([np.max(eta)*kk[-1], np.min(eta)*kk[-1]])
    ax3.set_ylim([np.max(eta)*kk[-1], np.min(eta)*kk[-1]]) 
    ax4.set_ylim([np.max(eta)*kk[-1], np.min(eta)*kk[-1]]) 

    ax1.set_ylim([1+0.1, 1-0.1])
    ax2.set_ylim([1+0.1, 1-0.1])
    ax3.set_ylim([1+0.1, 1-0.1]) 
    ax4.set_ylim([1+0.1, 1-0.1]) 
    
    
    k = kk[0] 
    
    print(kk[-1]*np.min(eta), k*np.min(eta), k*np.max(eta),kk[-1]*np.max(eta))
    left  = np.linspace(kk[-1]*np.min(eta)/(k+1e-10),np.min(eta),10000)
    right = np.linspace(np.max(eta),kk[-1]*np.max(eta)/(k+1e-10),10000)
    
    H_analy_left =np.ones_like(left)*H_analy1[0]
    C_analy_left=np.ones_like(left)*C_analy1[0]
    H_analy_right =np.ones_like(right)*H_analy1[-1]
    C_analy_right=np.ones_like(right)*C_analy1[-1] 
    
    H_analy1 = np.concatenate((H_analy_left,H_analy1 ,H_analy_right),axis=0)
    C_analy1 = np.concatenate((C_analy_left,C_analy1 ,C_analy_right),axis=0)    
    eta = np.concatenate((left,eta,right),axis=0)
    
    ###Add 1+
    ax1.fill_betweenx(1+k*eta, phi_i(H_analy1,C_analy1)+phi_w(H_analy1,C_analy1)+phi_g(H_analy1,C_analy1), facecolor='white',label=r'$\phi_g$')
    ax1.fill_betweenx(1+k*eta, phi_w(H_analy1,C_analy1)+phi_i(H_analy1,C_analy1), facecolor=blue,label=r'$\phi_w$')
    ax1.fill_betweenx(1+k*eta, phi_i(H_analy1,C_analy1), facecolor=light_gray,label=r'$\phi_i$')

    k =  kk[1]
    ax2.fill_betweenx(1+k*eta, phi_i(H_analy1,C_analy1)+phi_w(H_analy1,C_analy1)+phi_g(H_analy1,C_analy1), facecolor='white',label=r'$\phi_g$')
    ax2.fill_betweenx(1+k*eta, phi_w(H_analy1,C_analy1)+phi_i(H_analy1,C_analy1), facecolor=blue,label=r'$\phi_w$')
    ax2.fill_betweenx(1+k*eta, phi_i(H_analy1,C_analy1), facecolor=light_gray,label=r'$\phi_i$')    
    
    k =  kk[2]
    
    ax3.fill_betweenx(1+k*eta, phi_i(H_analy1,C_analy1)+phi_w(H_analy1,C_analy1)+phi_g(H_analy1,C_analy1), facecolor='white',label=r'$\phi_g$')
    ax3.fill_betweenx(1+k*eta, phi_w(H_analy1,C_analy1)+phi_i(H_analy1,C_analy1), facecolor=blue,label=r'$\phi_w$')
    ax3.fill_betweenx(1+k*eta, phi_i(H_analy1,C_analy1), facecolor=light_gray,label=r'$\phi_i$')
    
    k =  kk[3]
    
    ax4.fill_betweenx(1+k*eta, phi_i(H_analy1,C_analy1)+phi_w(H_analy1,C_analy1)+phi_g(H_analy1,C_analy1), facecolor='white',label=r'$\phi_g$')
    ax4.fill_betweenx(1+k*eta, phi_w(H_analy1,C_analy1)+phi_i(H_analy1,C_analy1), facecolor=blue,label=r'$\phi_w$')
    ax4.fill_betweenx(1+k*eta, phi_i(H_analy1,C_analy1), facecolor=light_gray,label=r'$\phi_i$')
    
    ax1.legend(loc='lower left', shadow=False, fontsize='medium')
    plt.subplots_adjust(wspace=0, hspace=0.2)
    ax1.set_title(r'$\tau=$%.2f'%(3.56+kk[0]))
    ax2.set_title(r'%.2f'%(3.56+kk[1]))
    ax3.set_title(r'%.2f'%(3.56+kk[2]))
    ax4.set_title(r'%.2f'%(3.56+kk[3]))
    plt.subplots_adjust(wspace=0.15, hspace=0)
    plt.savefig(f"../Figures/{case_no}_combined.pdf") 

def plot_time_varying(C_analy1,H_analy1,eta,Ste,Cpr,case_no):

    brown  = [181/255 , 101/255, 29/255]
    red    = [255/255 ,255/255 ,255/255 ]
    blue   = [ 30/255 ,144/255 , 255/255 ]
    green  = [  0/255 , 166/255 ,  81/255]
    orange = [247/255 , 148/255 ,  30/255]
    purple = [102/255 ,  45/255 , 145/255]
    brown  = [155/255 ,  118/255 ,  83/255]
    tan    = [199/255 , 178/255 , 153/255]
    gray   = [100/255 , 100/255 , 100/255]    

    # First set up the figure, the axis
    
    fig = plt.figure(figsize=(10,7.5) , dpi=100)
    ax1 = fig.add_subplot(1, 4, 1)
    ax2 = fig.add_subplot(1, 4, 2)
    ax3 = fig.add_subplot(1, 4, 3)
    ax4 = fig.add_subplot(1, 4, 4)
    
    ax1.set_ylabel(r'Dimensionless depth')
    ax1.set_xlim([0,1])
    
    ax2.set_xlim([0,1])
    ax2.axes.yaxis.set_visible(False)
    
    ax3.set_xlim([0,1])
    ax3.axes.yaxis.set_visible(False)

    ax4.set_xlim([0,1])
    ax4.axes.yaxis.set_visible(False)
    
    #manager = plt.get_current_fig_manager()
    #manager.window.showMaximized()
    
    kk = np.array([1e-7,1.1866666666666668,2.3733333333333335,3.56])  #time stamps
    
    ax1.set_ylim([np.max(eta)*kk[-1], np.min(eta)*kk[-1]])
    ax2.set_ylim([np.max(eta)*kk[-1], np.min(eta)*kk[-1]])
    ax3.set_ylim([np.max(eta)*kk[-1], np.min(eta)*kk[-1]]) 
    ax4.set_ylim([np.max(eta)*kk[-1], np.min(eta)*kk[-1]]) 

    ax1.set_ylim([1, -0.1])
    ax2.set_ylim([1, -0.1])
    ax3.set_ylim([1, -0.1]) 
    ax4.set_ylim([1, -0.1]) 
    
    
    k = kk[0] 
    
    print(kk[-1]*np.min(eta), k*np.min(eta), k*np.max(eta),kk[-1]*np.max(eta))
    left  = np.linspace(kk[-1]*np.min(eta)/(k+1e-10),np.min(eta),10000)
    right = np.linspace(np.max(eta),kk[-1]*np.max(eta)/(k+1e-10),10000)
    
    H_analy_left =np.ones_like(left)*H_analy1[0]
    C_analy_left=np.ones_like(left)*C_analy1[0]
    H_analy_right =np.ones_like(right)*H_analy1[-1]
    C_analy_right=np.ones_like(right)*C_analy1[-1] 
    
    H_analy1 = np.concatenate((H_analy_left,H_analy1 ,H_analy_right),axis=0)
    C_analy1 = np.concatenate((C_analy_left,C_analy1 ,C_analy_right),axis=0)    
    eta = np.concatenate((left,eta,right),axis=0)
    
    ###Add 1+
    ax1.fill_betweenx(k*eta, phi_i(H_analy1,C_analy1)+phi_w(H_analy1,C_analy1)+phi_g(H_analy1,C_analy1), facecolor='white',label=r'$\phi_g$')
    ax1.fill_betweenx(k*eta, phi_w(H_analy1,C_analy1)+phi_i(H_analy1,C_analy1), facecolor=blue,label=r'$\phi_w$')
    ax1.fill_betweenx(k*eta, phi_i(H_analy1,C_analy1), facecolor=light_gray,label=r'$\phi_i$')

    k =  kk[1]
    ax2.fill_betweenx(k*eta, phi_i(H_analy1,C_analy1)+phi_w(H_analy1,C_analy1)+phi_g(H_analy1,C_analy1), facecolor='white',label=r'$\phi_g$')
    ax2.fill_betweenx(k*eta, phi_w(H_analy1,C_analy1)+phi_i(H_analy1,C_analy1), facecolor=blue,label=r'$\phi_w$')
    ax2.fill_betweenx(k*eta, phi_i(H_analy1,C_analy1), facecolor=light_gray,label=r'$\phi_i$')    
    
    k =  kk[2]
    
    ax3.fill_betweenx(k*eta, phi_i(H_analy1,C_analy1)+phi_w(H_analy1,C_analy1)+phi_g(H_analy1,C_analy1), facecolor='white',label=r'$\phi_g$')
    ax3.fill_betweenx(k*eta, phi_w(H_analy1,C_analy1)+phi_i(H_analy1,C_analy1), facecolor=blue,label=r'$\phi_w$')
    ax3.fill_betweenx(k*eta, phi_i(H_analy1,C_analy1), facecolor=light_gray,label=r'$\phi_i$')
    
    k =  kk[3]
    
    ax4.fill_betweenx(k*eta, phi_i(H_analy1,C_analy1)+phi_w(H_analy1,C_analy1)+phi_g(H_analy1,C_analy1), facecolor='white',label=r'$\phi_g$')
    ax4.fill_betweenx(k*eta, phi_w(H_analy1,C_analy1)+phi_i(H_analy1,C_analy1), facecolor=blue,label=r'$\phi_w$')
    ax4.fill_betweenx(k*eta, phi_i(H_analy1,C_analy1), facecolor=light_gray,label=r'$\phi_i$')
    
    ax1.legend(loc='lower left', shadow=False, fontsize='medium')
    plt.subplots_adjust(wspace=0, hspace=0.2)
    ax1.set_title(r'$\tau=$%.2f'%(kk[0]))
    ax2.set_title(r'%.2f'%(kk[1]))
    ax3.set_title(r'%.2f'%(kk[2]))
    ax4.set_title(r'%.2f'%(kk[3]))
    plt.subplots_adjust(wspace=0.15, hspace=0)
    plt.savefig(f"../Figures/{case_no}_combined.pdf")  
    
    
def plot_time_varying_lower_layer(C_analy1,H_analy1,eta,Ste,Cpr,case_no):  #when in continuity with single front for AGU

    brown  = [181/255 , 101/255, 29/255]
    red    = [255/255 ,255/255 ,255/255 ]
    blue   = [ 30/255 ,144/255 , 255/255 ]
    green  = [  0/255 , 166/255 ,  81/255]
    orange = [247/255 , 148/255 ,  30/255]
    purple = [102/255 ,  45/255 , 145/255]
    brown  = [155/255 ,  118/255 ,  83/255]
    tan    = [199/255 , 178/255 , 153/255]
    gray   = [100/255 , 100/255 , 100/255]    

    # First set up the figure, the axis
    
    fig = plt.figure(figsize=(10,7.5) , dpi=100)
    ax1 = fig.add_subplot(1, 4, 1)
    ax2 = fig.add_subplot(1, 4, 2)
    ax3 = fig.add_subplot(1, 4, 3)
    ax4 = fig.add_subplot(1, 4, 4)
    
    ax1.set_ylabel(r'Dimensionless depth')
    ax1.set_xlim([0,1])
    
    ax2.set_xlim([0,1])
    ax2.axes.yaxis.set_visible(False)
    
    ax3.set_xlim([0,1])
    ax3.axes.yaxis.set_visible(False)

    ax4.set_xlim([0,1])
    ax4.axes.yaxis.set_visible(False)
    
    #manager = plt.get_current_fig_manager()
    #manager.window.showMaximized()
    
    kk = np.array([1e-7,1.1866666666666668,2.3733333333333335,3.56])  #time stamps
    
    ax1.set_ylim([np.max(eta)*kk[-1], np.min(eta)*kk[-1]])
    ax2.set_ylim([np.max(eta)*kk[-1], np.min(eta)*kk[-1]])
    ax3.set_ylim([np.max(eta)*kk[-1], np.min(eta)*kk[-1]]) 
    ax4.set_ylim([np.max(eta)*kk[-1], np.min(eta)*kk[-1]]) 
    
        
    ax1.set_ylim([1+0.5,1-1])
    ax2.set_ylim([1+0.5,1-1])
    ax3.set_ylim([1+0.5,1-1]) 
    ax4.set_ylim([1+0.5,1-1])

    k = kk[0] 
    
    print(kk[-1]*np.min(eta), k*np.min(eta), k*np.max(eta),kk[-1]*np.max(eta))
    left  = np.linspace(kk[-1]*np.min(eta)/(k+1e-10),np.min(eta),10000)
    right = np.linspace(np.max(eta),kk[-1]*np.max(eta)/(k+1e-10),10000)
    
    H_analy_left =np.ones_like(left)*H_analy1[0]
    C_analy_left=np.ones_like(left)*C_analy1[0]
    H_analy_right =np.ones_like(right)*H_analy1[-1]
    C_analy_right=np.ones_like(right)*C_analy1[-1] 
    
    H_analy1 = np.concatenate((H_analy_left,H_analy1 ,H_analy_right),axis=0)
    C_analy1 = np.concatenate((C_analy_left,C_analy1 ,C_analy_right),axis=0)    
    eta = np.concatenate((left,eta,right),axis=0)
    
    ###Add 1+
    ax1.fill_betweenx(1+k*eta, phi_i(H_analy1,C_analy1)+phi_w(H_analy1,C_analy1)+phi_g(H_analy1,C_analy1), facecolor='white',label=r'$\phi_g$')
    ax1.fill_betweenx(1+k*eta, phi_w(H_analy1,C_analy1)+phi_i(H_analy1,C_analy1), facecolor=blue,label=r'$\phi_w$')
    ax1.fill_betweenx(1+k*eta, phi_i(H_analy1,C_analy1), facecolor=light_gray,label=r'$\phi_i$')

    k =  kk[1]
    ax2.fill_betweenx(1+k*eta, phi_i(H_analy1,C_analy1)+phi_w(H_analy1,C_analy1)+phi_g(H_analy1,C_analy1), facecolor='white',label=r'$\phi_g$')
    ax2.fill_betweenx(1+k*eta, phi_w(H_analy1,C_analy1)+phi_i(H_analy1,C_analy1), facecolor=blue,label=r'$\phi_w$')
    ax2.fill_betweenx(1+k*eta, phi_i(H_analy1,C_analy1), facecolor=light_gray,label=r'$\phi_i$')    
    
    k =  kk[2]
    
    ax3.fill_betweenx(1+k*eta, phi_i(H_analy1,C_analy1)+phi_w(H_analy1,C_analy1)+phi_g(H_analy1,C_analy1), facecolor='white',label=r'$\phi_g$')
    ax3.fill_betweenx(1+k*eta, phi_w(H_analy1,C_analy1)+phi_i(H_analy1,C_analy1), facecolor=blue,label=r'$\phi_w$')
    ax3.fill_betweenx(1+k*eta, phi_i(H_analy1,C_analy1), facecolor=light_gray,label=r'$\phi_i$')
    
    k =  kk[3]
    
    ax4.fill_betweenx(1+k*eta, phi_i(H_analy1,C_analy1)+phi_w(H_analy1,C_analy1)+phi_g(H_analy1,C_analy1), facecolor='white',label=r'$\phi_g$')
    ax4.fill_betweenx(1+k*eta, phi_w(H_analy1,C_analy1)+phi_i(H_analy1,C_analy1), facecolor=blue,label=r'$\phi_w$')
    ax4.fill_betweenx(1+k*eta, phi_i(H_analy1,C_analy1), facecolor=light_gray,label=r'$\phi_i$')
    
    ax1.legend(loc='lower left', shadow=False, fontsize='medium')
    plt.subplots_adjust(wspace=0, hspace=0.2)
    ax1.set_title(r'$\tau=$%.2f'%(3.56+kk[0]))
    ax2.set_title(r'%.2f'%(3.56+kk[1]))
    ax3.set_title(r'%.2f'%(3.56+kk[2]))
    ax4.set_title(r'%.2f'%(3.56+kk[3]))
    plt.subplots_adjust(wspace=0.15, hspace=0)
    plt.savefig(f"../Figures/{case_no}_combined_later.pdf")  


def plot_time_varying_paper(C_analy1,H_analy1,eta,Ste,Cpr,case_no):

    brown  = [181/255 , 101/255, 29/255]
    red    = [255/255 ,255/255 ,255/255 ]
    blue   = [ 30/255 ,144/255 , 255/255 ]
    green  = [  0/255 , 166/255 ,  81/255]
    orange = [247/255 , 148/255 ,  30/255]
    purple = [102/255 ,  45/255 , 145/255]
    brown  = [155/255 ,  118/255 ,  83/255]
    tan    = [199/255 , 178/255 , 153/255]
    gray   = [100/255 , 100/255 , 100/255]    

    # First set up the figure, the axis
    
    fig = plt.figure(figsize=(7.5,7.5) , dpi=100)
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    
    ax1.set_ylabel(r'Dimensionless depth')
    ax1.set_xlim([0,1])
    
    ax2.set_xlim([0,1])
    ax2.axes.yaxis.set_visible(False)
    
    ax3.set_xlim([0,1])
    ax3.axes.yaxis.set_visible(False)
    
    #manager = plt.get_current_fig_manager()
    #manager.window.showMaximized()
    
    kk = np.array([1e-3, 0.5, 1.0])  #time stamps
    
    ax1.set_ylim([np.max(eta)*kk[-1], np.min(eta)*kk[-1]])
    ax2.set_ylim([np.max(eta)*kk[-1], np.min(eta)*kk[-1]])
    ax3.set_ylim([np.max(eta)*kk[-1], np.min(eta)*kk[-1]]) 

    ax1.set_ylim([1, -0.05])
    ax2.set_ylim([1, -0.05])
    ax3.set_ylim([1, -0.05]) 
    
    
    k = kk[0] 
    
    print(kk[-1]*np.min(eta), k*np.min(eta), k*np.max(eta),kk[-1]*np.max(eta))
    left  = np.linspace(kk[-1]*np.min(eta)/(k+1e-10),np.min(eta),10000)
    right = np.linspace(np.max(eta),kk[-1]*np.max(eta)/(k+1e-10),10000)
    
    H_analy_left =np.ones_like(left)*H_analy1[0]
    C_analy_left=np.ones_like(left)*C_analy1[0]
    H_analy_right =np.ones_like(right)*H_analy1[-1]
    C_analy_right=np.ones_like(right)*C_analy1[-1] 
    
    H_analy1 = np.concatenate((H_analy_left,H_analy1 ,H_analy_right),axis=0)
    C_analy1 = np.concatenate((C_analy_left,C_analy1 ,C_analy_right),axis=0)    
    eta = np.concatenate((left,eta,right),axis=0)
    
    ###Add 1+
    ax1.fill_betweenx(k*eta, phi_i(H_analy1,C_analy1)+phi_w(H_analy1,C_analy1)+phi_g(H_analy1,C_analy1), facecolor='white',label=r'$\phi_g$')
    ax1.fill_betweenx(k*eta, phi_w(H_analy1,C_analy1)+phi_i(H_analy1,C_analy1), facecolor=blue,label=r'$\phi_w$')
    ax1.fill_betweenx(k*eta, phi_i(H_analy1,C_analy1), facecolor=light_gray,label=r'$\phi_i$')

    k =  kk[1]
    ax2.fill_betweenx(k*eta, phi_i(H_analy1,C_analy1)+phi_w(H_analy1,C_analy1)+phi_g(H_analy1,C_analy1), facecolor='white',label=r'$\phi_g$')
    ax2.fill_betweenx(k*eta, phi_w(H_analy1,C_analy1)+phi_i(H_analy1,C_analy1), facecolor=blue,label=r'$\phi_w$')
    ax2.fill_betweenx(k*eta, phi_i(H_analy1,C_analy1), facecolor=light_gray,label=r'$\phi_i$')    
    
    k =  kk[2]
    
    ax3.fill_betweenx(k*eta, phi_i(H_analy1,C_analy1)+phi_w(H_analy1,C_analy1)+phi_g(H_analy1,C_analy1), facecolor='white',label=r'$\phi_g$')
    ax3.fill_betweenx(k*eta, phi_w(H_analy1,C_analy1)+phi_i(H_analy1,C_analy1), facecolor=blue,label=r'$\phi_w$')
    ax3.fill_betweenx(k*eta, phi_i(H_analy1,C_analy1), facecolor=light_gray,label=r'$\phi_i$')
    
    ax1.legend(loc='lower left', shadow=False, fontsize='medium')
    plt.subplots_adjust(wspace=0, hspace=0.2)
    ax1.set_title(r'$\tau=$%.2f'%(kk[0]))
    ax2.set_title(r'%.2f'%(kk[1]))
    ax3.set_title(r'%.2f'%(kk[2]))
    plt.subplots_adjust(wspace=0.15, hspace=0)
    plt.savefig(f"../Figures/{case_no}_combined.pdf")  


def plot_time_varying_paper_variable_limits_variable_time(C_analy1,H_analy1,eta,Ste,Cpr,case_no,times=np.array([1e-3, 0.5, 1.0]),min_depth=-0.05,max_depth=1.0):

    brown  = [181/255 , 101/255, 29/255]
    red    = [255/255 ,255/255 ,255/255 ]
    blue   = [ 30/255 ,144/255 , 255/255 ]
    green  = [  0/255 , 166/255 ,  81/255]
    orange = [247/255 , 148/255 ,  30/255]
    purple = [102/255 ,  45/255 , 145/255]
    brown  = [155/255 ,  118/255 ,  83/255]
    tan    = [199/255 , 178/255 , 153/255]
    gray   = [100/255 , 100/255 , 100/255]    

    # First set up the figure, the axis
    
    fig = plt.figure(figsize=(7.5,7.5) , dpi=100)
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    
    ax1.set_ylabel(r'Dimensionless depth')
    ax1.set_xlim([0,1])
    
    ax2.set_xlim([0,1])
    ax2.axes.yaxis.set_visible(False)
    
    ax3.set_xlim([0,1])
    ax3.axes.yaxis.set_visible(False)
    
    #manager = plt.get_current_fig_manager()
    #manager.window.showMaximized()
    
    kk = times  #time stamps
    
    ax1.set_ylim([np.max(eta)*kk[-1], np.min(eta)*kk[-1]])
    ax2.set_ylim([np.max(eta)*kk[-1], np.min(eta)*kk[-1]])
    ax3.set_ylim([np.max(eta)*kk[-1], np.min(eta)*kk[-1]]) 

    ax1.set_ylim([max_depth,min_depth])
    ax2.set_ylim([max_depth,min_depth])
    ax3.set_ylim([max_depth,min_depth]) 
    
    
    k = kk[0] 
    
    print(kk[-1]*np.min(eta), k*np.min(eta), k*np.max(eta),kk[-1]*np.max(eta))
    left  = np.linspace(kk[-1]*np.min(eta)/(k+1e-10),np.min(eta),10000)
    right = np.linspace(np.max(eta),kk[-1]*np.max(eta)/(k+1e-10),10000)
    
    H_analy_left =np.ones_like(left)*H_analy1[0]
    C_analy_left=np.ones_like(left)*C_analy1[0]
    H_analy_right =np.ones_like(right)*H_analy1[-1]
    C_analy_right=np.ones_like(right)*C_analy1[-1] 
    
    H_analy1 = np.concatenate((H_analy_left,H_analy1 ,H_analy_right),axis=0)
    C_analy1 = np.concatenate((C_analy_left,C_analy1 ,C_analy_right),axis=0)    
    eta = np.concatenate((left,eta,right),axis=0)
    
    ###Add 1+
    ax1.fill_betweenx(k*eta, phi_i(H_analy1,C_analy1)+phi_w(H_analy1,C_analy1)+phi_g(H_analy1,C_analy1), facecolor='white',label=r'$\phi_g$')
    ax1.fill_betweenx(k*eta, phi_w(H_analy1,C_analy1)+phi_i(H_analy1,C_analy1), facecolor=blue,label=r'$\phi_w$')
    ax1.fill_betweenx(k*eta, phi_i(H_analy1,C_analy1), facecolor=light_gray,label=r'$\phi_i$')

    k =  kk[1]
    ax2.fill_betweenx(k*eta, phi_i(H_analy1,C_analy1)+phi_w(H_analy1,C_analy1)+phi_g(H_analy1,C_analy1), facecolor='white',label=r'$\phi_g$')
    ax2.fill_betweenx(k*eta, phi_w(H_analy1,C_analy1)+phi_i(H_analy1,C_analy1), facecolor=blue,label=r'$\phi_w$')
    ax2.fill_betweenx(k*eta, phi_i(H_analy1,C_analy1), facecolor=light_gray,label=r'$\phi_i$')    
    
    k =  kk[2]
    
    ax3.fill_betweenx(k*eta, phi_i(H_analy1,C_analy1)+phi_w(H_analy1,C_analy1)+phi_g(H_analy1,C_analy1), facecolor='white',label=r'$\phi_g$')
    ax3.fill_betweenx(k*eta, phi_w(H_analy1,C_analy1)+phi_i(H_analy1,C_analy1), facecolor=blue,label=r'$\phi_w$')
    ax3.fill_betweenx(k*eta, phi_i(H_analy1,C_analy1), facecolor=light_gray,label=r'$\phi_i$')
    
    #ax1.legend(loc='center left', shadow=False, fontsize='medium')
    plt.subplots_adjust(wspace=0, hspace=0.2)
    ax1.set_title(r'$\tau=$%.2f'%(kk[0]))
    ax2.set_title(r'%.2f'%(kk[1]))
    ax3.set_title(r'%.2f'%(kk[2]))
    plt.subplots_adjust(wspace=0.15, hspace=0)
    plt.savefig(f"../Figures/{case_no}_combined.pdf")  
