#Coding the non-linear double component solution
#Mohammad Afzal Shadab
#Date modified: 06/05/21

from aux import *
from matplotlib.markers import MarkerStyle

#enter case number

case_no = 1

#test details
simulation_name = f'solutions_two_component_'

#parameters
m = 3 #Cozeny-Karman coefficient for numerator K = K0 (1-phi_i)^m
n = 2 #Corey-Brooks coefficient krw = krw0 * sw^n
Cpw = 4186   #specific heat of water at constant pressure (J/kg.K)
Cpi = 2106.1 #specific heat of ice at constant pressure (J/kg.K)
Tm  = 273.16 #melting temperature (K)
L   = 333.55e3 #latent heat of fusion (J/kg)
Ste = Cpw*Tm/L #Stefan number
Cpr = Cpi/Cpw  #ratio of specific heat for ice to water



################################################
#AGU
################################################
#Saturated regions
C_analy1,H_analy1,C_case_plot1,H_case_plot1 = [[],[],[],[]]
C_analy2,H_analy2,C_case_plot2,H_case_plot2 = [[],[],[],[]] 
#Combined cases (R2 to R1)
#plotting(simulation_name,H_plotCH,C_plotCH,m,n) #plotting in hodograph plane
eta,C_analy1,H_analy1,C_case_plot1,H_case_plot1 = analytical(32,-2.1,1,0.7,0.4,0.7,-0.088395, m, n)

plotting_combined5AGU(simulation_name,C_case_plot1,H_case_plot1,C_case_plot2,H_case_plot2,m,n)

marker = MarkerStyle("d")
marker._transform.rotate_deg(0)

markera = MarkerStyle("v")

markera._transform.rotate_deg(0)

fig = plt.figure(figsize=(4,7), dpi=100)
if np.isnan(H_analy1[1]) == False:
    plt.subplot(111)
    #plt.plot((0.2,0.8), (0,0), 'k--',linewidth=3,label='Initial jump')
    plot = [plt.plot(C_analy1,eta,'r--',linewidth=3)]
    plt.ylim([0.3,-0.4])
    #plt.xlim([np.min(C_analy1)-0.1,np.max(C_analy1)+0.1])     
    plt.plot(C_case_plot1[0],-0.35,'kv', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot1[6666],-0.15,marker=marker,color="black", markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot1[13332],0.05,'ks', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot1[-1],0.2,'ko', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.xlabel(r'Dim-less Composition $\mathcal{C}$')    
    plt.ylabel(r'Dim-less Depth / Dim-less time')
    
    #plt.plot((0.2,0.6), (0,0), 'k--',linewidth=3,label='Initial jump')
    plot = [plt.plot(H_analy1,eta,'b--',linewidth=3)]
    plt.ylim([0.3,-0.4])
    plt.xlim([np.min([np.min(H_analy1),np.min(C_analy1)])-0.075,np.max([np.min(H_analy1),np.max(C_analy1)])+0.1])     
    plt.plot(H_case_plot1[0],-0.35,'kv', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot1[6666],-0.15,marker=marker,color="black", markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot1[13332],0.05,'ks', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot1[-1],0.2,'ko', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.xlabel(r'Dim-less Enthalpy $\mathcal{H}$')

                     
#plt.ylim([np.min([np.min(H_sol),np.min(C_sol)])-0.05,np.max([np.max(H_sol),np.max(C_sol)])+0.05])

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig(f"../Figures/{simulation_name}_tmax_combined_new5.pdf")

plot_phiw_CH(C_analy1,H_analy1,eta,Ste,Cpr,'5_saturated_region')
plot_time_varying_lower_layer(C_analy1,H_analy1,eta,Ste,Cpr,'5_saturated_region-front-new')


#Ice lens formation
C_analy1,H_analy1,C_case_plot1,H_case_plot1 = [[],[],[],[]]
C_analy2,H_analy2,C_case_plot2,H_case_plot2 = [[],[],[],[]] 
#Combined cases (R2 to R1)
#plotting(simulation_name,H_plotCH,C_plotCH,m,n) #plotting in hodograph plane
eta,C_analy1,H_analy1,C_case_plot1,H_case_plot1 = analytical(33,-7,1,0.7,0.4,0.95,-0.17995457952330984, m, n)

plotting_combined6AGU(simulation_name,C_case_plot1,H_case_plot1,C_case_plot2,H_case_plot2,m,n)

marker = MarkerStyle("d")
marker._transform.rotate_deg(0)

markera = MarkerStyle("v")

markera._transform.rotate_deg(0)

fig = plt.figure(figsize=(10,10), dpi=100)
if np.isnan(H_analy1[1]) == False:
    plt.subplot(121)
    #plt.plot((0.2,0.8), (0,0), 'k--',linewidth=3,label='Initial jump')
    plot = [plt.plot(C_analy1,eta,'b-',linewidth=3)]
    plt.ylim([1,-7])
    plt.xlim([np.min(C_analy1)-0.01,np.max(C_analy1)+0.01])    
    plt.plot(C_case_plot1[0],-6.5,marker=markera,color="blue", markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot1[6666],-3,marker=marker,color="blue", markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot1[13332],0,'bs', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot1[-1],0.5,'bo', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.xlabel(r'Dim-less Composition $\mathcal{C}$')    
    plt.ylabel(r'Dim-less Velocity $\eta = \zeta/\tau$')
    
    plt.subplot(122)
    #plt.plot((0.2,0.6), (0,0), 'k--',linewidth=3,label='Initial jump')
    plot = [plt.plot(H_analy1,eta,'b-',linewidth=3)]
    plt.ylim([1,-7])
    plt.xlim([np.min(H_analy1)-0.05,np.max(H_analy1)+0.05]) 
    plt.plot(H_case_plot1[0],-6.5,marker=markera,color="blue", markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot1[-1],0.5,'bo', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot1[6666],-3,marker=marker,color="blue", markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot1[13332],0,'bs', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.xlabel(r'Dim-less Enthalpy $\mathcal{H}$')
             
#plt.ylim([np.min([np.min(H_sol),np.min(C_sol)])-0.05,np.max([np.max(H_sol),np.max(C_sol)])+0.05])

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig(f"../Figures/{simulation_name}_tmax_combined_new6.pdf")

plot_phiw_CH(C_analy1,H_analy1,eta,Ste,Cpr,'6_icelens')

plot_time_varying_ice_lensAGU(C_analy1,H_analy1,eta,Ste,Cpr,'case6-icelens-front-new')




#combined three
#plotting(simulation_name,H_plotCH,C_plotCH,m,n) #plotting in hodograph plane
eta,C_analy1,H_analy1,C_case_plot1,H_case_plot1 = analytical(1,-0.1,1,0.7,0.4,0.89691358,0.45, m, n)
eta,C_analy2,H_analy2,C_case_plot2,H_case_plot2 = analytical(2,-0.1,1,0.7, 0.4,0.85,0.55, m, n)
eta,C_analy3,H_analy3,C_case_plot3,H_case_plot3 = analytical(3,-0.1,1,0.7, 0.4,0.5,0.2, m, n)
#eta,C_analy3,H_analy3,C_case_plot3,H_case_plot3 = analytical(3,-0.1,1,0.7, 0.4,0.3,0.0, m, n)  #shock

plotting_combined(simulation_name,C_case_plot1,H_case_plot1,C_case_plot2,H_case_plot2,C_case_plot3,H_case_plot3,m,n)

fig = plt.figure(figsize=(10,10), dpi=100)
if np.isnan(H_analy1[1]) == False:
    plt.subplot(121)
    #plt.plot((0.2,0.8), (0,0), 'k--',linewidth=3,label='Initial jump')
    plot = [plt.plot(C_analy1,eta,'g-',linewidth=3)]
    plot = [plt.plot(C_analy2,eta,'b-',linewidth=3)]
    plot = [plt.plot(C_analy3,eta,'r-',linewidth=3)]
    plt.ylim([1,-0.1])
    
    plt.plot(C_case_plot1[0],-0.05,'kv', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot1[-1],0.95,'go', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot2[-1],0.95,'b^', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot3[-1],0.95,'rs', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.xlabel(r'Dim-less Composition $\mathcal{C}$')
    plt.ylabel(r'Dim-less Velocity $\eta = \zeta/\tau$')

    plt.subplot(122)
    #plt.plot((0.2,0.6), (0,0), 'k--',linewidth=3,label='Initial jump')
    plot = [plt.plot(H_analy1,eta,'g-',linewidth=3)]
    plot = [plt.plot(H_analy2,eta,'b-',linewidth=3)]
    plot = [plt.plot(H_analy3,eta,'r-',linewidth=3)]
    plt.ylim([1,-0.1])
    
    plt.plot(H_case_plot1[0],-0.05,'kv', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot1[-1],0.95,'go', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot2[-1],0.95,'b^', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot3[-1],0.95,'rs', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.xlabel(r'Dim-less Enthalpy $\mathcal{H}$')
                     
#plt.ylim([np.min([np.min(H_sol),np.min(C_sol)])-0.05,np.max([np.max(H_sol),np.max(C_sol)])+0.05])

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig(f"../Figures/{simulation_name}_tmax_combined.pdf")

plotting_singularAGU(simulation_name,C_case_plot3,H_case_plot3,m,n)  #singular plot for shock

fig = plt.figure(figsize=(4,7), dpi=100)
if np.isnan(H_analy1[1]) == False:
    plt.subplot(111)
    #plt.plot((0.2,0.8), (0,0), 'k--',linewidth=3,label='Initial jump')
    plot = [plt.plot(C_analy3,eta,'r--',linewidth=3)]
    plt.ylim([1,-0.1])
    
    plt.plot(C_case_plot1[0],-0.05,'kv', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot3[-1],0.95,'ks', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.xlabel(r'Dim-less Composition $\mathcal{C}$')
    plt.ylabel(r'Dim-less Depth / Dim-less time')

    plot = [plt.plot(H_analy3,eta,'b--',linewidth=3)]
    plt.ylim([1,-0.1])
    
    plt.plot(H_case_plot1[0],-0.05,'kv', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot3[-1],0.95,'ks', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.xlabel(r'Dim-less Enthalpy & Comp.')
                     
#plt.ylim([np.min([np.min(H_sol),np.min(C_sol)])-0.05,np.max([np.max(H_sol),np.max(C_sol)])+0.05])

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig(f"../Figures/{simulation_name}_tmax_combined.pdf")
plot_phiw_CH(C_analy3,H_analy3,eta,Ste,Cpr,'case3-wetting-front-')

plot_time_varying(C_analy3,H_analy3,eta,Ste,Cpr,'case3-wetting-front-new')




#Combined cases
#plotting(simulation_name,H_plotCH,C_plotCH,m,n) #plotting in hodograph plane
eta,C_analy1,H_analy1,C_case_plot1,H_case_plot1 = analytical(4,-0.1,1,0.3,0.1,0.948,0.528, m, n)
eta,C_analy2,H_analy2,C_case_plot2,H_case_plot2 = analytical(5,-0.1,1,0.948,0.528,0.3,0.1, m, n)

plotting_combined2(simulation_name,C_case_plot1,H_case_plot1,C_case_plot2,H_case_plot2,m,n)

marker = MarkerStyle("d")
marker._transform.rotate_deg(0)

markera = MarkerStyle("v")

markera._transform.rotate_deg(0)

fig = plt.figure(figsize=(10,10), dpi=100)
if np.isnan(H_analy1[1]) == False:
    plt.subplot(121)
    #plt.plot((0.2,0.8), (0,0), 'k--',linewidth=3,label='Initial jump')
    plot = [plt.plot(C_analy1,eta,'b-',linewidth=3)]
    plot = [plt.plot(C_analy2,eta,'r-',linewidth=3)]
    plt.ylim([1,-0.1])
    
    plt.plot(C_case_plot1[0],-0.05,marker=markera,color="blue", markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot1[-1],0.95,'bo', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot2[0],-0.05,'rv', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot2[-1],0.95,'ro', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot1[int(10000-1)],eta[int(3000-1)],marker=marker,color="blue", markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot2[int(10000-1)],eta[int(5500-1)],'rs', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.xlabel(r'Dim-less Composition $\mathcal{C}$')    
    plt.ylabel(r'Dim-less Velocity $\eta = \zeta/\tau$')

    plt.subplot(122)
    #plt.plot((0.2,0.6), (0,0), 'k--',linewidth=3,label='Initial jump')
    plot = [plt.plot(H_analy1,eta,'b-',linewidth=3)]
    plot = [plt.plot(H_analy2,eta,'r-',linewidth=3)]
    plt.ylim([1,-0.1])
    
    plt.plot(H_case_plot1[0],-0.05,marker=markera,color="blue", markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot1[-1],0.95,'bo', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot2[0],-0.05,'rv', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot2[-1],0.95,'ro', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot1[int(10000-1)],eta[int(3000-1)],marker=marker,color="blue", markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot2[int(10000-1)],eta[int(5500-1)],'rs', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.xlabel(r'Dim-less Enthalpy $\mathcal{H}$')

                     
#plt.ylim([np.min([np.min(H_sol),np.min(C_sol)])-0.05,np.max([np.max(H_sol),np.max(C_sol)])+0.05])

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig(f"../Figures/{simulation_name}_tmax_combined_new.pdf")

plot_time_varying_diff_times(C_analy1,H_analy1,eta,Ste,Cpr,'Case4_C1R2',np.array([1e-7,0.6,1.2,2.4]))
plot_time_varying_diff_times(C_analy2,H_analy2,eta,Ste,Cpr,'Case5_C1S2',np.array([1e-7,0.6,1.2,2.4]))



#Combined cases (left state in Region 1)
#plotting(simulation_name,H_plotCH,C_plotCH,m,n) #plotting in hodograph plane
eta,C_analy1,H_analy1,C_case_plot1,H_case_plot1 = analytical(6,-0.1,1,0.6,-0.01,0.8,-0.1, m, n)
eta,C_analy2,H_analy2,C_case_plot2,H_case_plot2 = analytical(12,-0.1,1,0.7,-0.1,0.8,0.6, m, n)

plotting_combined3(simulation_name,C_case_plot1,H_case_plot1,C_case_plot2,H_case_plot2,m,n)


marker = MarkerStyle("d")
marker._transform.rotate_deg(0)

markera = MarkerStyle("v")

markera._transform.rotate_deg(0)

fig = plt.figure(figsize=(10,10), dpi=100)
if np.isnan(H_analy1[1]) == False:
    plt.subplot(121)
    #plt.plot((0.2,0.8), (0,0), 'k--',linewidth=3,label='Initial jump')
    plot = [plt.plot(C_analy1,eta,'b-',linewidth=3)]
    plot = [plt.plot(C_analy2,eta,'r-',linewidth=3)]
    plt.ylim([1,-0.1])
    
    plt.plot(C_case_plot1[0],-0.05,marker=markera,color="blue", markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot1[-1],0.95,'bo', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot2[0],-0.05,'rv', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot2[-1],0.95,'ro', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot2[int(10000-1)],0,'rs', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.xlabel(r'Dim-less Composition $\mathcal{C}$')
    plt.ylabel(r'Dim-less Velocity $\eta = \zeta/\tau$')
    
    plt.subplot(122)
    #plt.plot((0.2,0.6), (0,0), 'k--',linewidth=3,label='Initial jump')
    plot = [plt.plot(H_analy1,eta,'b-',linewidth=3)]
    plot = [plt.plot(H_analy2,eta,'r-',linewidth=3)]
    plt.ylim([1,-0.1])
    
    plt.plot(H_case_plot1[0],-0.05,marker=markera,color="blue", markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot1[-1],0.95,'bo', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot2[0],-0.05,'rv', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot2[-1],0.95,'ro', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot2[int(10000-1)],0,'rs', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.xlabel(r'Dim-less Enthalpy $\mathcal{H}$')    

                     
#plt.ylim([np.min([np.min(H_sol),np.min(C_sol)])-0.05,np.max([np.max(H_sol),np.max(C_sol)])+0.05])

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig(f"../Figures/{simulation_name}_tmax_combined_new3.pdf")

C_analy1,H_analy1,C_case_plot1,H_case_plot1 = [[],[],[],[]]
C_analy2,H_analy2,C_case_plot2,H_case_plot2 = [[],[],[],[]] 
#Combined cases (R2 to R1)
#plotting(simulation_name,H_plotCH,C_plotCH,m,n) #plotting in hodograph plane
eta,C_analy1,H_analy1,C_case_plot1,H_case_plot1 = analytical(13,-0.1,1,0.8,0.25,0.5,-0.05, m, n)
eta,C_analy2,H_analy2,C_case_plot2,H_case_plot2 = analytical(14,-0.1,1,0.0894,0.0894,0.7,-0.1, m, n)

plotting_combined4(simulation_name,C_case_plot1,H_case_plot1,C_case_plot2,H_case_plot2,m,n)

marker = MarkerStyle("d")
marker._transform.rotate_deg(0)

markera = MarkerStyle("v")

markera._transform.rotate_deg(0)

fig = plt.figure(figsize=(10,10), dpi=100)
if np.isnan(H_analy1[1]) == False:
    plt.subplot(121)
    #plt.plot((0.2,0.8), (0,0), 'k--',linewidth=3,label='Initial jump')
    plot = [plt.plot(C_analy1,eta,'b-',linewidth=3)]
    plot = [plt.plot(C_analy2,eta,'r-',linewidth=3)]
    plt.ylim([1,-0.1])
    
    plt.plot(C_case_plot1[0],-0.05,marker=markera,color="blue", markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot1[-1],0.95,'bo', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot2[0],-0.05,'rv', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot2[-1],0.95,'ro', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot2[int(10000-1)],0,'rs', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.xlabel(r'Dim-less Composition $\mathcal{C}$')    
    plt.ylabel(r'Dim-less Velocity $\eta = \zeta/\tau$')
    
    plt.subplot(122)
    #plt.plot((0.2,0.6), (0,0), 'k--',linewidth=3,label='Initial jump')
    plot = [plt.plot(H_analy1,eta,'b-',linewidth=3)]
    plot = [plt.plot(H_analy2,eta,'r-',linewidth=3)]
    plt.ylim([1,-0.1])
    
    plt.plot(H_case_plot1[0],-0.05,marker=markera,color="blue", markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot1[-1],0.95,'bo', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot2[0],-0.05,'rv', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot2[-1],0.95,'ro', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot2[int(10000-1)],0,'rs', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.xlabel(r'Dim-less Enthalpy $\mathcal{H}$')

                     
#plt.ylim([np.min([np.min(H_sol),np.min(C_sol)])-0.05,np.max([np.max(H_sol),np.max(C_sol)])+0.05])

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig(f"../Figures/{simulation_name}_tmax_combined_new4.pdf")



#Saturated regions
C_analy1,H_analy1,C_case_plot1,H_case_plot1 = [[],[],[],[]]
C_analy2,H_analy2,C_case_plot2,H_case_plot2 = [[],[],[],[]] 
#Combined cases (R2 to R1)
#plotting(simulation_name,H_plotCH,C_plotCH,m,n) #plotting in hodograph plane
eta,C_analy1,H_analy1,C_case_plot1,H_case_plot1 = analytical(32,-2.1,0.6,0.95,0.65,0.5,-0.09471293659121571, m, n)

plotting_combined5(simulation_name,C_case_plot1,H_case_plot1,C_case_plot2,H_case_plot2,m,n)

marker = MarkerStyle("d")
marker._transform.rotate_deg(0)

markera = MarkerStyle("v")

markera._transform.rotate_deg(0)

fig = plt.figure(figsize=(10,10), dpi=100)
if np.isnan(H_analy1[1]) == False:
    plt.subplot(121)
    #plt.plot((0.2,0.8), (0,0), 'k--',linewidth=3,label='Initial jump')
    plot = [plt.plot(C_analy1,eta,'b-',linewidth=3)]
    plt.ylim([0.6,-2.1])
    plt.xlim([np.min(C_analy1)-0.1,np.max(C_analy1)+0.1])     
    plt.plot(C_case_plot1[0],-2,marker=markera,color="blue", markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot1[6666],-1,marker=marker,color="blue", markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot1[13332],0,'bs', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot1[-1],0.5,'bo', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.xlabel(r'Dim-less Composition $\mathcal{C}$')    
    plt.ylabel(r'Dim-less Velocity $\eta = \zeta/\tau$')
    
    plt.subplot(122)
    #plt.plot((0.2,0.6), (0,0), 'k--',linewidth=3,label='Initial jump')
    plot = [plt.plot(H_analy1,eta,'b-',linewidth=3)]
    plt.ylim([0.6,-2.1])
    plt.xlim([np.min(H_analy1)-0.075,np.max(H_analy1)+0.1])     
    plt.plot(H_case_plot1[0],-2,marker=markera,color="blue", markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot1[6666],-1,marker=marker,color="blue", markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot1[13332],0,'bs', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot1[-1],0.5,'bo', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.xlabel(r'Dim-less Enthalpy $\mathcal{H}$')

                     
#plt.ylim([np.min([np.min(H_sol),np.min(C_sol)])-0.05,np.max([np.max(H_sol),np.max(C_sol)])+0.05])

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig(f"../Figures/{simulation_name}_tmax_combined_new5.pdf")

plot_phiw_CH(C_analy1,H_analy1,eta,Ste,Cpr,'5_saturated_region')

#Ice lens formation
C_analy1,H_analy1,C_case_plot1,H_case_plot1 = [[],[],[],[]]
C_analy2,H_analy2,C_case_plot2,H_case_plot2 = [[],[],[],[]] 
#Combined cases (R2 to R1)
#plotting(simulation_name,H_plotCH,C_plotCH,m,n) #plotting in hodograph plane
eta,C_analy1,H_analy1,C_case_plot1,H_case_plot1 = analytical(33,-7,1,0.95,0.65,0.95,-0.17995457952330984, m, n)

plotting_combined6(simulation_name,C_case_plot1,H_case_plot1,C_case_plot2,H_case_plot2,m,n)

marker = MarkerStyle("d")
marker._transform.rotate_deg(0)

markera = MarkerStyle("v")

markera._transform.rotate_deg(0)

fig = plt.figure(figsize=(10,10), dpi=100)
if np.isnan(H_analy1[1]) == False:
    plt.subplot(121)
    #plt.plot((0.2,0.8), (0,0), 'k--',linewidth=3,label='Initial jump')
    plot = [plt.plot(C_analy1,eta,'b-',linewidth=3)]
    plt.ylim([1,-7])
    plt.xlim([np.min(C_analy1)-0.01,np.max(C_analy1)+0.01])    
    plt.plot(C_case_plot1[0],-6.5,marker=markera,color="blue", markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot1[6666],-3,marker=marker,color="blue", markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot1[13332],0,'bs', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot1[-1],0.5,'bo', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.xlabel(r'Dim-less Composition $\mathcal{C}$')    
    plt.ylabel(r'Dim-less Velocity $\eta = \zeta/\tau$')
    
    plt.subplot(122)
    #plt.plot((0.2,0.6), (0,0), 'k--',linewidth=3,label='Initial jump')
    plot = [plt.plot(H_analy1,eta,'b-',linewidth=3)]
    plt.ylim([1,-7])
    plt.xlim([np.min(H_analy1)-0.05,np.max(H_analy1)+0.05]) 
    plt.plot(H_case_plot1[0],-6.5,marker=markera,color="blue", markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot1[-1],0.5,'bo', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot1[6666],-3,marker=marker,color="blue", markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot1[13332],0,'bs', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.xlabel(r'Dim-less Enthalpy $\mathcal{H}$')
             
#plt.ylim([np.min([np.min(H_sol),np.min(C_sol)])-0.05,np.max([np.max(H_sol),np.max(C_sol)])+0.05])

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig(f"../Figures/{simulation_name}_tmax_combined_new6.pdf")

plot_phiw_CH(C_analy1,H_analy1,eta,Ste,Cpr,'6_icelens')

#Temperate saturated formation
C_analy1,H_analy1,C_case_plot1,H_case_plot1 = [[],[],[],[]]
C_analy2,H_analy2,C_case_plot2,H_case_plot2 = [[],[],[],[]] 
#Combined cases (R2 to R1)
#plotting(simulation_name,H_plotCH,C_plotCH,m,n) #plotting in hodograph plane
eta,C_analy1,H_analy1,C_case_plot1,H_case_plot1 = analytical(31,-0.5,0.5,0.9,0.4,0.8, 0.1, m, n)

plotting_combined7(simulation_name,C_case_plot1,H_case_plot1,C_case_plot2,H_case_plot2,m,n)

marker = MarkerStyle("d")
marker._transform.rotate_deg(0)

markera = MarkerStyle("v")

markera._transform.rotate_deg(0)

fig = plt.figure(figsize=(10,10), dpi=100)
if np.isnan(H_analy1[1]) == False:
    plt.subplot(121)
    #plt.plot((0.2,0.8), (0,0), 'k--',linewidth=3,label='Initial jump')
    plot = [plt.plot(C_analy1,eta,'b-',linewidth=3)]
    plt.ylim([0.5,-0.5])
    plt.xlim([np.min(C_analy1)-0.01,np.max(C_analy1)+0.02])    
    plt.plot(C_case_plot1[0],-0.4,marker=markera,color="blue", markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot1[6666],-0.15,marker=marker,color="blue", markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot1[13332],0.1,'bs', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot1[-1],0.35,'bo', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.xlabel(r'Dim-less Composition $\mathcal{C}$')    
    plt.ylabel(r'Dim-less Velocity $\eta = \zeta/\tau$')

    plt.subplot(122)
    #plt.plot((0.2,0.6), (0,0), 'k--',linewidth=3,label='Initial jump')
    plot = [plt.plot(H_analy1,eta,'b-',linewidth=3)]
    plt.ylim([0.5,-0.5])
    plt.xlim([np.min(H_analy1)-0.05,np.max(H_analy1)+0.05]) 
    plt.plot(H_case_plot1[0],-0.4,marker=markera,color="blue", markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot1[-1],0.35,'bo', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot1[6666],-0.15,marker=marker,color="blue", markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot1[13332],0.1,'bs', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.xlabel(r'Dim-less Enthalpy $\mathcal{H}$')
             
#plt.ylim([np.min([np.min(H_sol),np.min(C_sol)])-0.05,np.max([np.max(H_sol),np.max(C_sol)])+0.05])

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig(f"../Figures/{simulation_name}_tmax_combined_new7.pdf")

plot_phiw_CH(C_analy1,H_analy1,eta,Ste,Cpr,'7')
