#Unified kinematic wave theory
#Manuscript DOI: https://arxiv.org/pdf/2403.15996
#Mohammad Afzal Shadab, Anja Rutishauser, Cyril Grima and Marc Hesse
#University of Texas at Austin
#Contact email: mashadab@utexas.edu / mashadab@princeton.edu
#Date modified: 10/14/24

#Function Call: analytical(case_no,etaL,etaR,C_L,H_L,C_R,H_R, m, n)
#Analytical function arguments: case_number in code,
#etaL: left velocity,etaR: right velocity
#C_L: left composition,H_L: left enthalpy
#C_R: right composition,H_R: right enthalpy, 
#m: Porosity exponent in absolute permeability definition, n: saturation exponent in relative permeability definition

#import libraries and auxillaries: aux
from aux import *

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

################################################################################################
#Combined cases I, II and III
#Case I: Contact discontinuity only
eta,C_analy1,H_analy1,C_case_plot1,H_case_plot1 = analytical(1,-0.1,1,0.7,0.4,0.89691358,0.45, m, n) #calculating analytic solution
plot_time_varying_paper(C_analy1,H_analy1,eta,Ste,Cpr,'1_contact') #plotting time variation of volume fraction for case I

#Case II: Rarefaction only
eta,C_analy2,H_analy2,C_case_plot2,H_case_plot2 = analytical(2,-0.1,1,0.7, 0.4,0.85,0.55, m, n) #calculating analytic solution
plot_time_varying_paper(C_analy2,H_analy2,eta,Ste,Cpr,'2_rarefaction') #plotting time variation of volume fraction for case II

#Case III: Shock only
eta,C_analy3,H_analy3,C_case_plot3,H_case_plot3 = analytical(3,-0.1,1,0.7, 0.4,0.55,0.25, m, n)  #calculating analytic solution
plot_time_varying_paper(C_analy3,H_analy3,eta,Ste,Cpr,'3_shock') #plotting time variation of volume fraction for case III

plotting_combined(simulation_name,C_case_plot1,H_case_plot1,C_case_plot2,H_case_plot2,C_case_plot3,H_case_plot3,m,n) #combined plotting states for cases I-III in hodograph plane

#Plotting the C and H in the velocity domain for case I to III
fig = plt.figure(figsize=(10,10), dpi=100)
if np.isnan(H_analy1[1]) == False:
    plt.subplot(121)
    plot = [plt.plot(C_analy1,eta,'g-',linewidth=3)]
    plot = [plt.plot(C_analy2,eta,'b-',linewidth=3)]
    plot = [plt.plot(C_analy3,eta,'r-',linewidth=3)]
    plt.ylim([1,-0.1])
    
    plt.plot(C_case_plot1[0],-0.05,'kx', markersize=20,markerfacecolor='white',markeredgewidth=3)
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
    
    plt.plot(H_case_plot1[0],-0.05,'kx', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot1[-1],0.95,'go', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot2[-1],0.95,'b^', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot3[-1],0.95,'rs', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.xlabel(r'Dim-less Enthalpy $\mathcal{H}$')

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig(f"../Figures/{simulation_name}_tmax_combined.pdf")

################################################################################################
#Combined cases IV and V
#Case IV: Contact discontinuity 1 and Rarefaction 2
eta,C_analy1,H_analy1,C_case_plot1,H_case_plot1 = analytical(4,-0.1,1,0.3,0.1,0.948,0.528, m, n) #calculating analytic solution
plot_time_varying_paper_variable_limits_variable_time(C_analy1,H_analy1,eta,Ste,Cpr,'4-C1R2-rarefaction',np.array([1e-4,0.75,1.5]),-0.05,1) #plotting time variation of volume fraction for case IV

#Case V: Contact discontinuity 1 and Shock 2
eta,C_analy2,H_analy2,C_case_plot2,H_case_plot2 = analytical(5,-0.1,1,0.948,0.528,0.3,0.1, m, n)#calculating analytic solution
plot_time_varying_paper_variable_limits_variable_time(C_analy2,H_analy2,eta,Ste,Cpr,'5-C1S2-shock',np.array([1e-4,0.75,1.5]),-0.05,1)  #plotting time variation of volume fraction for case V

plotting_combined2(simulation_name,C_case_plot1,H_case_plot1,C_case_plot2,H_case_plot2,m,n) #combined plotting states for cases IV-V in hodograph plane

marker = MarkerStyle("d")
marker._transform.rotate_deg(90)

#Plotting the C and H in the velocity domain for case IV to V
fig = plt.figure(figsize=(10,10), dpi=100)
if np.isnan(H_analy1[1]) == False:
    plt.subplot(121)
    #plt.plot((0.2,0.8), (0,0), 'k--',linewidth=3,label='Initial jump')
    plot = [plt.plot(C_analy1,eta,'b-',linewidth=3)]
    plot = [plt.plot(C_analy2,eta,'r-',linewidth=3)]
    plt.ylim([1,-0.1])
    
    plt.plot(C_case_plot1[0],-0.05,'bx', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot1[-1],0.95,'bo', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot2[0],-0.05,'rx', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot2[-1],0.95,'ro', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot1[int(10000-1)],eta[int(3000-1)],marker=marker,color="blue", markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot2[int(10000-1)],eta[int(5500-1)],'rs', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.xlabel(r'Dim-less Composition $\mathcal{C}$')    
    plt.ylabel(r'Dim-less Velocity $\eta = \zeta/\tau$')

    plt.subplot(122)
    plot = [plt.plot(H_analy1,eta,'b-',linewidth=3)]
    plot = [plt.plot(H_analy2,eta,'r-',linewidth=3)]
    plt.ylim([1,-0.1])
    
    plt.plot(H_case_plot1[0],-0.05,'bx', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot1[-1],0.95,'bo', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot2[0],-0.05,'rx', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot2[-1],0.95,'ro', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot1[int(10000-1)],eta[int(3000-1)],marker=marker,color="blue", markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot2[int(10000-1)],eta[int(5500-1)],'rs', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.xlabel(r'Dim-less Enthalpy $\mathcal{H}$')

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig(f"../Figures/{simulation_name}_tmax_combined_new.pdf")

################################################################################################

#Case VI: Backfilling shock 1, Jump 2, and Shock 3
#Temperate saturated formation
C_analy1,H_analy1,C_case_plot1,H_case_plot1 = [[],[],[],[]]
C_analy2,H_analy2,C_case_plot2,H_case_plot2 = [[],[],[],[]] 

eta,C_analy1,H_analy1,C_case_plot1,H_case_plot1 = analytical(31,-0.5,0.5,0.9,0.4,0.8, 0.1, m, n) #calculating analytic solution

plotting_combined7(simulation_name,C_case_plot1,H_case_plot1,C_case_plot2,H_case_plot2,m,n) #combined plotting states for case VI in hodograph plane
plot_time_varying_paper_variable_limits_variable_time(C_analy1,H_analy1,eta,Ste,Cpr,'31-temperate-saturated-region',np.array([1e-4,0.5,1.0]),-0.4,0.3)   #plotting time variation of volume fraction for case VI

marker = MarkerStyle("d")
marker._transform.rotate_deg(90)

#Plotting the C and H in the velocity domain for case VI
fig = plt.figure(figsize=(10,10), dpi=100)
if np.isnan(H_analy1[1]) == False:
    plt.subplot(121)
    #plt.plot((0.2,0.8), (0,0), 'k--',linewidth=3,label='Initial jump')
    plot = [plt.plot(C_analy1,eta,'b-',linewidth=3)]
    plt.ylim([0.5,-0.5])
    plt.xlim([np.min(C_analy1)-0.01,np.max(C_analy1)+0.02])    
    plt.plot(C_case_plot1[0],-0.4,'bx', markersize=20,markerfacecolor='white',markeredgewidth=3)
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
    plt.plot(H_case_plot1[0],-0.4,'bx', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot1[-1],0.35,'bo', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot1[6666],-0.15,marker=marker,color="blue", markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot1[13332],0.1,'bs', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.xlabel(r'Dim-less Enthalpy $\mathcal{H}$')

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig(f"../Figures/{simulation_name}_tmax_combined_new7.pdf")

################################################################################################
#Combined cases VII and VIII

#Case VII: Contact discontinuity 
eta,C_analy1,H_analy1,C_case_plot1,H_case_plot1 = analytical(6,-0.1,1,0.6,-0.01,0.8,-0.1, m, n)  #calculating analytic solution
plot_time_varying_paper(C_analy1,H_analy1,eta,Ste,Cpr,'6-C-contact') #plotting time variation of volume fraction for case VII

#Case VIII: Contact discontinuity 1 and Rarefaction 2
eta,C_analy2,H_analy2,C_case_plot2,H_case_plot2 = analytical(12,-0.1,1,0.7,-0.1,0.8,0.6, m, n)  #calculating analytic solution
plot_time_varying_paper(C_analy2,H_analy2,eta,Ste,Cpr,'12-contact-shock') #plotting time variation of volume fraction for case VIII

plotting_combined3(simulation_name,C_case_plot1,H_case_plot1,C_case_plot2,H_case_plot2,m,n) #combined plotting states for cases VII-VIII in hodograph plane

marker = MarkerStyle("d")
marker._transform.rotate_deg(90)

#Plotting the C and H in the velocity domain for case VII and VIII
fig = plt.figure(figsize=(10,10), dpi=100)
if np.isnan(H_analy1[1]) == False:
    plt.subplot(121)
    #plt.plot((0.2,0.8), (0,0), 'k--',linewidth=3,label='Initial jump')
    plot = [plt.plot(C_analy1,eta,'b-',linewidth=3)]
    plot = [plt.plot(C_analy2,eta,'r-',linewidth=3)]
    plt.ylim([1,-0.1])
    
    plt.plot(C_case_plot1[0],-0.05,'bx', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot1[-1],0.95,'bo', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot2[0],-0.05,'rx', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot2[-1],0.95,'ro', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot2[int(10000-1)],0,'rs', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.xlabel(r'Dim-less Composition $\mathcal{C}$')
    plt.ylabel(r'Dim-less Velocity $\eta = \zeta/\tau$')
    
    plt.subplot(122)
    #plt.plot((0.2,0.6), (0,0), 'k--',linewidth=3,label='Initial jump')
    plot = [plt.plot(H_analy1,eta,'b-',linewidth=3)]
    plot = [plt.plot(H_analy2,eta,'r-',linewidth=3)]
    plt.ylim([1,-0.1])
    
    plt.plot(H_case_plot1[0],-0.05,'bx', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot1[-1],0.95,'bo', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot2[0],-0.05,'rx', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot2[-1],0.95,'ro', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot2[int(10000-1)],0,'rs', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.xlabel(r'Dim-less Enthalpy $\mathcal{H}$')    


plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig(f"../Figures/{simulation_name}_tmax_combined_new3.pdf")


################################################################################################
#Combined cases IX and X

#Case IX: Shock
#Combined cases (R2 to R1)
C_analy1,H_analy1,C_case_plot1,H_case_plot1 = [[],[],[],[]]
C_analy2,H_analy2,C_case_plot2,H_case_plot2 = [[],[],[],[]] 
eta,C_analy1,H_analy1,C_case_plot1,H_case_plot1 = analytical(13,-0.1,1,0.8,0.25,0.5,-0.05, m, n)  #calculating analytic solution
plot_time_varying_paper_variable_limits_variable_time(C_analy1,H_analy1,eta,Ste,Cpr,'13-C-shock',np.array([1e-4,0.5,1.0]),-0.05,0.15)  #plotting time variation of volume fraction for case IX

#Case X: Contact Discontinuity 1 and Shock 2
eta,C_analy2,H_analy2,C_case_plot2,H_case_plot2 = analytical(14,-0.1,1,0.4,0.1,0.65,-0.08, m, n)  #calculating analytic solution
plot_time_varying_paper_variable_limits_variable_time(C_analy2,H_analy2,eta,Ste,Cpr,'14-contact-shock',np.array([1e-4,0.5,1.0]),-0.005,0.05)  #plotting time variation of volume fraction for case X

plotting_combined4(simulation_name,C_case_plot1,H_case_plot1,C_case_plot2,H_case_plot2,m,n)  #combined plotting states for cases IX and X in hodograph plane


marker = MarkerStyle("d")
marker._transform.rotate_deg(90)

#Plotting the C and H in the velocity domain for cases IX and X
fig = plt.figure(figsize=(10,10), dpi=100)
if np.isnan(H_analy1[1]) == False:
    plt.subplot(121)
    plot = [plt.plot(C_analy1,eta,'b-',linewidth=3)]
    plot = [plt.plot(C_analy2,eta,'r-',linewidth=3)]
    plt.ylim([1,-0.1])
    
    plt.plot(C_case_plot1[0],-0.05,'bv', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot1[-1],0.95,'bo', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot2[0],-0.05,'rv', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot2[-1],0.95,'ro', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot2[int(10000-1)],0.01,'rs', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.xlabel(r'Dim-less Composition $\mathcal{C}$')    
    plt.ylabel(r'Dim-less Velocity $\eta = \zeta/\tau$')
    
    plt.subplot(122)
    plot = [plt.plot(H_analy1,eta,'b-',linewidth=3)]
    plot = [plt.plot(H_analy2,eta,'r-',linewidth=3)]
    plt.ylim([1,-0.1])
    
    plt.plot(H_case_plot1[0],-0.05,'bv', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot1[-1],0.95,'bo', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot2[0],-0.05,'rv', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot2[-1],0.95,'ro', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot2[int(10000-1)],0.01,'rs', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.xlabel(r'Dim-less Enthalpy $\mathcal{H}$')

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig(f"../Figures/{simulation_name}_tmax_combined_new4.pdf")

################################################################################################
#Case XI: Backfilling shock 1, Jump 2, and Shock 3 (in cold firn)
#Saturated region in cold medium

C_analy1,H_analy1,C_case_plot1,H_case_plot1 = [[],[],[],[]]
C_analy2,H_analy2,C_case_plot2,H_case_plot2 = [[],[],[],[]] 
eta,C_analy1,H_analy1,C_case_plot1,H_case_plot1 = analytical(32,-2.1,0.7,0.85,0.65,0.5,-0.09471293659121571, m, n) #calculating analytic solution
plot_time_varying_paper_variable_limits_variable_time(C_analy1,H_analy1,eta,Ste,Cpr,'32-Saturated-S1J2S3',np.array([1e-4,0.1,0.2]),-0.4,0.1) #plotting time variation of volume fraction for case XI
plotting_combined5(simulation_name,C_case_plot1,H_case_plot1,C_case_plot2,H_case_plot2,m,n) #combined plotting states for case XI in hodograph plane

marker = MarkerStyle("d")
marker._transform.rotate_deg(90)

#Plotting the C and H in the velocity domain for case XI
fig = plt.figure(figsize=(10,10), dpi=100)
if np.isnan(H_analy1[1]) == False:
    plt.subplot(121)
    #plt.plot((0.2,0.8), (0,0), 'k--',linewidth=3,label='Initial jump')
    plot = [plt.plot(C_analy1,eta,'b-',linewidth=3)]
    plt.ylim([0.6,-1.6])
    plt.xlim([np.min(C_analy1)-0.1,np.max(C_analy1)+0.1])     
    plt.plot(C_case_plot1[0],-1.5,'bx', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot1[6666],-0.6,marker=marker,color="blue", markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot1[13332],0.2,'bs', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(C_case_plot1[-1],0.5,'bo', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.xlabel(r'Dim-less Composition $\mathcal{C}$')    
    plt.ylabel(r'Dim-less Velocity $\eta = \zeta/\tau$')
    
    plt.subplot(122)
    #plt.plot((0.2,0.6), (0,0), 'k--',linewidth=3,label='Initial jump')
    plot = [plt.plot(H_analy1,eta,'b-',linewidth=3)]
    plt.ylim([0.6,-1.6])
    plt.xlim([np.min(H_analy1)-0.075,np.max(H_analy1)+0.1])     
    plt.plot(H_case_plot1[0],-1.5,'bx', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot1[6666],-0.6,marker=marker,color="blue", markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot1[13332],0.2,'bs', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot1[-1],0.5,'bo', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.xlabel(r'Dim-less Enthalpy $\mathcal{H}$')

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig(f"../Figures/{simulation_name}_tmax_combined_new5.pdf")

################################################################################################
#Case XII: Backfilling shock 1, Jump 2, and Contact discontinuity 3 (in cold firn)
#Impermeable ice layer formation

#Ice lens formation
C_analy1,H_analy1,C_case_plot1,H_case_plot1 = [[],[],[],[]]
C_analy2,H_analy2,C_case_plot2,H_case_plot2 = [[],[],[],[]] 
eta,C_analy1,H_analy1,C_case_plot1,H_case_plot1 = analytical(33,-7,1,0.95,0.65,0.95,-0.17995457952330984, m, n) #calculating analytic solution
plot_time_varying_paper_variable_limits_variable_time(C_analy1,H_analy1,eta,Ste,Cpr,'33-impermeable-ice-lens',np.array([1e-5,0.02,0.04]),-0.25,0.05) #plotting time variation of volume fraction for case XII

plotting_combined6(simulation_name,C_case_plot1,H_case_plot1,C_case_plot2,H_case_plot2,m,n) #combined plotting states for case XII in hodograph plane

marker = MarkerStyle("d")
marker._transform.rotate_deg(90)

#Plotting the C and H in the velocity domain for case XII
fig = plt.figure(figsize=(10,10), dpi=100)
if np.isnan(H_analy1[1]) == False:
    plt.subplot(121)
    #plt.plot((0.2,0.8), (0,0), 'k--',linewidth=3,label='Initial jump')
    plot = [plt.plot(C_analy1,eta,'b-',linewidth=3)]
    plt.ylim([1,-7])
    plt.xlim([np.min(C_analy1)-0.01,np.max(C_analy1)+0.01])    
    plt.plot(C_case_plot1[0],-6.5,'bx', markersize=20,markerfacecolor='white',markeredgewidth=3)
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
    plt.plot(H_case_plot1[0],-6.5,'bx', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot1[-1],0.5,'bo', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot1[6666],-3,marker=marker,color="blue", markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.plot(H_case_plot1[13332],0,'bs', markersize=20,markerfacecolor='white',markeredgewidth=3)
    plt.xlabel(r'Dim-less Enthalpy $\mathcal{H}$')
             
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig(f"../Figures/{simulation_name}_tmax_combined_new6.pdf")
