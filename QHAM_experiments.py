import qiskit
from qiskit import *
%matplotlib inline
from qiskit.tools.visualization import plot_histogram
from qiskit.providers.aer.noise import NoiseModel
import qiskit.providers.aer.noise as noise
#import qiskit.providers.models as mods
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator, FormatStrFormatter
from qiskit.tools.monitor import job_monitor
from qiskit.circuit import Parameter
import seaborn as sns
import pandas as pd
from pprint import pprint

#import matplotlib as mpl
matplotlib.rc('font',family='arial')

qiskit.__version__
IBMQ.load_account()
backend = Aer.get_backend('qasm_simulator')
#backend = Aer.get_backend('statevector_simulator')
provider = IBMQ.get_provider(hub='ibm-q')
#device = provider.get_backend('ibmq_16_melbourne') #ibmq_16_melbourne ibmqx2
#noise_model = NoiseModel.from_backend(device)

def hop_weight_setting_2_attractors(u1,u2,length):
    wm = np.zeros((length, length))
    for i in range(length):
        for j in range(length):
            wm[i][j] = (1/2)*(u1[i]*u1[j]+u2[i]*u2[j])    
    return wm

def add_RUS1_JustRy_with_bias(qc,phis,control_registers,bias,qubit):
    num_ctrls = len(control_registers)
    for i in range(num_ctrls):
        qc.cry((phis[i]),control_registers[i],qubit) #control([num_ctrl_qubits, label, ctrl_state])
    qc.ry(bias,qubit)
    qc.barrier()

# create hopfield network in a single function using given attractors and initial state
def create_hopfield_simplified(initstate,attractor1,attractor2,shape_m,shape_n,RUSk,updates,num_tries,be,useNoise,nm,ploteach,fsize,cmp,nshots):
    n = len(attractor1)
    if(n != len(attractor2) or n != len(initstate)):
        print("Error: Attractors and input states must be the same length.")
        return
    wm = hop_weight_setting_2_attractors(attractor1,attractor2,n)
    thresholds = [0]*n
    qubits_to_update = updates
    if qubits_to_update == []:
        for i in range(n):
            if(initstate[i] == 0): # not(isinstance(initstate[i],int)) this can be changed depending on what convention we want to use
                qubits_to_update.append(i)
    k = RUSk
    
    a1_quantum = np.reshape(((0.5*np.array(attractor1))+0.5),[shape_m,shape_n])
    a2_quantum = np.reshape(((0.5*np.array(attractor2))+0.5),[shape_m,shape_n])
    init_quantum = np.reshape(((0.5*np.array(initstate))+0.5),[shape_m,shape_n])
    dfin1 = pd.DataFrame(a1_quantum)
    dfin2 = pd.DataFrame(a2_quantum)
    dfinit = pd.DataFrame(init_quantum)
    
    #gs_kw = dict(width_ratios=[1,1], height_ratios=[1,1])
    if(ploteach == True):
        num_plots = 3+len(qubits_to_update)
    else:
        num_plots = 4
    cols = 2
    rows = math.ceil(num_plots/cols)
    #rows = 2
    fig, axes = plt.subplots(figsize=(9,6),ncols=cols,nrows=rows,sharex=True,sharey=True)
    axs = [0] * (cols*rows)
    for i, ax in enumerate(axes.flat):
        ax.set_xlim([0,2])
        ax.set_ylim([0,2])
        axs[i] = ax
        #p1 = sns.heatmap(dfin1,cmap='PuBu',ax=ax)
    
    rfsize = fsize-4
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    p0 = sns.heatmap(dfin1,annot=True,annot_kws={"fontsize":fsize},cmap=cmp,cbar=0,cbar_ax=None,ax=axs[0],xticklabels="",yticklabels="",vmin=0, vmax=1)
    p1 = sns.heatmap(dfin2,annot=True,annot_kws={"fontsize":fsize},cmap=cmp,cbar=0,cbar_ax=None,ax=axs[1],xticklabels="",yticklabels="",vmin=0, vmax=1)
    p2 = sns.heatmap(dfinit,annot=True,annot_kws={"fontsize":fsize},cmap=cmp,cbar=0,cbar_ax=None,ax=axs[2],xticklabels="",yticklabels="",vmin=0, vmax=1)
    axs[0].set_title('Attractor 1',size=rfsize)
    axs[1].set_title('Attractor 2',size=rfsize)
    axs[2].set_title('Initial Corrupted State',size=rfsize)
    
    #init here
    num_qubits = n+k
    qc = QuantumCircuit(num_qubits,n)
    for q in range(num_qubits-1):
        qc.reset(q)
    for q in range(n):
        if(initstate[q] == 1):
            qc.x(q)
        elif(initstate[q] == 0): #not(isinstance(initstate[q],int))
            qc.h(q)
    
    tries = num_tries
    plot_num = 2
    num_updates = 0
    for trie in range(tries):    
        for updated_qubit in qubits_to_update:
            print("Updating qubit " + str(updated_qubit))
            qc.reset(num_qubits-1)
            controls = [i for i in range(n) if i!=updated_qubit]
            gamma = math.pi/(4*wm.max()*(n-1)+max(thresholds)) #n-1 is because we need size of control layer
            beta = (math.pi/4)+gamma*(thresholds[updated_qubit]-sum(wm[updated_qubit][controls]))
            add_RUS1_JustRy_with_bias(qc,4*gamma*wm[updated_qubit][controls],controls,2*beta,n)
            qc.swap(updated_qubit,n)
            plot_num = plot_num + 1
            num_updates = num_updates + 1
            if(ploteach == True):
                for q in range(n):
                    qc.measure(q,q)
                if useNoise == True:
                    result = execute(qc, backend=backend, shots=nshots, noise_model=nm).result()
                else:
                    result = execute(qc, backend=backend, shots=nshots).result()
                print("Result Completed")
                counts = result.get_counts()

                each_reg_count = np.zeros(n)
                for key, value in counts.items():
                    for i in range(n):
                        if key[n-i-1] == '1':
                            each_reg_count[i] = each_reg_count[i] + value
                #print(each_reg_count)
                reg_density = np.reshape((each_reg_count/nshots),[shape_m,shape_n])    
                dfdensity = pd.DataFrame(reg_density)
                #print(dfdensity)
                
                p_update = sns.heatmap(dfdensity,annot=True,annot_kws={"fontsize":fsize},cmap=cmp,ax=axs[plot_num],xticklabels="",yticklabels="") #square=True
                if(num_updates == len(qubits_to_update)):
                    if(num_updates == 1):
                        axs[plot_num].set_title('Final State (' + str(num_updates) + ' Update)', size=rfsize)
                    else:
                        axs[plot_num].set_title('Final State (' + str(num_updates) + ' Updates)', size=rfsize)
                elif(num_updates == 1):
                    axs[plot_num].set_title('State After ' + str(num_updates) + ' Update', size=rfsize)
                else:
                    axs[plot_num].set_title('State After ' + str(num_updates) + ' Updates', size=rfsize)
                cbar.ax.tick_params(labelsize=20)
    #print("We got out of the loops!")
    if(ploteach == False):
        for q in range(n):
            qc.measure(q,q)
        print("Executing...")
        if useNoise == True:
            result = execute(qc, backend=be, shots=nshots, noise_model=nm).result()
        else:
            result = execute(qc, backend=be, shots=nshots).result()
        print("Result Completed")
        counts = result.get_counts()

        each_reg_count = np.zeros(n)
        for key, value in counts.items():
            for i in range(n):
                if key[n-i-1] == '1':
                    each_reg_count[i] = each_reg_count[i] + value
        #print(each_reg_count)
        reg_density = np.reshape((each_reg_count/nshots),[shape_m,shape_n])    
        dfdensity = pd.DataFrame(reg_density)
        #print(dfdensity)
        p_all = sns.heatmap(dfdensity,annot=True,annot_kws={"fontsize":fsize},cmap=cmp,cbar_ax=cbar_ax,ax=axs[3],xticklabels="",yticklabels="",vmin=0, vmax=1)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=20)
        if(num_updates == 1):
            axs[3].set_title('Final State (' + str(num_updates) + ' Update)', size=rfsize)
        else:
            axs[3].set_title('Final State (' + str(num_updates) + ' Updates)', size=rfsize)
    plt.tight_layout(rect=[0, 0, .9, 1])
    plt.show()
    fig.savefig('test.png', bbox_inches='tight',dpi=600)
    
    return qc # return full hop quantum circuit

################################################################################################################################
# Testing Effective Memory Capacity
################################################################################################################################

# Generic weight setting for any number of attractors of any given length
def hop_weight_setting_X_attractors(us):
    X = len(us)
    length = len(us[0])
    wm = np.zeros((length, length))
    for i in range(length):
        for j in range(length):
            for u in us:
                wm[i][j] = wm[i][j] + u[i]*u[j]
            wm[i][j] = wm[i][j]/X
    return wm

# create hopfield network in a single function using given attractors and initial state
def create_hopfield_simplified_forCapacityCalc_noresets(initstate,attractors,updates,be,useNoise,nm,nshots):
    n = len(attractors[0])
    if(n != len(initstate)):
        print("Error: Attractors and input states must be the same length.")
        return
    wm = hop_weight_setting_X_attractors(attractors)
    thresholds = [0]*n
    qubits_to_update = updates
    
    attractors_quantum = []
    for attractor in attractors:
        attractors_quantum.append(0.5*np.array(attractor)+0.5)
    init_quantum = 0.5*np.array(initstate)+0.5
    
    num_qubits = n+len(qubits_to_update)
    qc = QuantumCircuit(num_qubits,n)
    for q in range(num_qubits):
        qc.reset(q)
    for q in range(n):
        if(initstate[q] == 1):
            qc.x(q)
        elif(initstate[q] == 0):
            qc.h(q)
    
    num_updates = 0  
    for updated_qubit in qubits_to_update:
        controls = [i for i in range(n) if i!=updated_qubit]
        gamma = math.pi/(4*wm.max()*(n-1)+max(thresholds)) #n-1 is because we need size of control layer
        beta = (math.pi/4)+gamma*(thresholds[updated_qubit]-sum(wm[updated_qubit][controls]))
        add_RUS1_JustRy_with_bias(qc,4*gamma*wm[updated_qubit][controls],controls,2*beta,n+num_updates)
        qc.swap(updated_qubit,n+num_updates)
        num_updates = num_updates + 1
    
    for q in range(n):
        qc.measure(q,n-q-1) # perform measurements backwards to get around order convention
    #print("Executing...")
    if useNoise == True:
        result = execute(qc, backend=be, shots=nshots, noise_model=nm).result()
    else:
        job = execute(qc, backend=be, shots=nshots)
        result = job.result()
     
    counts = result.get_counts()

    each_reg_count = np.zeros(n)
    for key, value in counts.items():
        for i in range(n):
            if key[i] == '1':
                each_reg_count[i] = each_reg_count[i] + value
    reg_density = each_reg_count/nshots    
    
    return counts, reg_density


def create_hopfield_simplified_forCapacityCalc_withresets(initstate,attractors,updates,be,useNoise,nm,nshots):
    n = len(attractors[0])
    if(n != len(initstate)):
        print("Error: Attractors and input states must be the same length.")
        return
    wm = hop_weight_setting_X_attractors(attractors)
    thresholds = [0]*n
    qubits_to_update = updates
    
    attractors_quantum = []
    for attractor in attractors:
        attractors_quantum.append(0.5*np.array(attractor)+0.5)
    init_quantum = 0.5*np.array(initstate)+0.5
    
    num_qubits = n+1
    qc = QuantumCircuit(num_qubits,n)
    for q in range(num_qubits):
        qc.reset(q)
    for q in range(n):
        if(initstate[q] == 1):
            qc.x(q)
        elif(initstate[q] == 0):
            qc.h(q)
    
    num_updates = 0  
    for updated_qubit in qubits_to_update:
        #print("Updating qubit " + str(updated_qubit))
        qc.reset(num_qubits-1)
        controls = [i for i in range(n) if i!=updated_qubit]
        gamma = math.pi/(4*wm.max()*(n-1)+max(thresholds)) #n-1 is because we need size of control layer
        beta = (math.pi/4)+gamma*(thresholds[updated_qubit]-sum(wm[updated_qubit][controls]))
        add_RUS1_JustRy_with_bias(qc,4*gamma*wm[updated_qubit][controls],controls,2*beta,n)
        qc.swap(updated_qubit,n)
        num_updates = num_updates + 1
    
    for q in range(n):
        qc.measure(q,n-q-1) # perform measurements backwards to get around order convention
    #print("Executing...")
    if useNoise == True:
        result = execute(qc, backend=be, shots=nshots, noise_model=nm).result()
    else:
        job = execute(qc, backend=be, shots=nshots)
        #job_monitor(job)
        result = job.result()
     
    counts = result.get_counts()

    each_reg_count = np.zeros(n)
    for key, value in counts.items():
        for i in range(n):
            if key[i] == '1':
                each_reg_count[i] = each_reg_count[i] + value
    reg_density = each_reg_count/nshots    
    
    return counts, reg_density

def create_hopfield_simplified_forCapacityCalc_parallel(initstate,attractors,updates,be,useNoise,nm,nshots):
    n = len(attractors[0])
    if(n != len(initstate)):
        print("Error: Attractors and input states must be the same length.")
        return
    wm = hop_weight_setting_X_attractors(attractors)
    thresholds = [0]*n
    qubits_to_update = updates
    
    attractors_quantum = []
    for attractor in attractors:
        attractors_quantum.append(0.5*np.array(attractor)+0.5)
    init_quantum = 0.5*np.array(initstate)+0.5
    
    num_qubits = 2*n
    qc = QuantumCircuit(num_qubits,n)
    for q in range(num_qubits):
        qc.reset(q)
    for q in range(n):
        if(initstate[q] == 1):
            qc.x(q)
        elif(initstate[q] == 0):
            qc.h(q)
    
    num_updates = 0  
    for updated_qubit in qubits_to_update:
        #print("Updating qubit " + str(updated_qubit))
        controls = [i for i in range(n) if i!=updated_qubit]
        gamma = math.pi/(4*wm.max()*(n-1)+max(thresholds)) #n-1 is because we need size of control layer
        beta = (math.pi/4)+gamma*(thresholds[updated_qubit]-sum(wm[updated_qubit][controls]))
        add_RUS1_JustRy_with_bias(qc,4*gamma*wm[updated_qubit][controls],controls,2*beta,n+num_updates)
        num_updates = num_updates + 1
    
    for q in range(n):
        qc.measure(n+q,n-qubits_to_update[q]-1) # perform measurements backwards to get around order convention
    #print("Executing...")
    if useNoise == True:
        result = execute(qc, backend=be, shots=nshots, noise_model=nm).result()
    else:
        job = execute(qc, backend=be, shots=nshots)
        #job_monitor(job)
        result = job.result()
    
    #print("Result Completed")    
    counts = result.get_counts()

    each_reg_count = np.zeros(n)
    for key, value in counts.items():
        for i in range(n):
            if key[i] == '1':
                each_reg_count[i] = each_reg_count[i] + value
    #print(each_reg_count)
    reg_density = each_reg_count/nshots    
    
    return counts, reg_density


def generate_memories_and_probe(m,n):
    size = 2**n
    possible_mems = np.arange(size)
    binary_mems = np.empty(shape=(0),dtype=str)
    for pm in possible_mems:
        binary_mems = np.append(binary_mems,["{0:b}".format(pm).zfill(n)],0)
    chosen_ms = np.random.choice(binary_mems, size=m, replace=False)
    memories = np.empty((0,n))
    for m in chosen_ms:
        hop_m = np.empty((0))
        for i in m:
            hop_m = np.append(hop_m,[2*(int(i,2))-1],0)
        memories = np.append(memories,[hop_m],0)
    
    generate_again = True
    p_limit = 0.01*n
    while(generate_again):
        probe = np.random.choice([-1,1],n)
        best = len(probe)+1
        bad = False
        target = np.array([])
        for memory in memories:
        #print(memory)
            p = np.count_nonzero(memory-probe)
            if(p<best):
                best = p
                target = memory
            elif(p==best):
                bad = True
        if(not(bad or (best>=p_limit))):
            generate_again = False
    
    return memories, probe, target

def generate_memories_and_probe_independent(m,n):
    identity = np.identity(n)
    rng = np.arange(n)
    indices = np.random.choice(rng, size=m, replace=False)
    memories = identity[indices,:]
    probe_index = np.random.choice(rng)
    probe = identity[probe_index,:]
    return memories, probe, target

def convert_states_to_quantum_and_string(states):
    quantum_states = np.empty((0,len(states[0])))
    str_states = np.array([])
    for state in states:
        quantum_state = (0.5*state+0.5).astype(int)
        quantum_states = np.append(quantum_states,[(0.5*state+0.5).astype(int)],0)
        str_states = np.append(str_states, np.array2string(quantum_state,separator='')[1:-1])

    return quantum_states, str_states




## capacity experiments #########################################################################
#################################################################################################

provider = IBMQ.get_provider(hub='ibm-q')
device = provider.get_backend('ibmq_quito') #ibmq_16_melbourne used originally, now retired
backend = Aer.get_backend('qasm_simulator')
noise_model = NoiseModel.from_backend(device)

shots = 1024
num_tests = 1000
num_updates = [6] #[4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
ns = [4] #4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20
ms = [1, 2, 3, 4] #2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20
accuracy_results = np.empty((0,7))
for n in ns:
    for m in ms:
        for num_update in num_updates:
            if (num_update < n) or (num_update > 2*n):
                print("continued")
                continue
            total_accuracy = 0
            correct_majorities = 0
            most_often_accuracy = 0
            num_ran = 0
            total_density_accuracy = 0
            while(num_ran < num_tests):
                memories, probe, target = generate_memories_and_probe(m,n)
                num_ran = num_ran + 1
                qubit_list = np.arange(len(probe))
                update_list = np.random.choice(qubit_list, size=num_update, replace=True)
                #print(update_list)
                quantum_target = (0.5*target+0.5).astype(int)
                str_target = np.array2string(quantum_target,separator='')[1:-1]
                counts, density = create_hopfield_simplified_forCapacityCalc_withresets(probe,memories,update_list,backend,False,noise_model,shots)
                #if(num_ran%10==0):
                #    print(n,m,num_ran)
                if str_target in counts:
                    accuracy = counts[str_target]/shots
                else:
                    accuracy = 0
                
                most_often = max(counts, key=counts.get)
                
                density_accuracy = 0
                for i in range(len(quantum_target)):
                    if quantum_target[i] == 0:
                        density_accuracy = density_accuracy + (1-density[i])
                    else:
                        density_accuracy = density_accuracy + density[i]
                total_density_accuracy = total_density_accuracy + (density_accuracy/n)
                majority_vote_result = np.round(density).astype(int)
                quantum_probe = (0.5*probe+0.5).astype(int)
                if((quantum_target == majority_vote_result).all()):
                    correct_majorities = correct_majorities + 1
                if(str_target == most_often):
                    most_often_accuracy = most_often_accuracy + 1
                total_accuracy = total_accuracy + accuracy
            print('n = ' + str(n) + ', m = ' + str(m) + ', num_ran = ' + str(num_ran) + ', num_updates = ' + str(num_update))
            print('Percent Correct Majority Vote: ' + str(correct_majorities/num_ran))
            print('Most Often Occurance Accuracy: ' + str(most_often_accuracy/num_ran))
            print('Total Average Density: ' + str(total_density_accuracy/num_ran))
            print('Total Percent Exact Accuracy: ' + str(total_accuracy/num_ran))
            accuracy_results = np.append(accuracy_results,[[n,m,num_ran,correct_majorities/num_ran,total_density_accuracy/num_ran,total_accuracy/num_ran,num_update]],0) #,oneOffAccuracy/num_ran,twoOffAccuracy/num_ran
            # return accuracy_results
print(accuracy_results)    