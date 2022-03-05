from self_py_fun.NNAttentionFun import *
from self_py_fun.ExistMLFun import *
import self_py_fun.GlobalSIM as sg
sns.set_context('notebook')

if sg.local_use:
    DATA_DIC = '/Users/niubilitydiu/Box Sync/Dissertation/Dataset and Rcode/EEG_MATLAB_data'
    SAVE_DIC = '/Users/niubilitydiu/Box Sync/Dissertation/Dataset and Rcode/EEG_MATLAB_data/SIM_files/'
else:
    DATA_DIC = '/home/mtianwen/EEG_MATLAB_data'
    SAVE_DIC = '/home/mtianwen/EEG_MATLAB_data/SIM_files/'

sim_name_short = 'sim_' + str(sg.design_num + 1)
method_name = 'NNAttn'
sim_type_2 = sg.sim_type + '_down_' + sg.scenario_name
results = get_nn_attention_results(
    sg.design_num+1, sg.subset_num+1, sg.scenario_name,
    sg.repet_num_fit,
    sg.REPETITION_TEST,
    datadic=DATA_DIC, savedic=SAVE_DIC
)

# print(results.keys())
real_label = results['real_label']
test_label = results['test_label']
test_accuracy = results['test_accuracy']

print(results.keys())
# print('test_label: ', test_label)
# print(test_label.shape)
# print(test_label)
print('test_accuracy: ', test_accuracy)

print(results)
