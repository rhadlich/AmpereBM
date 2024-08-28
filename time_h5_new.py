import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

lw = 3
plt.rcParams.update({'font.size': 14})

dir = os.getcwd()
filedir = os.path.join(dir, 'new_time.h5')

with h5py.File(filedir, 'a') as f:
    # # sanity check
    # grp1 = f['small']
    # print(f'small {grp1.keys()}')
    # grp2 = grp1['48']
    # print(grp2.keys())
    # print(grp2['epoch time'][...])
    # grp1 = f['large']
    # print(f'large {grp1.keys()}')
    # grp2 = grp1['48']
    # print(grp2.keys())
    # print(grp2['epoch time'][...])

    # Small Model
    # Make dataframe
    proc_per_node = 16
    grp1 = f['small']
    grp3 = f['large']
    grp5 = f['medium']
    small = []
    large = []
    medium = []
    for i, key in enumerate(grp1.keys()):
        small.append({})
        large.append({})
        medium.append({})
        grp2 = grp1[key]
        grp4 = grp3[key]
        grp6 = grp5[key]
        for key2 in grp2.keys():
            small[i]['Nodes'] = large[i]['Nodes'] = medium[i]['Nodes'] = int(int(key) / proc_per_node)
            small[i][key2] = np.mean(grp2[key2][...])
            large[i][key2] = np.mean(grp4[key2][...])
            medium[i][key2] = np.mean(grp6[key2][...])
            if key2 == 'total time':
                small[i]['total time std'] = np.std(grp2[key2][...])
                cov = small[i]['total time std'] / small[i][key2]
                small[i]['total time cov'] = cov

                large[i]['total time std'] = np.std(grp4[key2][...])
                cov = large[i]['total time std'] / large[i][key2]
                large[i]['total time cov'] = cov

                medium[i]['total time std'] = np.std(grp6[key2][...])
                cov = medium[i]['total time std'] / medium[i][key2]
                medium[i]['total time cov'] = cov

    print(small)

    # small dataframe
    df1 = pd.DataFrame(small).sort_values('Nodes')
    df1 = df1.reset_index(drop=True)
    print(df1)

    # medium dataframe
    df2 = pd.DataFrame(medium).sort_values('Nodes')
    df2 = df2.reset_index(drop=True)
    print(df2)

    # large dataframe
    df3 = pd.DataFrame(large).sort_values('Nodes')
    df3 = df3.reset_index(drop=True)
    print(df3)


    def calc_n_examples(nodes):
        n_proc = nodes * 16
        steps = int(23100 / 4 / n_proc)
        return n_proc * 4 * steps


    def comp_time_theoretical(nodes):
        # return (nodes - df1['Nodes'][0])*100
        return nodes


    def comp_time_actual(time):
        return df1['total time'][0] / time


    def comp_time_epoch_actual(time):
        return df1['epoch time'][0] / time


    # plotting small model
    df1['theoretical speed up'] = df1['Nodes'].apply(comp_time_theoretical)
    df1['actual speed up'] = df1['total time'].apply(comp_time_actual)
    df1['actual speed up (epoch)'] = df1['epoch time'].apply(comp_time_epoch_actual)
    df1['speed up error'] = df1['actual speed up'] * df1['total time cov']
    df1['num examples'] = df1['Nodes'].apply(calc_n_examples)
    max_ex = df1['num examples'].max()
    df1['normalized speed up'] = df1['actual speed up'] * df1['num examples'] / max_ex
    df1['normalized error'] = df1['normalized speed up'] * df1['total time cov']
    print(df1[['theoretical speed up', 'actual speed up', 'num examples', 'normalized speed up']])

    # total time
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, layout='constrained', sharey=True, figsize=(17, 4))
    df1.plot(kind='line', linewidth=lw,
             x='Nodes', y='theoretical speed up', color='#d95f02', ax=ax1, label='Theoretical Speedup')
    df1.plot(kind='line', linewidth=lw,
             x='Nodes', y='actual speed up', yerr='speed up error', capsize=4, color='#7570b3', ax=ax1,
             linestyle='dotted', label='Actual Speedup')
    df1.plot(kind='line', linewidth=lw,
             x='Nodes', y='normalized speed up', yerr='normalized error', capsize=4, color='#1b9e77',
             linestyle='dashed', ax=ax1, label='Normalized Speedup')
    ax1.set_ylabel('Computation Speed-Up\n(relative to one node)', multialignment='center')
    ax1.set_xlabel('Number of Nodes')
    ax1.set_title('Small Model')
    # plt.title('Small Model')
    ax1.grid(True)
    ax1.set_xlim([0, df1['Nodes'].max() + 5])
    ax1.set_ylim([0, df1['theoretical speed up'].max()+40])
    # plt.savefig('Speedup Small Mapping.png', dpi=600, format='png')
    # plt.show()
    # plt.close()
    #
    # # # plot mse
    # # ax = plt.gca()
    # # df1.plot(kind='line', x='Nodes', y='mse', ax=ax)
    # # ax.get_legend().remove()
    # # plt.ylabel('Mean Squared Error')
    # # plt.xlabel('Number of Processes')
    # # plt.title('Large Model')
    # # plt.grid(True)
    # # plt.savefig('MSE Small Mapping.png', dpi=600, format='png')
    # # plt.close()

    # plot medium model

    def comp_time_actual2(time):
        return df2['total time'][0] / time


    def comp_time_epoch_actual2(time):
        return df2['epoch time'][0] / time

    df2['theoretical speed up'] = df2['Nodes'].apply(comp_time_theoretical)
    df2['actual speed up'] = df2['total time'].apply(comp_time_actual2)
    df2['actual speed up (epoch)'] = df2['epoch time'].apply(comp_time_epoch_actual2)
    df2['speed up error'] = df2['actual speed up'] * df2['total time cov']
    df2['num examples'] = df2['Nodes'].apply(calc_n_examples)
    max_ex = df2['num examples'].max()
    df2['normalized speed up'] = df2['actual speed up'] * df2['num examples'] / max_ex
    df2['normalized error'] = df2['normalized speed up'] * df2['total time cov']
    print(df2[['theoretical speed up', 'actual speed up', 'num examples', 'normalized speed up']])

    # total time
    # ax = plt.gca()
    df2.plot(kind='line', linewidth=lw,
             x='Nodes', y='theoretical speed up', color='#d95f02', ax=ax2, label='Theoretical Speedup')
    df2.plot(kind='line', linewidth=lw,
             x='Nodes', y='actual speed up', yerr='speed up error', capsize=4, color='#7570b3',
             linestyle='dotted', ax=ax2, label='Actual Speedup')
    df2.plot(kind='line', linewidth=lw,
             x='Nodes', y='normalized speed up', yerr='normalized error', capsize=4, color='#1b9e77',
             linestyle='dashed', ax=ax2, label='Normalized Speedup')
    # ax1.set_ylabel('Computation Speed-Up (relative to one node)')
    ax2.set_xlabel('Number of Nodes')
    # plt.title('Large Model')
    ax2.grid(True)
    ax2.set_xlim([0, df2['Nodes'].max() + 5])
    ax2.set_title('Optimal Model')

    # plot large model

    def comp_time_actual3(time):
        return df3['total time'][0] / time


    def comp_time_epoch_actual3(time):
        return df3['epoch time'][0] / time


    df3['theoretical speed up'] = df3['Nodes'].apply(comp_time_theoretical)
    df3['actual speed up'] = df3['total time'].apply(comp_time_actual3)
    df3['actual speed up (epoch)'] = df3['epoch time'].apply(comp_time_epoch_actual3)
    df3['speed up error'] = df3['actual speed up'] * df3['total time cov']
    df3['num examples'] = df3['Nodes'].apply(calc_n_examples)
    max_ex = df3['num examples'].max()
    df3['normalized speed up'] = df3['actual speed up'] * df3['num examples'] / max_ex
    df3['normalized error'] = df3['normalized speed up'] * df3['total time cov']
    print(df3[['theoretical speed up', 'actual speed up', 'num examples', 'normalized speed up']])

    # total time
    # ax = plt.gca()
    df3.plot(kind='line', linewidth=lw,
             x='Nodes', y='theoretical speed up', color='#d95f02', ax=ax3, label='Theoretical Speedup')
    df3.plot(kind='line',  linewidth=lw,
             x='Nodes', y='actual speed up', yerr='speed up error', capsize=4, color='#7570b3',
             linestyle='dotted', ax=ax3, label='Actual Speedup')
    df3.plot(kind='line', linewidth=lw,
             x='Nodes', y='normalized speed up', yerr='normalized error', capsize=4, color='#1b9e77',
             linestyle='dashed', ax=ax3, label='Normalized Speedup')
    # ax3.ylabel('Computation Speed-Up (relative to one node)')
    ax3.set_xlabel('Number of Nodes')
    # plt.title('Large Model')
    ax3.grid(True)
    ax3.set_xlim([0, df3['Nodes'].max() + 5])
    ax3.set_title('Large Model')


    # secondaty x-axis
    def node2proc(x):
        return x * 16


    def proc2node(x):
        return x / 16


    # Secondary axes
    # secax = ax1.secondary_xaxis('top', functions=(node2proc, proc2node))
    # secax.set_xlabel('Total Number of Processes (16 per node)')
    # secax = ax2.secondary_xaxis('top', functions=(node2proc, proc2node))
    # secax.set_xlabel('Total Number of Processes (16 per node)')
    # secax = ax3.secondary_xaxis('top', functions=(node2proc, proc2node))
    # secax.set_xlabel('Total Number of Processes (16 per node)')

    # plt.ylim([0, df2['theoretical speed up'].max() + 10])
    plt.savefig('Speedup Time Combined.png', dpi=600, format='png')
    plt.show()
    plt.close()
    #

    # # Plot MAE
    # df1['Processes'] = df2['Processes'] = df3['Processes'] = df1['Nodes']*proc_per_node
    # fig, ax1 = plt.subplots(1, 1, layout='constrained', sharey=True, figsize=(7, 5))
    # df1.plot(kind='line', x='Processes', y='mae', color='#c7e9b4', ax=ax1, label='Small Model')
    # df2.plot(kind='line', x='Processes', y='mae', color='#41b6c4', ax=ax1, label='Optimal Model')
    # df3.plot(kind='line', x='Processes', y='mae', color='#253494', ax=ax1, label='Large Model')
    # # ax1.get_legend().remove()
    # ax1.set_ylabel('Mean Absolute Error')
    # ax1.set_xlabel('Number of Processes')
    # # ax1.set_title('Large Model')
    # plt.grid(True)
    # plt.show()
    # # plt.savefig('MSE Small Mapping.png', dpi=600, format='png')
    # plt.close()

