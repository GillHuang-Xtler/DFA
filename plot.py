import matplotlib.pyplot as plt
import numpy as np
import umap
from tqdm.auto import tqdm

def plt_txt():
    filename = 'iid-bart.txt'
    X,Y = [],[]
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split()]
            X.append(value[0])
    print(X)

    plt.plot(X)
    plt.show()

def plt_acc():
    path1 = './res/9140_results.csv'
    filename1 = path1
    X1 = []
    with open(filename1, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X1.append(value[0])
    Y1 = (max(X1)-max(X1))

    path2 = './res/9141_results.csv'
    filename2 = path2
    X2 = []
    with open(filename2, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X2.append(value[0])
    Y2 = (max(X1)-max(X2))

    path3='./res/9145_results.csv'
    filename3 = path3
    X3 = []
    with open(filename3, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X3.append(value[0])
    Y3 = (max(X1)-max(X3))

    path4='./res/9147_results.csv'
    filename4 = path4
    X4 = []
    with open(filename4, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X4.append(value[0])
    Y4 = (max(X1)-max(X4))

    path5='./res/9191_results.csv'
    filename5 = path5
    X5 = []
    with open(filename5, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X5.append(value[0])
    Y5 = (max(X1)-max(X5))

    path6='./res/9192_results.csv'
    filename6 = path6
    X6 = []
    with open(filename6, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X6.append(value[0])
    Y6 = (max(X1)-max(X6))
    #
    # path7='./res/1177_results.csv'
    # filename7 = path7
    # X7 = []
    # with open(filename7, 'r') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         value = [float(s) for s in line.split(',')]
    #         X7.append(value[0])
    # print(X7)
    #
    # path8='./res/1178_results.csv'
    # filename8 = path8
    # X8 = []
    # with open(filename8, 'r') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         value = [float(s) for s in line.split(',')]
    #         X8.append(value[0])
    # print(X8)
    #
    # path9='./res/1179_results.csv'
    # filename9 = path9
    # X9 = []
    # with open(filename9, 'r') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         value = [float(s) for s in line.split(',')]
    #         X9.append(value[0])
    # print(X9)
    #
    # path10='./res/1170_results.csv'
    # filename10 = path10
    # X10 = []
    # with open(filename10, 'r') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         value = [float(s) for s in line.split(',')]
    #         X10.append(value[0])
    # print(X10)
    plt.figure(1)
    # plt.subplot(211)
    plt.plot(X1, label='avg')
    # plt.plot(X2, label='free')
    plt.plot(X3, label='free-rand')
    plt.plot(X4, label='free-rand-mkrum')
    plt.plot(X5, label='free-rand3')
    plt.plot(X6,  label='free-rand3-mkrum')
    # plt.plot(X7, color='green', label='Reduce-class-quitp')
    # plt.plot(X8, color='skyblue', label='Reduce-class-quito')
    # plt.plot(X9, color='purple', label='Reduce-class-introp')
    # plt.plot(X10, color='gray', label='Reduce-class-introo')
    plt.legend()
    plt.xlabel('GLOBAL ROUNDS')
    plt.ylabel('ACCURACY')



    # plt.subplot(212)
    plt.figure(2)
    index_ls = ['avg',"free-rand", "free-rand-mkrum", "free-rand3", "free-rand3-mkrum"]
    asr_ls = [Y1, Y3, Y4, Y5, Y6]
    plt.bar(index_ls, asr_ls )

    plt.legend()
    plt.xlabel('DIFFERENT STRATEGIES')
    plt.ylabel('ASR')


    plt.show()

def plt_class_recall_1():
    path1 = './res/1171_results.csv'
    filename1 = path1
    X1 = []
    with open(filename1, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X1.append(value[13])

    print(X1)

    path2 = './res/1172_results.csv'
    filename2 = path2
    X2 = []
    with open(filename2, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X2.append(value[13])
    print(X2)

    path3 = './res/1191_results.csv'
    filename3 = path3
    X3 = []
    with open(filename3, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X3.append(value[13])
    print(X3)

    path4 = './res/1091_results.csv'
    filename4 = path4
    X4 = []
    with open(filename4, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X4.append(value[13])
    print(X4)
    # plt.plot(X4)

    path5='./res/1113_results.csv'
    filename5 = path5
    X5 = []
    with open(filename5, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X5.append(value[13])
    print(X5)

    path6='./res/1122_results.csv'
    filename6 = path6
    X6 = []
    with open(filename6, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X6.append(value[13])
    print(X6)

    path7='./res/1177_results.csv'
    filename7 = path7
    X7 = []
    with open(filename7, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X7.append(value[13])
    print(X7)

    path9='./res/1179_results.csv'
    filename9 = path9
    X9 = []
    with open(filename9, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X9.append(value[13])
    print(X9)

    plt.plot(X1, color='yellow', label='IID', linestyle=':', marker = 'o', markersize = 2)
    plt.plot(X2, color='black', label='Reduce-class',linestyle=':', marker = 'o', markersize = 2)
    plt.plot(X3, color='brown', label='Reduce-class-plus',linestyle=':', marker = 'o', markersize = 2)
    # plt.plot(X4, color='skyblue', label='Reduce-class-only',linestyle=':', marker = 'o', markersize = 2)
    # plt.plot(X5, color='orange', label='Reduce-class-must',linestyle=':', marker = 'o', markersize = 2)
    plt.plot(X6, color='purple', label='Reduce-class-mustp',linestyle=':', marker = 'o', markersize = 2)
    # plt.plot(X7, color='green', label='Reduce-class-quitp',linestyle=':', marker = 'o', markersize = 2)
    # plt.plot(X9, color='purple', label='Reduce-class-introp',linestyle=':', marker = 'o', markersize = 2)

    plt.legend()
    plt.xlabel('GLOBAL ROUNDS')
    plt.ylabel('SOURCE CLASS RECALL')
    plt.show()

def find_count(X1):
    max_acc = 75
    max40 = max_acc * 0.7
    max60 = max_acc * 0.8
    max80 = max_acc * 0.9

    res4 = []
    res6 = []
    res8 = []
    for i in range(len(X1)):
        if X1[i] > max40:
            res4.append(i)
        if X1[i] > max60:
            res6.append(i)
        if X1[i] > max80:
            res8.append(i)
    return [res4[0], res6[0], res8[0]]


def nun_maverick():

    path3 = './res/1191_results.csv'
    filename3 = path3
    X3 = []
    with open(filename3, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X3.append(value[13])

    path4 = './res/1091_results.csv'
    filename4 = path4
    X4 = []
    with open(filename4, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X4.append(value[13])


    path5 = './res/1113_results.csv'
    filename5 = path5
    X5 = []
    with open(filename5, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X5.append(value[13])

    name_list = ['Random', 'FedFast', 'New']
    num_list1 = []
    num_list2 = []
    num_list3 = []
    num_list1.extend([X3[0], X4[0], X5[0]])
    num_list2.extend([X3[1], X4[1], X5[1]])
    num_list3.extend([X3[2], X4[2], X5[2]])

    x = list(range(len(num_list1)))
    total_width, n = 0.8, 2
    width = total_width / n

    plt.bar(x, num_list1, width=width, label='1 maverick', fc='y')
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, num_list2, width=width, label='2 maverick', tick_label=name_list, fc='r')
    plt.bar(x, num_list3, width=width, label='3 maverick', tick_label=name_list, fc='r')

    plt.legend()
    plt.show()

    # path6='./res/1122_results.csv'
    # filename6 = path6
    # X6 = []
    # with open(filename6, 'r') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         value = [float(s) for s in line.split(',')]
    #         X6.append(value[13])
    # print(X6)
    #
    # path7='./res/1177_results.csv'
    # filename7 = path7
    # X7 = []
    # with open(filename7, 'r') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         value = [float(s) for s in line.split(',')]
    #         X7.append(value[13])
    # print(X7)
    #
    # path9='./res/1179_results.csv'
    # filename9 = path9
    # X9 = []
    # with open(filename9, 'r') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         value = [float(s) for s in line.split(',')]
    #         X9.append(value[13])
    # print(X9)
    #
    # plt.plot(X1, color='yellow', label='IID', linestyle=':', marker = 'o', markersize = 2)
    # plt.plot(X2, color='black', label='Reduce-class',linestyle=':', marker = 'o', markersize = 2)
    plt.plot(X3, color='brown', label='Reduce-class-plus',linestyle=':', marker = 'o', markersize = 2)
    # plt.plot(X4, color='skyblue', label='Reduce-class-only',linestyle=':', marker = 'o', markersize = 2)
    # plt.plot(X5, color='orange', label='Reduce-class-must',linestyle=':', marker = 'o', markersize = 2)
    # plt.plot(X6, color='purple', label='Reduce-class-mustp',linestyle=':', marker = 'o', markersize = 2)

def plt_utility():
    path1 = './shapley-cifar.txt'
    filename1 = path1
    X1, Y1 = [0], [1]
    with open(filename1, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X1.append(value[0])
            Y1.append(1-((abs(value[1]/value[6]-0.1)+abs(value[2]/value[6]-0.1)+abs(value[3]/value[6]-0.1)+abs(value[4]/value[6]-0.1)+abs(value[5]/value[6]-0.6))))
    X2 = ['REF', 'AVG']
    Y2 = [1, sum(Y1[1:])/(len(Y1)-1)]

    grid = plt.GridSpec(1, 4)
    plt.subplot(grid[0,0:3])
    plt.bar(X1[0],  Y1[0], color = 'r', width=4)
    plt.bar(X1[1:], Y1[1:], color = 'b', width=4)
    plt.plot(X1[1:], Y1[1:], color = 'g',marker = 'o', markersize = 4)
    plt.legend()
    plt.xlabel('GLOBAL ROUNDS')
    plt.ylabel('FAIRNESS UTILITY')
    plt.subplot(grid[0,3])
    plt.bar(X2[0], Y2[0], color = 'r', width=0.5)
    plt.bar(X2[1], Y2[1], color = 'b', width=0.5)
    plt.legend()
    # plt.xlabel('GLOBAL ROUNDS')
    plt.tight_layout()
    plt.show()
    # print(X1, Y1)

def sub_plot():
    path1 = './res/1013_results.csv'
    filename1 = path1
    X1 = []
    with open(filename1, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X1.append(value[0])

    path2 = './res/1314_results.csv'
    filename2 = path2
    X2 = []
    with open(filename2, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X2.append(value[0])

    path3 = './res/1012_results.csv'
    filename3 = path3
    X3 = []
    with open(filename3, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X3.append(value[0])

    path4 = './res/1313_results.csv'
    filename4 = path4
    X4 = []
    with open(filename4, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X4.append(value[0])

    path6='./res/1215_results.csv'
    filename6 = path6
    X6 = []
    with open(filename6, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X6.append(value[0])

    path7 = './res/3315_results.csv'
    filename7 = path7
    X7 = []
    with open(filename7, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X7.append(value[0])

    path8='./res/1253_results.csv'
    filename8 = path8
    X8 = []
    with open(filename8, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X8.append(value[0])

    path9='./res/3314_results.csv'
    filename9 = path9
    X9 = []
    with open(filename9, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X9.append(value[0])
    plt.figure(22)
    plt.subplot(221)
    plt.plot(X1, color='blue', label='sv', linewidth='1')
    plt.plot(X2, color='red', label='new', linewidth='1')
    plt.xlabel('GLOBAL ROUNDS')
    plt.ylabel('ACCURACY')
    plt.ylim([40,75])
    plt.legend()

    plt.subplot(222)
    plt.plot(X3, color='blue', label='sv', linewidth='1')
    plt.plot(X4, color='red', label='new', linewidth='1')
    plt.xlabel('GLOBAL ROUNDS')
    plt.legend()

    plt.subplot(223)
    plt.plot(X8, color='red', label='new', linewidth='1')
    plt.plot(X9, color='blue', label='sv', linewidth='1')
    plt.xlabel('GLOBAL ROUNDS')
    plt.ylabel('ACCURACY')
    plt.ylim([40,75])
    plt.legend()

    plt.subplot(224)
    plt.plot(X6, color='red', label='new', linewidth='1')
    plt.plot(X7, color='blue', label='sv', linewidth='1')
    plt.xlabel('GLOBAL ROUNDS')
    # plt.ylabel('ACCURACY')
    plt.tight_layout()
    plt.legend()
    plt.show()

def plt_atk_bar():
    X1 = [13.7, 31.7, 40.6, 38.7 ,34.1,  22.2 , 42.1 , 10.2]
    X2 = [30.2, 30.4, 41.1, 27.9, 39.5 , 16.0 ,19.9, 10.2]
    X3 = [1.1, 79.5, 18.7, 24.7, 19.8, 74, 78.3, 0.8]
    index_ls = ['Krum', 'MKrum', 'Bulyan', 'TrMean', 'Median', 'AFA', 'FTrmean', "FoolsG"]
    plt.bar(index_ls, X3)
    plt.xlabel('Defenses')
    plt.ylabel('Attacking Successful Rate')
    plt.legend()
    plt.show()

def plt_agg_comp():
    path0 = './res/4024_results.csv'
    filename0 = path0
    X0 = []
    with open(filename0, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X0.append(value[0])

    path1 = './res/4025_results.csv'
    filename1 = path1
    X1 = []
    with open(filename1, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X1.append(value[0])
    path2 = './res/4021_results.csv'
    filename2 = path2
    X2 = []
    with open(filename2, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X2.append(value[0])
    path3 = './res/4022_results.csv'
    filename3 = path3
    X3 = []
    with open(filename3, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X3.append(value[0])

    filename4 = './res/4026_results.csv'
    X4 = []
    with open(filename4, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X4.append(value[0])

    filename5 = './res/4041_results.csv'
    X5 = []
    with open(filename5, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X5.append(value[0])

    filename6 = './res/4042_results.csv'
    X6 = []
    with open(filename6, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X6.append(value[0])

    filename7 = './res/4141_results.csv'
    X7 = []
    with open(filename7, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X7.append(value[0])

    plt.plot(X0, color='blue', label='IID', linewidth = '1')
    plt.plot(X1, color='black', label='lf', linewidth = '1')
    plt.plot(X2, label='krum', linewidth = '1')
    plt.plot(X3, label='mkrum', linewidth = '1')
    plt.plot(X4, label='bulyan', linewidth = '1')
    plt.plot(X5, label='trmean', linewidth = '1')
    plt.plot(X6, label='median', linewidth = '1')
    plt.plot(X7, label='fgold', linewidth = '1')
    plt.xlabel('GLOBAL ROUNDS')
    plt.ylabel('ACCURACY')
    plt.legend()
    plt.show()

def plt_reverse():
    path0 = './res/4024_results.csv'
    filename0 = path0
    X0 = []
    with open(filename0, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X0.append(value[0])

    path1 = './res/4025_results.csv'
    filename1 = path1
    X1 = []
    with open(filename1, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X1.append(value[0])

    filename2 = './res/5081_results.csv'
    X2 = []
    with open(filename2, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X2.append(value[0])

    filename3 = './res/5082_results.csv'
    X3 = []
    with open(filename3, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X3.append(value[0])

    filename4 = './res/5084_results.csv'
    X4 = []
    with open(filename4, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X4.append(value[0])

    filename5 = './res/5083_results.csv'
    X5 = []
    with open(filename5, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X5.append(value[0])

    filename6 = './res/5102_results.csv'
    X6 = []
    with open(filename6, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X6.append(value[0])

    filename7 = './res/5121_results.csv'
    X7 = []
    with open(filename7, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X7.append(value[0])

    plt.plot(X0, label='IID', linewidth = '1')
    plt.plot(X1, label='labelFlip', linewidth = '1')
    plt.plot(X2, label='reverseAll_1', linewidth = '1')
    # plt.plot(X5, label='reverse5_1', linewidth = '1')
    plt.plot(X4, label='reverse3_1', linewidth = '1')
    plt.plot(X6, label='reverseAllG_1', linewidth = '1')
    plt.plot(X7, label='reverseAllG_2', linewidth = '1')


    plt.xlabel('GLOBAL ROUNDS')
    plt.ylabel('ACCURACY')
    plt.ylim(50)
    plt.legend()
    plt.show()

def plt_non_iid():
    path0 = './res/4024_results.csv'
    filename0 = path0
    X0 = []
    with open(filename0, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X0.append(value[0])

    path1 = './res/5141_results.csv'
    filename1 = path1
    X1 = []
    with open(filename1, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X1.append(value[0])

    filename2 = './res/5151_results.csv'
    X2 = []
    with open(filename2, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X2.append(value[0])

    plt.plot(X0, label='IID', linewidth = '1')
    plt.plot(X1, label='1-class', linewidth = '1')
    plt.plot(X2, label='2-class', linewidth = '1')
    plt.xlabel('GLOBAL ROUNDS')
    plt.ylabel('ACCURACY')
    # plt.ylim(50)
    plt.legend()
    plt.show()

def plt_batch_size():
    path0 = './res/4024_results.csv'
    filename0 = path0
    X0 = []
    with open(filename0, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X0.append(value[0])

    path1 = './res/61300_results.csv'
    filename1 = path1
    X1 = []
    with open(filename1, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X1.append(value[0])

    filename2 = './res/62100_results.csv'
    X2 = []
    with open(filename2, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X2.append(value[0])

    filename3 = './res/6211_results.csv'
    X3 = []
    with open(filename3, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X3.append(value[0])

    filename4 = './res/6212_results.csv'
    X4 = []
    with open(filename4, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X4.append(value[0])

    plt.plot(X0, label='B=4, C=50', linewidth = '1')
    plt.plot(X1, label='B=4, C=100', linewidth = '1')
    plt.plot(X2, label='B=10, C=100', linewidth = '1')
    plt.plot(X3, label='B=100, C=100', linewidth = '1')
    plt.plot(X4, label='B=100, C=1', linewidth = '1')

    plt.xlabel('GLOBAL ROUNDS')
    plt.ylabel('ACCURACY')
    # plt.ylim(50)
    plt.legend()
    plt.show()

def plt_non_iid_100():
    path0 = './res/61300_results.csv'
    filename0 = path0
    X0 = []
    with open(filename0, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X0.append(value[0])
    Y0 = 100 - max(X0)

    path1 = './res/6213_results.csv'
    filename1 = path1
    X1 = []
    with open(filename1, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X1.append(value[0])

    Y1 = 100-max(X1)


    filename2 = './res/6214_results.csv'
    X2 = []
    with open(filename2, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X2.append(value[0])
    print("non2-none"+": "+str(100-max(X2)))
    Y2 = 100-max(X2)

    path3 = './res/6221_results.csv'
    filename3 = path3
    X3 = []
    with open(filename3, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X3.append(value[0])
    Y3 = 100-max(X3)
    print("iid-lie"+": "+str(100-max(X3)))

    path4 = './res/6222_results.csv'
    filename4 = path4
    X4 = []
    with open(filename4, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X4.append(value[0])
    Y4 = 100-max(X4)
    print("non2-lie"+": "+str(100-max(X4)))


    path5='./res/6223_results.csv'
    filename5 = path5
    X5 = []
    with open(filename5, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X5.append(value[0])
    Y5 = 100-max(X5)
    print("non1-lie"+": "+str(100-max(X5)))

    labels = ['no_attack', 'lie']
    x = np.arange(len(labels))

    width = 0.25
    plt.xticks(x, labels=labels)
    iid = [Y0, Y3]
    non_iid_2_class = [Y2, Y4]
    non_iid_1_class = [Y1, Y5]
    plt.bar(x - width, iid, width, label='iid')
    plt.bar(x, non_iid_2_class, width, label='non-iid-2-class')
    plt.bar(x + width, non_iid_1_class, width, label='non-iid-1-class')

    plt.title("Error Rate without Defense")
    plt.legend()
    plt.show()

def plt_non_iid_defense():
    path00 = './res/61300_results.csv'
    filename00 = path00
    X00 = []
    with open(filename00, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X00.append(value[0])
    Y00 = 100 - max(X00)

    path01 = './res/6213_results.csv'
    filename01 = path01
    X01 = []
    with open(filename01, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X01.append(value[0])

    Y01 = 100 - max(X01)

    filename02 = './res/6214_results.csv'
    X02 = []
    with open(filename02, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X02.append(value[0])
    Y02 = 100 - max(X02)

    path0 = './res/6221_results.csv'
    filename0 = path0
    X0 = []
    with open(filename0, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X0.append(value[0])
    Y0 = 100 - max(X0)

    path1 = './res/6222_results.csv'
    filename1 = path1
    X1 = []
    with open(filename1, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X1.append(value[0])
    Y1 = 100-max(X1)


    filename2 = './res/6223_results.csv'
    X2 = []
    with open(filename2, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X2.append(value[0])
    Y2 = 100-max(X2)

    path3 = './res/6231_results.csv'
    filename3 = path3
    X3 = []
    with open(filename3, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X3.append(value[0])
    Y3 = 100-max(X3)

    path4 = './res/6232_results.csv'
    filename4 = path4
    X4 = []
    with open(filename4, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X4.append(value[0])
    Y4 = 100-max(X4)


    path5='./res/6233_results.csv'
    filename5 = path5
    X5 = []
    with open(filename5, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X5.append(value[0])
    Y5 = 100-max(X5)


    path6 = './res/6234_results.csv'
    filename6 = path6
    X6 = []
    with open(filename6, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X6.append(value[0])
    Y6 = 100-max(X6)

    filename7 = './res/6235_results.csv'
    X7 = []
    with open(filename7, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X7.append(value[0])
    Y7 = 100-max(X7)


    path8='./res/6236_results.csv'
    filename8 = path8
    X8 = []
    with open(filename8, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X8.append(value[0])
    Y8 = 100-max(X8)


    filename9 = './res/6237_results.csv'
    X9 = []
    with open(filename9, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X9.append(value[0])
    Y9 = 100-max(X9)

    filename10 = './res/6238_results.csv'
    X10 = []
    with open(filename10, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X10.append(value[0])
    Y10 = 100-max(X10)


    filename11='./res/6239_results.csv'
    X11 = []
    with open(filename11, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X11.append(value[0])
    Y11 = 100-max(X11)

    labels = ['no_defense', 'mkrum',  'bulyan', "trmean"]
    x = np.arange(len(labels))

    width = 0.25
    plt.xticks(x, labels=labels)
    iid = [Y0-Y00, Y3-Y00, Y6-Y00, Y9-Y00]
    non_iid_2_class = [Y1-Y02, Y4-Y02, Y7-Y02, Y10-Y02]
    non_iid_1_class = [Y2-Y01, Y5-Y01, Y8-Y01, Y11-Y01]
    plt.bar(x - width, iid, width, label='iid')
    plt.bar(x, non_iid_2_class, width, label='non-iid-2-class')
    plt.bar(x + width, non_iid_1_class, width, label='non-iid-1-class')

    plt.title("ASR of LIE under Defense")
    plt.legend()
    plt.show()

def plt_non_iid_distribution():

    path0 = './res/61300_results.csv'
    filename0 = path0
    X0 = []
    with open(filename0, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X0.append(value[0])
    Y0 = 100 - max(X0)

    filename01 = './res/7120_results.csv'
    X01 = []
    with open(filename01, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X01.append(value[0])
    Y01 = 100 - max(X01)

    filename02 = './res/6214_results.csv'
    X02 = []
    with open(filename02, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X02.append(value[0])
    Y02 = 100 - max(X02)

    filename03 = './res/7120_results.csv'
    X03 = []
    with open(filename03, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X03.append(value[0])
    Y03 = 100 - max(X03)

    filename04 = './res/7122_results.csv'
    X04 = []
    with open(filename04, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X04.append(value[0])
    Y04 = 100 - max(X04)

    filename05 = './res/7190_results.csv'
    X05 = []
    with open(filename05, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X05.append(value[0])
    Y05 = 100 - max(X05)

    filename06 = './res/7191_results.csv'
    X06 = []
    with open(filename06, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X06.append(value[0])
    Y06 = 100 - max(X06)

    filename07 = './res/7193_results.csv'
    X07 = []
    with open(filename07, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X07.append(value[0])
    Y07 = 100 - max(X07)

    filename08 = './res/7192_results.csv'
    X08 = []
    with open(filename08, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X08.append(value[0])
    Y08 = 100 - max(X08)

    path1 = './res/6221_results.csv'
    filename1 = path1
    X1 = []
    with open(filename1, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X1.append(value[0])
    Y1 = 100-max(X1)


    filename2 = './res/6222_results.csv'
    X2 = []
    with open(filename2, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X2.append(value[0])
    Y2 = 100-max(X2)

    path3 = './res/6231_results.csv'
    filename3 = path3
    X3 = []
    with open(filename3, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X3.append(value[0])
    Y3 = 100-max(X3)

    path4 = './res/6232_results.csv'
    filename4 = path4
    X4 = []
    with open(filename4, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X4.append(value[0])
    Y4 = 100-max(X4)
    print(Y4)

    path5='./res/6234_results.csv'
    filename5 = path5
    X5 = []
    with open(filename5, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X5.append(value[0])
    Y5 = 100-max(X5)


    path6 = './res/6235_results.csv'
    filename6 = path6
    X6 = []
    with open(filename6, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X6.append(value[0])
    Y6 = 100-max(X6)

    filename7 = './res/6237_results.csv'
    X7 = []
    with open(filename7, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X7.append(value[0])
    Y7 = 100-max(X7)


    path8='./res/6238_results.csv'
    filename8 = path8
    X8 = []
    with open(filename8, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X8.append(value[0])
    Y8 = 100-max(X8)

    filename9 = './res/7073_results.csv'
    X9 = []
    with open(filename9, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X9.append(value[0])
    Y9 = 100-max(X9)

    filename10 = './res/7074_results.csv'
    X10 = []
    with open(filename10, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X10.append(value[0])
    Y10 = 100-max(X10)
    print(Y10)


    filename11='./res/7075_results.csv'
    X11 = []
    with open(filename11, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X11.append(value[0])
    Y11 = 100-max(X11)

    filename12='./res/7076_results.csv'
    X12 = []
    with open(filename12, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X12.append(value[0])
    Y12 = 100-max(X12)


    labels = ['no_defense', 'mkrum', 'bulyan', 'trmean']
    x = np.arange(len(labels))

    width = 0.25
    plt.xticks(x, labels=labels)
    iid = [Y1-Y0, Y3-Y03, Y5-Y05, Y7-Y07]
    non_iid_2_class = [Y2-Y02, Y4-Y04, Y6-Y06, Y8-Y08]
    non_iid_2_class_m = [Y9-Y02, Y10-Y04, Y11-Y06, Y12-Y08]
    print(non_iid_2_class)

    # non_iid_1_class = [Y2-Y01, Y5-Y01, Y8-Y01, Y11-Y01]
    plt.bar(x - width, iid, width, label='iid')
    plt.bar(x, non_iid_2_class, width, label='non-iid-2-wo')
    plt.bar(x + width, non_iid_2_class_m, width, label='non-iid-2-w')

    plt.title("ASR of LIE under Defense w/wo Distribution Reconstruction")
    plt.legend()
    plt.show()

def plt_none_noniid_defense():
    path00 = './res/61300_results.csv'
    filename00 = path00
    X00 = []
    with open(filename00, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X00.append(value[0])
    Y00 = 100 - max(X00)

    # path01 = './res/6213_results.csv'
    # filename01 = path01
    # X01 = []
    # with open(filename01, 'r') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         value = [float(s) for s in line.split(',')]
    #         X01.append(value[0])
    #
    # Y01 = 100 - max(X01)

    filename02 = './res/6214_results.csv'
    X02 = []
    with open(filename02, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X02.append(value[0])
    Y02 = 100 - max(X02)

    filename03 = './res/7120_results.csv'
    X03 = []
    with open(filename03, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X03.append(value[0])
    Y03 = 100 - max(X03)

    filename04 = './res/7122_results.csv'
    X04 = []
    with open(filename04, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X04.append(value[0])
    Y04 = 100 - max(X04)

    filename05 = './res/7190_results.csv'
    X05 = []
    with open(filename05, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X05.append(value[0])
    Y05 = 100 - max(X05)

    filename06 = './res/7191_results.csv'
    X06 = []
    with open(filename06, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X06.append(value[0])
    Y06 = 100 - max(X06)


    filename07 = './res/7193_results.csv'
    X07 = []
    with open(filename07, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X07.append(value[0])
    Y07 = 100 - max(X07)

    filename08 = './res/7192_results.csv'
    X08 = []
    with open(filename08, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X08.append(value[0])
    Y08 = 100 - max(X08)

    path0 = './res/6221_results.csv'
    filename0 = path0
    X0 = []
    with open(filename0, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X0.append(value[0])
    Y0 = 100 - max(X0)

    path1 = './res/6222_results.csv'
    filename1 = path1
    X1 = []
    with open(filename1, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X1.append(value[0])
    Y1 = 100-max(X1)

    path3 = './res/6231_results.csv'
    filename3 = path3
    X3 = []
    with open(filename3, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X3.append(value[0])
    Y3 = 100-max(X3)

    path4 = './res/6232_results.csv'
    filename4 = path4
    X4 = []
    with open(filename4, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X4.append(value[0])
    Y4 = 100-max(X4)

    path6 = './res/6234_results.csv'
    filename6 = path6
    X6 = []
    with open(filename6, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X6.append(value[0])
    Y6 = 100-max(X6)

    filename7 = './res/6235_results.csv'
    X7 = []
    with open(filename7, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X7.append(value[0])
    Y7 = 100-max(X7)

    filename9 = './res/6237_results.csv'
    X9 = []
    with open(filename9, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X9.append(value[0])
    Y9 = 100-max(X9)

    filename10 = './res/6238_results.csv'
    X10 = []
    with open(filename10, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split(',')]
            X10.append(value[0])
    Y10 = 100-max(X10)

    labels = ['no_defense', 'mkrum',  'bulyan', "trmean"]
    x = np.arange(len(labels))

    width = 0.25
    plt.xticks(x, labels=labels)
    iid = [Y00, Y03, Y05, Y07]
    non_iid_2_class = [Y02, Y04, Y06, Y08]
    plt.bar(x - width, iid, width, label='iid')
    plt.bar(x, non_iid_2_class, width, label='non-iid-2-class')

    plt.title("Error Rate of Noniid under Defense")
    plt.legend()
    plt.show()

def plt_fang_comp():
    plt.subplot(121)
    labels = ['krum',  "trmean",'median']
    x = np.arange(len(labels))

    width = 0.25
    plt.xticks(x, labels=labels)
    fang = [64,8,22]
    rep_fang = [20.5,1.8,1.7]
    plt.bar(x - width, fang, width, label='fang')
    plt.bar(x, rep_fang, width, label='rep_fang')

    plt.title("ASR updates unknown MNIST")

    plt.subplot(122)
    labels = ['krum',  "trmean",'median']
    x = np.arange(len(labels))

    width = 0.25
    plt.xticks(x, labels=labels)
    fang = [66, 17, 26]
    rep_fang = [17.4, 1.7, 1.5]
    plt.bar(x - width, fang, width, label='fang')
    plt.bar(x, rep_fang, width, label='rep_fang')

    plt.title("ASR updates known MNIST")

    plt.legend()
    plt.show()

def visualization_umap():
    from sklearn.datasets import load_digits
    import matplotlib.pyplot as plt
    import torch
    import numpy as np
    import seaborn as sns
    import pandas as pd


    dfa_g = torch.load('plt/DFA-g-15.pt').reshape([50, 28*28]).data
    dfa_r = torch.load('plt/DFA-r-15.pt').reshape([50, 28*28]).data
    # print(type(dfa_g))
    real = torch.load('/Users/huangjiyue/PycharmProjects/DFA/data/MNIST/processed/training.pt')
    real_0 = []
    for i in range(len(real[1])):
        if real[1][i] == 0 and len(real_0)<500:
            real_0.append(real[0][i])
    real_t = torch.stack(real_0[250:300]).reshape([50, 28*28]).data.float()

    data_g_r = torch.cat((dfa_g, dfa_r), 0).numpy()
    target = np.zeros([100, ], int)
    target[50:100] += 1
    # target[100:] += 2
    print(dfa_g.shape)
    print(real_t.shape)
    print(data_g_r.shape)

    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(data_g_r)
    print(embedding.shape)

    with sns.axes_style("darkgrid"):
        fig, ax = plt.subplots(1,1, figsize=(7,4))

    plt.scatter(embedding[:50, 0], embedding[:50, 1], c='#8ea0c5', cmap='Spectral', s=50, marker='^', label="ZKA-G")
    plt.scatter(embedding[50:100, 0], embedding[50:100, 1], c='#d7a384', cmap='Spectral', s=50, marker='o', label="ZKA-R")
    # plt.scatter(embedding[100:, 0], embedding[100:, 1], c='#BDB76B', cmap='Spectral', s=50, marker='*', label="Real-Y")

    plt.xticks([])
    plt.yticks([])

    plt.legend(loc='lower left')


    plt.show()

def visualization_stripplot():
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from pandas import Series,DataFrame

    with sns.axes_style("darkgrid"):
        f, ax = plt.subplots(2,1, figsize=(6,6))

    data_f = {"data":['real-data','ZKA-R','ZKA-G'],
            'mkrum':[5.81,	35.85,	21.58],
            'trmean':[28.6,	73.29,	37.43],
            'bulyan':[12.78,	13.66,	27.07],
            'median':[11.23,	24.39,	25.73]}

    iris_f = DataFrame(data_f)

    # "Melt" the dataset to "long-form" or "tidy" representation
    iris_f = pd.melt(iris_f, "data", var_name="defense")
    iris_f = iris_f.rename(columns={'value': 'attack success rate'})

    data_c = {"data":['real-data','ZKA-R','ZKA-G'],
            'mkrum':[34.68,	50.8,	51.2],
            'trmean':[71.62,	71.2,	75],
            'bulyan':[35.6,	55.6,	56.6],
            'median':[33.96,50.6,	52.4]}
    iris_c = DataFrame(data_c)
    iris_c = pd.melt(iris_c, "data", var_name="defense")
    iris_c = iris_c.rename(columns={'value': 'attack success rate'})


    # Initialize the figure
    sns.despine(bottom=True, left=True)

    # Show each observation with a scatterplot
    # sns.stripplot(x='attack success rate', y="defense", hue="data",
    #               data=iris, dodge=True, alpha=.4, zorder=1, s=10)

    # Show the conditional means, aligning each pointplot in the
    # center of the strips by adjusting the width allotted to each
    # category (.8 by default) by the number of hue levels
    sns.pointplot(ax=ax[0], x='attack success rate', y="defense", hue="data",
                  data=iris_f, dodge=.8 - .8 / 3,
                  join=False, palette="dark",
                  markers="d", scale=1, ci=None)

    # Improve the legend
    handles, labels = ax[0].get_legend_handles_labels()
    print(handles)
    print(labels)
    ax[0].legend(handles, labels,
              handletextpad=0, columnspacing=1,
              loc="lower right", ncol=3, frameon=True)


    sns.despine(bottom=True, left=True)
    sns.pointplot(ax=ax[1], x='attack success rate', y="defense", hue="data",
                  data=iris_c, dodge=.8 - .8 / 3,
                  join=False, palette="dark",
                  markers="d", scale=1, ci=None)

    # Improve the legend
    handles, labels = ax[1].get_legend_handles_labels()
    print(handles)
    print(labels)
    ax[1].legend(handles, labels,
              handletextpad=0, columnspacing=1,
              loc="upper right", ncol=3, frameon=True)

    plt.show()

if __name__ =='__main__':
    # plt_txt()
    # plt_acc()
    # plt_class_recall_1()
    # plt_utility()
    # plt_atk_bar()
    # plt_agg_comp()
    # plt_reverse()
    # plt_non_iid_100()
    # plt_batch_size()
    # plt_non_iid_defense()
    # plt_non_iid_distribution()
    # plt_none_noniid_defense()
    # plt_fang_comp()
    visualization_umap()



