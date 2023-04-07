import numpy as np
import torchvision.transforms as transforms
import random

def partition_data(dataset,train_label,test_label,partition, n_parties, beta=0.4):
    
    # 获取数据数目
    n_train = len(train_label)
    n_test  = len(test_label)

    train_label = np.array(train_label)
    test_label  = np.array(test_label)

    #IID 设置
    if partition == "homo":
        # 根据标签数目随机出一个表示数据点的arr
        idxs_train = np.random.permutation(n_train)
        idxs_test  = np.random.permutation(n_test)
        # 平均划分给每个client
        batch_idxs_train = np.array_split(idxs_train, n_parties)
        batch_idxs_test = np.array_split(idxs_test, n_parties)
        # 分配
        train_dataidx_map = {i: batch_idxs_train[i] for i in range(n_parties)}
        test_dataidx_map  = {i: batch_idxs_test[i]  for i in range(n_parties)}

    # Dirichlet 标签分布
    elif partition == "noniid-labeldir":
        min_size_train = 0
        min_size_test  = 0
        min_size = 0
        min_require_size = 10
        K = 10
        # 数据集的标签种类
        if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            K = 2
            # min_require_size = 100
        if dataset == 'cifar100':
            K = 100
        elif dataset == 'tinyimagenet':
            K = 200


        train_dataidx_map = {}
        test_dataidx_map = {}

        # 分别对train/test划分
        while min_size_train < min_require_size:
           train_idx_batch = [[] for _ in range(n_parties)]
           #每种标签 
           for k in range(K):
                #选出每种label的数据点
                idx_k_train = np.where(train_label == k)[0]
                np.random.shuffle(idx_k_train)
                # dir分布，每个client分配到label的比例,每个client都是beta
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                # 调整，调整每个客户端分配到标签数据集的比例，使得每个客户端的数据集大小都不超过n_train/n_parties 
                # 超过的会变为0,正常的还是原来的分布系数     
                p_train = np.array([p * (len(idx_j) < n_train / n_parties) for p, idx_j in zip(proportions, train_idx_batch)])
                # 归一化为一个概率分布。
                p_train = p_train / p_train.sum()
                # 得到每个client按照dir分布划分的k标签资源数目
                # 每个分布系数等于之前所有系数之和
                p_train = (np.cumsum(p_train) * len(idx_k_train)).astype(int)[:-1]
                # 将k的划分资源点加入到client总的划分。
                train_idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(train_idx_batch, np.split(idx_k_train, p_train))]
                # 选出最少的数据大小
                min_size_train = min([len(idx_j) for idx_j in train_idx_batch])

        while min_size_test < min_require_size:
           test_idx_batch = [[] for _ in range(n_parties)]
           #每种标签 
           for k in range(K):
                #选出每种label的数据点
                idx_k_test  = np.where(test_label  == k)[0]
                np.random.shuffle(idx_k_test)
                # dir分布，每个client分配到label的比例,每个client都是beta
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                # 调整，调整每个客户端分配到标签数据集的比例，使得每个客户端的数据集大小都不超过n_train/n_parties 
                # 超过的会变为0,正常的还是原来的分布系数     
                p_test  = np.array([p * (len(idx_j) < n_test / n_parties) for p, idx_j in zip(proportions, test_idx_batch)])
                # 归一化为一个概率分布。
                p_test  = p_test  / p_test.sum()
                # 得到每个client按照dir分布划分的k标签资源数目
                # 每个分布系数等于之前所有系数之和
                p_test  = (np.cumsum(p_test) * len(idx_k_test)).astype(int)[:-1]
                # 将k的划分资源点加入到client总的划分。
                test_idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(test_idx_batch, np.split(idx_k_test, p_test))]
                # 选出最少的数据大小
                min_size_test  = min([len(idx_j) for idx_j in test_idx_batch])

        for j in range(n_parties):
            np.random.shuffle(train_idx_batch[j])
            np.random.shuffle(test_idx_batch[j])
            train_dataidx_map[j] = train_idx_batch[j]
            test_dataidx_map[j] = test_idx_batch[j]

    
    elif partition > "noniid-#label0" and partition <= "noniid-#label9":
        # 获取每个client的划分标签数
        num = eval(partition[13:])
        if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            num = 1
            K = 2
        else:
            K = 10
        if dataset == "cifar100":
            K = 100
        elif dataset == "tinyimagenet":
            K = 200
        # num = 10 时每个client分到十种标签数据
        # Q: 对于K > 10 的怎么办？ 只使用十种吗
        if num == 10:
            train_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(n_parties)}
            test_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(n_parties)}
            for i in range(10):
                # 获取每种标签对应的数据点
                idx_k_train = np.where(train_label==i)[0]
                idx_k_test  = np.where(test_label==i)[0]
                np.random.shuffle(idx_k_train)
                np.random.shuffle(idx_k_test)
                # 将每种标签代表的数据划分到每个client
                split_train = np.array_split(idx_k_train,n_parties)
                split_test = np.array_split(idx_k_test,n_parties)
                # 按照client编号划分
                for j in range(n_parties):
                    train_dataidx_map[j]=np.append(train_dataidx_map[j],split_train[j])
                    test_dataidx_map[j] = np.append(test_dataidx_map[j],split_test[j])
        # 少于10
        else:
            # times 记录所有种类标签被分配的次数
            times=[0 for i in range(K)]
            # contain 记录所有client分配label的情况
            contain=[]
            # 对每个client划分不同的label
            for i in range(n_parties):
                # 第一个标签是i%k
                current=[i%K]
                times[i%K]+=1
                # 已经选择的标签
                selected_label=1
                # 随机选择一个没有选过的标签加入，直到选择num个
                while (selected_label<num):
                    ind=random.randint(0,K-1)
                    if (ind not in current):
                        selected_label=selected_label+1
                        current.append(ind)
                        times[ind]+=1
                # 将第i个client选择的label加入
                contain.append(current)

            train_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(n_parties)}
            test_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(n_parties)}

            for i in range(K):
                idx_k_train = np.where(train_label==i)[0]
                idx_k_test  = np.where(test_label==i)[0]
                np.random.shuffle(idx_k_train)
                np.random.shuffle(idx_k_test)
                # 根据被client选择的次数划分
                split_train = np.array_split(idx_k_train,times[i])
                split_test  = np.array_split(idx_k_test,times[i])
                ids=0
                # 根据前面每个client分配的label划分数据
                for j in range(n_parties):
                    if i in contain[j]:
                        train_dataidx_map[j]=np.append(train_dataidx_map[j],split_train[ids])
                        test_dataidx_map[j]=np.append(test_dataidx_map[j],split_test[ids])
                        ids+=1

    # 数量偏移
    elif partition == "iid-diff-quantity":
        # 随机生成数据点
        train_idxs = np.random.permutation(n_train)
        test_idxs = np.random.permutation(n_test)
        train_min_size = 0
        test_min_size = 0
        # 每次生成随机分布，知道每个分布都大于10
        # 分两次筛选，尽量保持最初状态，避免一部分划分过大
        while train_min_size < 10:
            # 这个生成分布式是随机的?
            p_train = np.random.dirichlet(np.repeat(beta, n_parties))
            p_train = p_train/p_train.sum()
            train_min_size = np.min(p_train*len(train_idxs))

        while test_min_size < 10:
            # 这个生成分布式是随机的?
            p_test = np.random.dirichlet(np.repeat(beta, n_parties))
            p_test = p_test/p_test.sum()
            test_min_size = np.min(p_test*len(test_idxs))
        p_train = (np.cumsum(p_train)*len(train_idxs)).astype(int)[:-1]
        p_test = (np.cumsum(p_test)*len(test_idxs)).astype(int)[:-1]
        train_batch_idxs = np.split(train_idxs,p_train)
        test_batch_idxs = np.split(test_idxs,p_test)
        train_dataidx_map = {i: train_batch_idxs[i] for i in range(n_parties)}
        test_dataidx_map = {i: test_batch_idxs[i] for i in range(n_parties)}
    
    return train_dataidx_map,test_dataidx_map



