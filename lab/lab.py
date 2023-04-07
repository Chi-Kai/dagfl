import os
import random
import sys
import time

import numpy as np
import itertools
from zlib import crc32
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from tangle.lab.utils.test import test_img_local_all
from ..models.baseline_constants import MODEL_PARAMS, ACCURACY_KEY
from ..core import Tangle, Transaction, Node, MaliciousNode, PoisonType
from ..core.tip_selection import TipSelector
from .lab_transaction_store import LabTransactionStore
from .utils.train_utils import get_model, getdata, read_data

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, name=None):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.name = name

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        if self.name is None:
            image, label = self.dataset[self.idxs[item]]
        elif 'femnist' in self.name:
            image = torch.reshape(torch.tensor(self.dataset['x'][item]),(1,28,28))
            label = torch.tensor(self.dataset['y'][item])
        elif 'sent140' in self.name:
            image = self.dataset['x'][item]
            label = self.dataset['y'][item]
        else:
            image, label = self.dataset[self.idxs[item]]
        return image, label

class Lab:
    def __init__(self,tip_selector_factory, config, model_config, node_config, poisoning_config, tx_store=None):

        self.tip_selector_factory = tip_selector_factory
        self.config = config
        self.model_config = model_config
        self.node_config = node_config
        self.poisoning_config = poisoning_config
        self.tx_store = tx_store if tx_store is not None else LabTransactionStore(self.config.tangle_dir, self.config.src_tangle_dir)
        
        # 加载数据
        if 'cifar' in model_config.dataset or model_config.dataset == 'mnist':
            dataset_train, dataset_test, dict_users_train, dict_users_test = getdata(model_config)
            for idx in dict_users_train.keys():
                np.random.shuffle(dict_users_train[idx])
        else:
            if 'femnist' in model_config.dataset:
                train_path = model_config.datadir + '/train'
                test_path = model_config.datadir  + '/test'
            clients, groups, dataset_train, dataset_test = read_data(train_path, test_path)
            lens = []
            for iii, c in enumerate(clients):
                lens.append(len(dataset_train[c]['x']))
            dict_users_train = list(dataset_train.keys()) 
            dict_users_test = list(dataset_test.keys())
            for c in dataset_train.keys():
                dataset_train[c]['y'] = list(np.asarray(dataset_train[c]['y']).astype('int64'))
                dataset_test[c]['y'] = list(np.asarray(dataset_test[c]['y']).astype('int64'))
        self.train_data, self.test_data, self.train_dict, self.test_dict = dataset_train, dataset_test, dict_users_train, dict_users_test
        self.clients_model_params = {}

        # Set the random seed if provided (affects client sampling, and batching)
        random.seed(1 + config.seed)
        np.random.seed(12 + config.seed)

#这段代码定义了一个静态方法 create_client_model，它接受两个参数 seed 和 model_config。
#在方法体内，首先构造了一个模型的路径 model_path，通过 importlib.import_module 方法导入这个模型，并获取其中的 ClientModel 类。
#然后根据传入的 model_config 创建模型参数 model_params，并将其传给 ClientModel 构造方法，最终得到一个模型。
#其中还为模型设置了 num_epochs、batch_size 和 num_batches 等属性，并将构造完成的模型返回。
    @staticmethod
    def create_client_model(model_config):
        return get_model(model_config)
    
    #创建创世交易
    def create_genesis(self):

        client_model = self.create_client_model(self.model_config)

        # 主要用来存放每个客户端的head参数
        # generate list of local models for each user
        model_glob = client_model

        for user in range(self.model_config.num_users):
            model_local_dict = {}
            for key in model_glob.state_dict().keys():
                model_local_dict[key] = model_glob.state_dict()[key]
            self.clients_model_params[user] = model_local_dict 

        genesis = Transaction([])
        genesis.add_metadata('time', 0)
        # 这里每次保存前几层base层的参数
        self.tx_store.save(genesis, client_model.get_params())

        return genesis

    def create_node_transaction(self, tangle, round, client_id,seed, model_config, tip_selector, tx_store):

        client_model = Lab.create_client_model(model_config)
        # 加载每个客户端的表示层参数
        client_model.load_state_dict(self.clients_model_params[client_id])
        # Choose which nodes are malicious based on a hash, not based on a random variable
        # to have it consistent over the entire experiment run
        # https://stackoverflow.com/questions/40351791/how-to-hash-strings-into-a-float-in-01
        use_poisoning_node = \
            self.poisoning_config.poison_type != PoisonType.Disabled and \
            self.poisoning_config.poison_from <= round and \
            (float(crc32(client_id.encode('utf-8')) & 0xffffffff) / 2**32) < self.poisoning_config.poison_fraction
        # 如果节点是有毒节点，就创建一个 MaliciousNode 对象，并输出一条消息表示此节点已被毒害。
        # 否则，就创建一个普通的 Node 对象。
        if use_poisoning_node:
            ts = TipSelector(tangle, particle_settings=self.tip_selector_factory.particle_settings) \
                if self.poisoning_config.use_random_ts else tip_selector
            print(f'client {client_id} is is poisoned {"and uses random ts" if self.poisoning_config.use_random_ts else ""}')
            node = MaliciousNode(tangle, tx_store, ts, client_id,self.train_data,self.test_data, self.train_dict,self.test_dict,client_model, self.poisoning_config.poison_type, config=self.node_config)
        else:
            node = Node(tangle, tx_store, tip_selector, client_id,self.train_data, self.test_data, self.train_dict,self.test_dict,client_model, config=self.node_config,model_config=self.model_config)

        tx, tx_weights,base_params = node.create_transaction()
        # 使用round做时间戳
        if tx is not None:
            tx.add_metadata('time', round)
        
        # 更新每个客户端的表示层参数
        for k,key in enumerate(client_model.state_dict().keys()):
            self.clients_model_params[client_id][key] = tx_weights[key]

        return tx, tx_weights,base_params

    # 重载函数
    def create_node_transactions(self, tangle, round, clients):

        tip_selectors = [self.tip_selector_factory.create(tangle) for _ in range(len(clients))]
        result = [self.create_node_transaction(tangle, round, client_id,self.config.seed, self.model_config, tip_selector, self.tx_store)
                  for ((client_id), tip_selector) in zip(clients, tip_selectors)]
        
        for tx, tx_weights,base_p in result:
            if tx is not None:
                self.tx_store.save(tx, base_p)

        return [tx for tx, _ ,_ in result]
    

    def create_malicious_transaction(self):
        pass

    def init_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed=seed)
        torch.manual_seed(seed=seed)
        torch.cuda.manual_seed(seed=seed)
        torch.cuda.manual_seed_all(seed=seed)

    # 训练
    def train(self,start_from_round, num_rounds, eval_every):

        self.init_seed(1)
        if num_rounds == -1:
            rounds_iter = itertools.count(start_from_round)
        else:
            rounds_iter = range(start_from_round, num_rounds)
        # 如果 start_from_round > 0，就从上一轮的 tangle 中加载数据。
        if start_from_round > 0:
            tangle_name = int(start_from_round)-1
            print('Loading previous tangle from round %s' % tangle_name)
            tangle = self.tx_store.load_tangle(tangle_name)

        accs = [] 

        for round in rounds_iter:
            begin = time.time()
            print('Started training for round %s' % round)
            sys.stdout.flush()
            # 如果是第一轮，就创建创世交易。作为基础权重      
            if round == 0:
                genesis = self.create_genesis()
                tangle = Tangle({genesis.id: genesis}, genesis.id)
            else:

                m = max(int(self.model_config.frac * self.model_config.num_users), 1)
                clients = random.sample(range(self.model_config.num_users), k=m)       

                print(f"Clients this round: {clients}")

                for tx in self.create_node_transactions(tangle, round, clients):
                    if tx is not None:
                        tangle.add_transaction(tx)

            print(f'This round took: {time.time() - begin}s')
            
            acc_test, loss_test = test_img_local_all(self.create_client_model( self.model_config),self.model_config, self.test_data, self.test_dict,
                                                        w_locals=self.clients_model_params)
            accs.append(acc_test)
            print('Round {:3d}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                        round, loss_test, acc_test))

            sys.stdout.flush()
            self.tx_store.save_tangle(tangle, round)
        
        accs_dir = './save/'
        accs_csv = './save/acc_'+self.model_config.model+'_'+self.model_config.dataset+'.csv' 
        if not os.path.exists(accs_dir):
            os.makedirs(accs_dir)
        accs = np.array(accs)
        df = pd.DataFrame({'round': range(len(accs)), 'acc': accs})
        df.to_csv(accs_csv, index=False)
           # if eval_every != -1 and round % eval_every == 0:
           #    self.print_validation_results(self.validate(round, dataset, eval_on_fraction), round)    
