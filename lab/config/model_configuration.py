class ModelConfiguration:
    dataset: str
    model: str
    lr: float
    num_epochs: int
    batch_size: int

    def __init__(self):
        self.dataset = None
        self.model = None
        self.lr = None
        self.num_epochs = None
        self.batch_size = None

    def define_args(self, parser):
        DATASETS = ['sent140', 'cifar10','cifar100', 'femnist', 'femnistclustered', 'shakespeare', 'celeba', 'synthetic', 'reddit', 'nextcharacter', 'poets', 'fake', 'synthetic_fedprox']

        parser.add_argument('-dataset',
                        help='name of dataset;',
                        type=str,
                        choices=DATASETS,
                        required=True) 
        parser.add_argument('-datadir',
                        help='dir of dataset;',
                        type=str,
                        default='./data',
                        required=False)
        parser.add_argument('-model',
                        help='name of model;',
                        type=str,
                        required=True)
        parser.add_argument('-lr',
                        help='learning rate for local optimizers;',
                        type=float,
                        default=0.01,
                        required=False)

        # Minibatch doesn't support num_epochs, so make them mutually exclusive
        epoch_capability_group = parser.add_mutually_exclusive_group()
        epoch_capability_group.add_argument('--minibatch',
                        help='None for FedAvg, else fraction;',
                        type=float,
                        default=None)
        epoch_capability_group.add_argument('--num-epochs',
                        help='number of epochs when clients train on data;',
                        type=int,
                        default=1)

        parser.add_argument('--batch-size',
                        help='batch size when clients train on data;',
                        type=int,
                        default=10)

        parser.add_argument('--num-batches',
                        help='num batches when clients train on data;',
                        type=int,
                        default=100)

        parser.add_argument('--limit-num-clients-in-dataset',
                        help='use fewer clients than contained in dataset;',
                        type=int,
                        default=0)
        
        parser.add_argument('--partition', 
                            type=str, 
                            default='noniid-#label2', 
                            help='the data partitioning strategy')
        parser.add_argument('--beta', 
                            type=float, 
                            default=0.5, 
                            help='The parameter for the dirichlet distribution for data partitioning')
        parser.add_argument('--num_users', 
                            type=int, 
                            default=100, 
                            help="number of users: n")
        
        parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
        parser.add_argument('--local_ep', type=int, default=11, help="the number of local epochs: E")
        parser.add_argument('--local_updates', type=int, default=1000000, help="maximum number of local updates")
        parser.add_argument('--local_rep_ep', type=int, default=1,
                        help="the number of local epochs for the representation for FedRep")
        parser.add_argument('--alg', type=str, default='fedrep', help='FL algorithm to use')

    def parse(self, args):
        self.dataset = args.dataset
        self.datadir = args.datadir
        self.model = args.model
        self.lr = args.lr
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.num_batches = args.num_batches
        self.limit_clients = args.limit_num_clients_in_dataset
        self.partition = args.partition
        self.beta = args.beta
        self.num_users = args.num_users
        self.frac = args.frac
        self.local_bs = args.batch_size
        self.local_ep = args.local_ep
        self.local_updates = args.local_updates
        self.local_rep_ep = args.local_rep_ep