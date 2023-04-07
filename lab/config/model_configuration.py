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
        parser.add_argument('--frac', 
                            type=float, 
                            default=0.1, 
                            help="the fraction of clients: C")
        parser.add_argument('--local_ep', 
                            type=int, 
                            default=11, 
                            help="the number of local epochs: E")
        parser.add_argument('--local_updates', 
                            type=int, 
                            default=1000000, 
                            help="maximum number of local updates")
        parser.add_argument('--local_rep_ep', 
                            type=int, 
                            default=1,
                            help="the number of local epochs for the representation for FedRep")
        parser.add_argument('--local_bs', 
                            type=int, 
                            default=10, 
                            help="local batch size: B")
        parser.add_argument('--num_classes', 
                            type=int, 
                            default=10, 
                            help="number of classes")

    def parse(self, args):
        self.dataset = args.dataset
        self.datadir = args.datadir
        self.model = args.model
        self.lr = args.lr
        self.partition = args.partition
        self.beta = args.beta
        self.num_users = args.num_users
        self.frac = args.frac
        self.local_bs = args.local_bs
        self.local_ep = args.local_ep
        self.local_updates = args.local_updates
        self.local_rep_ep = args.local_rep_ep
        self.num_classes = args.num_classes