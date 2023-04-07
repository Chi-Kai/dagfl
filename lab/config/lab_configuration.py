class LabConfiguration:
    seed: int
    tangle_dir: str
    src_tangle_dir: str

    def __init__(self):
        self.seed = None
        self.tangle_dir = None

    def define_args(self, parser):
        parser.add_argument('--seed',
                    help='seed for random client sampling and batch splitting',
                    type=int,
                    default=0)

        parser.add_argument('--tangle-dir',
                    help='dir for tangle data (DAG JSON)',
                    type=str,
                    default='tangle_data',
                    required=False)
        parser.add_argument('--src-tangle-dir',
                    help='dir to load initial tangle data from (DAG JSON)',
                    type=str,
                    default=None,
                    required=False)

    def parse(self, args):
        self.seed = args.seed
        self.tangle_dir = args.tangle_dir
        self.src_tangle_dir = args.src_tangle_dir
