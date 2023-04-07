class RunConfiguration:

    def define_args(self, parser):

        parser.add_argument('--num-rounds',
                        help='number of rounds to simulate;',
                        type=int,
                        default=-1)
        parser.add_argument('--eval-every',
                        help='evaluate every ____ rounds;',
                        type=int,
                        default=-1)
        parser.add_argument('--start-from-round',
                        help='at which round to start/resume training',
                        type=int,
                        default=0)
        parser.add_argument('--target-accuracy',
                        help='stop training after reaching this test accuracy',
                        type=float,
                        default=1,
                        required=False)

    def parse(self, args):
        self.num_rounds = args.num_rounds
        self.eval_every = args.eval_every
        self.start_from_round = args.start_from_round
        self.target_accuracy = args.target_accuracy
