import globals


class ClassificationDistillationOptions(object):

    def __init__(self, train_steps, num_targets, distill_n_examples, optimizer_lr, distill_lr,
                 distill_n_examples_for_target=1, device=globals.device):
        self.device = device
        self.train_steps = train_steps
        self.num_targets = num_targets
        self.num_of_distilled_examples = distill_n_examples
        self.distill_n_exmamples_for_target = distill_n_examples_for_target
        self.optim_lr = optimizer_lr
        self.distill_lr = distill_lr


class RegressionDistillationOptions(object):
    pass
