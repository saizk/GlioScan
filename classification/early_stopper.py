# import torch
# from collections.abc import Callable
# from functools import cached_property


class EarlyStopper:
    # mode_dict = {"min": torch.lt, "max": torch.gt}

    def __init__(self, stopping_threshold, patience):
        # self.monitor = monitor
        self.stopping_threshold = stopping_threshold
        self.patience = patience
        # self.mode = mode

        self.counter = 0
        self.best_validation_loss = float('inf')

    # @cached_property
    # def monitor_op(self) -> Callable:
    #     return self.mode_dict[self.mode]

    def on_validation_end(self, loss):
        should_stop = False
        if loss <= self.stopping_threshold:
            print("Desired validation loss reached!")
            should_stop = True

        # If you also want to incorporate early stopping:
        if loss < self.best_validation_loss:
            self.best_validation_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping!")
                should_stop = True

        return should_stop
