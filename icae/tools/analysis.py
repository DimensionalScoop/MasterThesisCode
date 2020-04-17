import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from tqdm import tqdm
import pickle
from box import Box
import uuid
from glob import glob

import icae.interactive_setup as interactive
from icae.tools.config_loader import config
from icae.tools.loss import EMD
#from icae.tools import performance


# TODO: Rename calc_auc to auc_calc
def calc_auc(pred, truth):
    fpr, tpr, _ = metrics.roc_curve(truth, pred)
    auc = metrics.auc(fpr, tpr)
    return max(auc, 1 - auc)


def plot_auc(pred, truth):
    fpr, tpr, _ = metrics.roc_curve(truth, pred)
    auc = metrics.auc(fpr, tpr)

    lw = 2
    plt.plot(
        fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % auc
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")


def smoothed_loss(losses):
    return np.mean(losses[int(len(losses) * 0.9) :])


def _forever():
    while True:
        yield None


class StabilityRunner:
    """Run a task for an arbitrary amount of time and save results to pickles. Load when needed."""

    def __init__(self, name, num_workers=1, catch_ctrl_c=False):
        self.name = name
        self.save_dir = (
            config.root + config.analysis.stability_test_pickle_dir + name + "/"
        )
        interactive.mkdir_p(self.save_dir)
        self.runs = []
        self.num_workers = num_workers
        self.catch_ctrl_c = catch_ctrl_c

    def _check_keys(self, item):
        forbidden_keys = self.__dict__.keys()
        for fk in forbidden_keys:
            if fk in item.keys():
                raise ValueError(
                    "Forbidden key {fk} found. Check that generate_item defines the right names."
                )

    def generate_item(self, item):
        raise NotImplementedError()

    def run(self, iterations=-1):
        def task(_=None):
            item = Box()
            item = self.generate_item(item)
            self._check_keys(item)

            unique_filename = str(uuid.uuid4())
            pickle.dump(item, open(self.save_dir + unique_filename + ".pickle", "bw"))

        try:
            if iterations == -1:
                iterator = _forever()
            else:
                iterator = range(iterations)
            for _ in tqdm(iterator, "Generating samples (press Ctrl+C to stop)",leave=True):
                task()
                #if self.num_workers > 1:
                #    performance.parallize(
                #        task, range(self.num_workers * 10)
                #    )  # ,self.num_workers)
                #else:

        except KeyboardInterrupt:
            if self.catch_ctrl_c:
                return
            else:
                raise

    def load(self, unpack_keys_as_attributes=True):
        runs = []
        for f in tqdm(glob(self.save_dir + "*.pickle"), "Loading " + self.name):
            runs.append(pickle.load(open(f, "rb")))

        if not unpack_keys_as_attributes:
            self.runs = runs
            return
        
        keys = set()
        for r in runs:
            for k in r.keys():
                keys.add(k)

        for k in keys:
            self.__dict__.update({k: np.array([r.get(k, None) for r in runs])})

        to_remove = []
        for k in keys:
            to_remove.extend(
                [i for i, x in enumerate(self.__dict__[k].tolist()) if x == None]
            )
        to_remove = set(to_remove)
        if len(to_remove) > 0:
            for k in keys:
                self.__dict__[k] = self.__dict__[k][~np.array(list(to_remove))]
            print(f"Removed {len(to_remove)} bad files.")

        self.runs = runs


class TrainingStability(StabilityRunner):
    def __init__(
        self,
        name,
        gym_factory,
        auc_classes=None,
        batches=1000,
        val_loss_func=lambda p, t: EMD.torch_auto(p, t, mean=False),
        num_workers=1,
        add_attributes={},
        catch_ctrl_c=False,
    ):
        super().__init__(name, num_workers,catch_ctrl_c)
        self.gym_factory = gym_factory
        self.batches = batches
        self.val_loss_func = val_loss_func
        self.auc_classes = auc_classes
        self.add_attributes = add_attributes

    def generate_item(self, b):
        model = self.gym_factory()

        batches = self.batches
        b.losses_train = model.train_batches(batches)
        b.loss_train = smoothed_loss(b.losses_train)

        b.model = model.model.state_dict()
        b.count_training_waveforms = batches * model.data_train.batch_size

        b.loss_val = np.hstack(model.validation_loss(self.val_loss_func))
        if self.auc_classes is not None:
            if callable(self.auc_classes):
                truth = self.auc_classes(model)
            else:
                truth = self.auc_classes != 0
            b.auc = calc_auc(b.loss_val, truth)

        for i in self.add_attributes:
            b[i] = self.add_attributes[i]
        return b
