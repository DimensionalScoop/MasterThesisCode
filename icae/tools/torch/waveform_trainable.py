import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import ray
import ray.tune as tune
from ray.tune import Trainable
import os
import numpy as np
from tqdm import tqdm
from ray.tune.utils import pin_in_object_store, get_pinned_object
import sklearn.metrics as metrics

import icae.models.waveform.flexible as models
import icae.tools.loss.EMD as EMD
from icae.tools.hyperparam import mappings

class MCTrainable(tune.Trainable):
    def _setup(self, config):
        # controls how often workers report model performance
        name = config.get('model_factory',None)
        self.model_factory = mappings.models[name]
        self.model = self.model_factory(config)
        
        self.batches_per_step = config.get('batches_per_step',1)
        
        self.max_validation_steps = config.get('max_validation_steps',10)
        
        # GPU or CPU?
        use_cuda = config.get("use_gpu") and torch.cuda.is_available()
        print("CUDA:",use_cuda)
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model.to(self.device)

        self.batch_size = config.get("batch_size", None)

        self.verbose = config.get("verbose", False)

        # Abstract: implement self.model

        optimizer = config.get("optimizer", "Adam")
        if optimizer == "SGD":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=config.get("lr", 0.01),
                momentum=config.get("momentum", 0.9),
            )
        elif optimizer == "Adam":
            self.optimizer = optim.Adam(self.model.parameters())
        else:
            raise NotImplemented()
        
        self.validation_loss_F = lambda p, t: EMD.torch_auto(p, t, mean=False)
        
        loss_name = config.get("training_loss", "mse_loss")
        self.train_loss_F = mappings.losses[loss_name]
        

    def _train(self):
        training_loss = self._train_batches(self.batches_per_step)
        auc, loss_mean = self._validation_loss()
        return {
            "auc": auc,
            "val_loss": loss_mean,
            "training_loss": np.mean(training_loss),
        }

    def _train_batches(self, train_for_n_batches=1):
        losses = []
        self.model.train()

        count = tqdm(
            range(train_for_n_batches), desc="Training", disable=not self.verbose, leave=True
        )

        data_train = get_pinned_object(self.get_train_pin())
        for data, i in zip(data_train, count):
            if self.batch_size and self.batch_size < data['data'].shape[0]:
                data = data['data'][:self.batch_size].to(self.device)
            else:
                data = data['data'].to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(data)
            loss = self.train_loss_F(pred, data)
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss))
            
        return losses


    def _validation_loss(self):
        self.model.eval()
        
        val_loss = []
        truths = []
        with torch.no_grad():
            data_val = get_pinned_object(self.get_val_pin())
            for data,_ in zip(data_val, range(self.max_validation_steps)):
                wf = data['data'].to(self.device)
                truth = (data['MC_type']>0)
                pred = self.model(wf)
                val_loss.append(self.validation_loss_F(pred, wf).cpu().numpy())
                truths.append(np.array(truth))

        try:
            pred = np.hstack(np.asarray(val_loss))
            truth = np.hstack(truths)
            fpr, tpr, _ = metrics.roc_curve(truth, pred)
            auc = metrics.auc(fpr, tpr)
        except ValueError:
            return 0
        
        return auc, np.mean(pred)
        

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model_.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))
