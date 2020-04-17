#%%
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from tqdm import tqdm
import pickle
import uuid
from box import Box
from glob import glob
from scipy.stats.stats import pearsonr
from scipy.stats import stats
import torch

import icae.results.n01_toy.n03_mode_training as training
from icae.tools.torch.gym import Gym
import icae.tools.loss.EMD as EMD
import icae.interactive_setup as interactive
from icae.models.waveform.simple import ConvAE
from icae.tools.analysis import calc_auc, plot_auc, TrainingStability

plt.set_plot_path(__file__)
interactive.set_saved_value_yaml_root_by_filename(__file__)

#%%
def val_loss_and_auc(model:Gym):
    loss_func = lambda p, t: EMD.torch_auto(p, t, mean=False)
    val_losses = np.hstack(model.validation_loss(loss_func))  # restack batches
    names = model.data_val.dataset.df["MC_name"].values
    truth = (names != "valid").astype("int")
    pred = val_losses
    fpr, tpr, _ = metrics.roc_curve(truth, pred)
    auc = metrics.auc(fpr, tpr)
    return val_losses, auc



r = TrainingStability("Ribbles_w_outliers_1e6",None)
r.load()

interactive.save_value("trained models used for stability estimates",len(r.model))
#%%
loss_val = r.loss_val.mean(axis=1)
cut = np.quantile(r.loss_train,0.8)#np.mean(loss_val) #+ np.std(loss_val)*0.5
#cut2 = np.mean(r.auc) + np.std(loss_val)*0.5
filter = (r.loss_train < cut) #& (r.auc>cut2)
interactive.save_value("excluded values from stability plot",f"{100*sum(~filter)/len(filter):0.1f}")

plt.hexbin(loss_val[filter], r.auc[filter], linewidths=0.2, gridsize=int(np.sqrt(len(r.auc[filter]))))
plt.xlabel("EMD validation loss")
plt.ylabel("AUC")
plt.colorbar(label="count")
plt.show_and_save(f"va training stability on {len(r.auc)} samples")

# %%
plt.hexbin(r.loss_train[filter], r.auc[filter],linewidths=0.2, gridsize=int(np.sqrt(len(r.auc[filter]))))
plt.xlabel("EMD training loss")
plt.ylabel("AUC")
plt.colorbar(label="count")
plt.show_and_save(f"ta training stability on {len(r.auc)} samples")

#%%
#cut = #np.quantile(r.loss_train,0.8)
filter = (r.loss_train < 0.08)#cut)
cut_percentage = 1-len(r.loss_train[filter])/len(r.loss_train)
interactive.save_value("underperformance percentage",cut_percentage*100,".1f")

plt.subplot(2,1,1)
plt.hist([np.array(r.auc[filter]),np.array(r.auc[~filter])], bins=60,histtype='stepfilled')
plt.ylabel("count")
plt.xlabel("AUC")
plt.gca().invert_xaxis()
plt.subplot(2,1,2)
plt.hist([r.loss_train[filter],r.loss_train[~filter]],bins=60,histtype='stepfilled')
#plt.hist(r.loss_train[filter], bins=15, color='C0')
#plt.hist(r.loss_train[~filter], bins=2, color='C1')
plt.xlabel("EMD training loss")
plt.ylabel("count")
plt.show_and_save("loss vs AUC")

#%%
r_pear, p_value = pearsonr(r.auc, r.loss_train)
interactive.save_value("all data Pearson loss-AUC correlation",f"{r_pear:.2f}")
interactive.save_value("all data Pearson loss-AUC p-value",p_value,".1e")

r_pear, p_value = pearsonr(r.auc[filter], r.loss_train[filter])
interactive.save_value("filtered best data Pearson train loss-AUC correlation",f"{r_pear:.2f}")
interactive.save_value("filtered best data Pearson train loss-AUC p-value",p_value,".3f")

r_pear, p_value = pearsonr(r.auc[filter], r.loss_val[filter].mean(axis=1))
interactive.save_value("filtered best data Pearson val loss-AUC correlation",f"{r_pear:.2f}")
interactive.save_value("filtered best data Pearson val loss-AUC p-value",p_value,".3f")

r_pear, p_value = pearsonr(r.loss_train[filter], r.loss_val[filter].mean(axis=1))
interactive.save_value("filtered best data Pearson train-val correlation",f"{r_pear:.2f}")
interactive.save_value("filtered best data Pearson train-val p-value",p_value,".1e")

def most_likely_value(data):
    hist, edges = np.histogram(data,bins=int(np.sqrt(len(data))))
    bin_width = edges[1] - edges[0]
    bin_centers = (edges[:-1] + edges[1:])/2
    most_likely_value = bin_centers[np.argmax(hist)]
    chance_of_below_maximum_likelihood_value = hist[:np.argmax(hist)].sum()/hist.sum()
    return most_likely_value, chance_of_below_maximum_likelihood_value

most_likely_auc, chance_of_below_maximum_likelihood = most_likely_value(r.auc[filter])

interactive.save_value("filtered best data most likely AUC",most_likely_auc,".2f")
interactive.save_value("filtered best data chance of below most likely AUC",chance_of_below_maximum_likelihood,".2f")

plt.show_and_save(f"loss vs AUC on {len(r.auc)} samples")

# %%

# %%
best_loss = np.argmin(r.loss_train)
best_auc = np.argmax(r.auc)
interactive.save_value("best toy loss", r.loss_train[best_loss],'.3f')
interactive.save_value("auc of best toy loss",r.auc[best_loss],'.2f')
interactive.save_value("best toy auc",r.auc[best_auc],'.2f')
interactive.save_value("worst toy auc",r.auc.min(),'.2f')
interactive.save_value("mean auc",np.mean(r.auc),'.2f')
interactive.save_value("std auc",np.std(r.auc,ddof=1),'.2f')
interactive.save_value("median auc",np.median(r.auc),'.2f')

def cross_validate(func, input, sample_size, shuffle=False, repeat=1):
    results = []
    for i in range(repeat):
        if shuffle:
            input = input.copy() # shuffeling is in-place
            np.random.shuffle(input)
        mini_datasets = np.array_split(input, int(len(input)/sample_size))
    results.extend([func(d) for d in mini_datasets])
    return results
    

def auc_of_lowest_loss(input):
    loss, auc = input.T
    #sort = np.argsort(loss)[::-1]
    #not_the_worst_loss = np.where(loss<loss.mean() + loss.std()*0.1)
    #optimal_loss = sort[not_the_worst_loss][0]
    #return auc[optimal_loss]
    return auc[np.argmin(loss)]

def auc_of_median_loss(input):
    loss, auc = input.T
    sort = np.argsort(loss)
    median = sort[len(sort)//2]
    #not_the_worst_loss = np.where(loss<loss.mean() + loss.std()*0.1)
    #optimal_loss = sort[not_the_worst_loss][0]
    #return auc[optimal_loss]
    return auc[median]

sample_size = 5
input = np.vstack([r.loss_val.mean(axis=1),r.auc]).T # shape: [samples, [loss, auc]]
aucs = cross_validate(auc_of_lowest_loss,input,sample_size)
interactive.save_value(f"expected mean auc for {sample_size} tries with val",np.mean(aucs),'.2f')
interactive.save_value(f"expected std auc for {sample_size} tires with val",np.std(aucs,ddof=1),'.2f')

# sample_size = 400
# input = np.vstack([r.loss_train[filter],r.auc[filter]]).T # shape: [samples, [loss, auc]]
# aucs = cross_validate(auc_of_median_loss,input,sample_size,shuffle=True,repeat=10000)
# most_likely, chance_worse = most_likely_value(aucs)
# interactive.save_value(f"most likely auc for {sample_size} tries with train",most_likely,'.2f')
# interactive.save_value(f"chance of worse AUC for {sample_size} tires with train",chance_worse,'.2f')

#%%
input = np.vstack([r.loss_train[filter],r.auc[filter]]).T # shape: [samples, [loss, auc]]
sample_sizes = np.asarray(np.arange(1,350,7).tolist() + [350])
tries = 1000
chance_worse_list = []
chance_worse_std_list = []
likely_aucs_list = []
for s in tqdm(sample_sizes):
    worse = []
    likely_aucs = []
    for t in range(tries):
        aucs = cross_validate(auc_of_median_loss,input,s,shuffle=True)
        most_likely, chance_worse = most_likely_value(aucs)
        worse.append(chance_worse)
        likely_aucs.append(most_likely)
    likely_aucs_list.append(np.mean(likely_aucs))
    chance_worse_list.append(np.mean(worse))
    chance_worse_std_list.append(np.std(worse,ddof=1))#/np.sqrt(len(worse)))

#%%
chance_worse_std_list = np.array(chance_worse_std_list)
chance_worse_std_list = np.array(chance_worse_std_list)
plt.plot(sample_sizes,chance_worse_list, label="$p_{\mathrm{worse}}$")
plt.fill_between(sample_sizes,chance_worse_list+chance_worse_std_list,chance_worse_list-chance_worse_std_list, label="$\sqrt{\mathrm{Var}[p_{\mathrm{worse}}]}$",alpha=0.3)

plt.plot(sample_sizes,likely_aucs_list,label="$\mathrm{AUC}_{\mathrm{mode}}$")
plt.xlabel("ensemble size")
plt.legend()
plt.show_and_save("chance of worse AUC")

#%%
interactive.save_value(f"most likely auc sample size per point",tries,'.0e')
interactive.save_value(f"most likely auc for {sample_sizes[-1]} tries",likely_aucs_list[-1],'.2f')
interactive.save_value(f"expected chance of worse AUC for {sample_sizes[-1]} tires",chance_worse_list[-1],'.2f')
interactive.save_value(f"std chance of worse AUC for {sample_sizes[-1]} tires",chance_worse_std_list[-1],'.2f')
#%%
# sample_size = 5
# input = np.vstack([r.loss_val,r.auc]).T # shape: [samples, [loss, auc]]
# aucs = cross_validate(auc_of_median_loss,input,sample_size)
# interactive.save_value(f"expected mean auc for {sample_size} tries with train",np.mean(aucs),'.2f')
# interactive.save_value(f"expected std auc for {sample_size} tires with train",np.std(aucs,ddof=1),'.2f')

sample_size = 20
aucs = cross_validate(auc_of_median_loss,input,sample_size)
interactive.save_value(f"expected mean auc for {sample_size} tries with median train",np.mean(aucs),'.2f')
interactive.save_value(f"expected std auc for {sample_size} tires with median train",np.std(aucs,ddof=1),'.2f')

sample_size = 20
input = np.vstack([r.loss_val.mean(axis=1),r.auc]).T # shape: [samples, [loss, auc]]
aucs = cross_validate(auc_of_lowest_loss,input,sample_size)
interactive.save_value(f"expected mean auc for {sample_size} tries",np.mean(aucs),'.2f')
interactive.save_value(f"expected std auc for {sample_size} tires",np.std(aucs,ddof=1),'.2f')

# %%
best_loss = np.argsort(r.loss_train)[len(r.loss_train)//2]

model = training.best_model_factory()
model.model.load_state_dict(r.model[best_loss].to_dict())
model.model.eval()

latent_space = []
types = []

for d in tqdm(model.data_val):
    data = d['data']
    type = d['MC_type']
    data = data.to(model.device)
    latent = model.model.encode(data).detach().cpu().numpy()
    
    latent_space.append(latent)
    types.append(type.numpy())
latent_space = np.vstack(latent_space)
types = np.hstack(types)

##%
#%%
cov = np.cov(latent_space.T)
import numpy.linalg as lin
eigvalues,eigvectors = lin.eig(cov)

def rotate(list_of_vectors):
    return np.dot(eigvectors.T,list_of_vectors.T).T

valid = types == 0
ls = rotate(latent_space)

# plt.figure(figsize=(5.78,3.57*2))

for axis in range(3):
    
    plt.subplot(3,1,axis+1)
    plt.hist([ls[valid][:,axis],ls[~valid][:,axis]] ,histtype='step', label=['valid','outlier'], bins=100, density=True)

    plt.xlabel(f"latent space dim {axis+1}")
    plt.ylabel("frequency")
    if axis==0:plt.legend(loc="left")
plt.show_and_save(f"latent space comparisons")

#%%
ls_AUCs = []
for i in range(3):
    loss = ls[:,i] #np.abs(np.mean(ls[:,2]) - ls[:,2])
    plt.hist(loss,bins=int(np.sqrt(len(loss))))
    plt.show_and_save("None")
    plt.clf()
    plot_auc(loss,types!=0)
    plt.show_and_save(f"ROC of {i} latent space distance")
    ls_AUCs.append(calc_auc(loss,types!=0))

interactive.save_value("AUC of best latent space distance",max(ls_AUCs),".2f")
#plot_auc(val_losses,types!=0)
#plt.show_and_save("None")

# %%
plt.scatter(latent_space[:,0],latent_space[:,2],c=(types==0),marker='.')
plt.legend()
plt.plot()
plt.clf()

# %%
from mpl_toolkits.mplot3d import Axes3D



fig = plt.figure(1)
fig.clf()

for i in range(4):
    ax = fig.add_subplot(2, 2, 1+i, projection='3d')
    ax.azim=i*360/5

    color = types == 0
    for c in [0,1]:
        filter = color==c
        label = "outlier" if c==0 else "valid"
        data = latent_space[filter]
        ax.scatter(*rotate(data).T,label=label,alpha=0.1+0.9*c)
    plt.legend()
    plt.xlabel("latent dim 1")
    plt.ylabel("latent dim 2")
    ax.set_zlabel("latent dim 3")
plt.show_and_save("latent space 3D")

#%%
fig = plt.figure(1)
fig.clf()
ax = fig.add_subplot(1, 2, 1, projection='3d')

color = types == 0
for c in [0]:
    filter = color==c
    label = "outlier" if c==0 else "valid"
    data = latent_space[filter]
    ax.scatter(*rotate(data).T,label=label,alpha=0.1)
plt.title("outlier")
plt.xlabel("latent dim 1")
plt.ylabel("latent dim 2")
ax.set_zlabel("latent dim 3")

plt.xlim(-0.6,1.5)
plt.ylim(-1.25,-0.4)
ax.set_zlim(-0.1,0.7)

ax = fig.add_subplot(1, 2, 2, projection='3d')
for c in [1]:
    filter = color==c
    label = "outlier" if c==0 else "valid"
    data = latent_space[filter]
    ax.scatter(*rotate(data).T,label=label,alpha=0.1)
plt.title("normal")
plt.xlabel("latent dim 1")
plt.ylabel("latent dim 2")
ax.set_zlabel("latent dim 3")


plt.xlim(-0.6,1.5)
plt.ylim(-1.25,-0.4)
ax.set_zlim(-0.1,0.7)

plt.show_and_save("latent space 3D sep",tight_layout=False)

#%%
fig = plt.figure(1)
fig.clf()
ax = fig.add_subplot(1, 1, 1, projection='3d')

color = types == 0
for c in [0,1]:
    filter = color==c
    label = "outlier" if c==0 else "valid"
    data = latent_space[filter]
    ax.scatter(*rotate(data).T,label=label,alpha=0.1)
plt.legend()
plt.xlabel("latent dim 1")
plt.ylabel("latent dim 2")
ax.set_zlabel("latent dim 3")

plt.show_and_save("latent space 3D comb")
# %%
def ks_test(observation_pdf, pdf):
    #observ_cdf = np.cumsum(observation_pdf)
    #cdf = np.cumsum(pdf)
    ks_stat, p_value = stats.ks_2samp(observation_pdf.reshape(-1),pdf.reshape(-1))
    return p_value
    #return np.max(np.abs(observ_cdf-cdf))

all_ks = []
losses_val = []
val_classes = []
for d in tqdm(model.data_val):
    data = d['data']
    type = d['MC_type']
    data = data.to(model.device)
    pred = model.model(data).detach()
    losses_val.append(EMD.torch_auto(pred,data,False).cpu().numpy())
    pred = pred.cpu().numpy()
    data = data.cpu().numpy()
    
    val_classes.append(type)
    ks = [ks_test(data[i], pred[i]) for i in range(len(pred))]
    all_ks.append(ks)
ks = np.hstack(all_ks)
losses_val = np.hstack(losses_val)
val_classes = np.hstack(val_classes)

#%%
calc_auc(losses_val,val_classes!=0)
#%%
interactive.save_value("AUC for KS as metric", calc_auc(-ks,val_classes!=0),".2f")
# %%
plt.hist(np.log(ks),bins=int(np.sqrt(len(ks))));
plt.xlim([-60,0])
#plt.xscale('log')
plt.show_and_save("None")
plt.hist(losses_val,bins=int(np.sqrt(len(losses_val))))
plt.show_and_save("None")
plt.clf()
#%%
plt.hist2d(np.log(ks),losses_val,bins=100);
plt.xlim([-20,0])
plt.ylim([0,0.02])
plt.show_and_save("None")
# %%
plt.hist([ks,ks[types==0],ks[types!=0]],bins=int(np.sqrt(len(ks))), density=True,histtype='step',label=["all","valid","outlier"]);
plt.legend()
plt.show_and_save("None")

fpr, tpr, _ = metrics.roc_curve(types!=0, ks)
auc = metrics.auc(fpr, tpr)
print(auc)

# %%
loss_func = lambda p, t: EMD.torch_auto(p, t, mean=False)
val_losses = np.hstack(model.validation_loss(loss_func))

plt.hist([val_losses,val_losses[types==0],val_losses[types!=0]],bins=int(np.sqrt(len(val_losses))), density=True,histtype='step',label=["all","valid","outlier"]);
plt.legend()
plt.show_and_save("None")



# %%
