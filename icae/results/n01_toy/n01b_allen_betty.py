#%%
from icae.results.n01_toy.n01_generate_data import *

from icae import interactive_setup as interactive

print("Generating Allen (training set):")
gen = toy.Generator(distances_to_DOM, angles_to_DOM, photons_per_event)
base_size = int(5e6)
noise_factor = 0.01
interactive.save_value("noise factor in training set",noise_factor)

#%%
gen.generate_valid(base_size)
#%%
# plot examples for noise
print("Gaussian reshape noise:")

example_params = {"d": 25, "η": 0.1}
valid = gen.pdf_valid(**example_params)
means = np.linspace(0.2, 0.8, num=5)
stds = np.linspace(0.1, 0.8, num=5)
impacts = np.linspace(0.1, 1, num=5)
count_variations = len(means) * len(stds) * len(impacts)

for impact in tqdm(impacts, "Plotting examples"):
    for std in stds:
        for mean in means:
            times, pdf = gen.pdf_gaussian_reshaped(
                **example_params, mean=mean, std=std, impact=impact,
            )
            plt.plot(times, pdf, label=f"{mean:.1}")

        plt.fill_between(*valid, alpha=0.5, label="valid")

        plt.figtext(0, 0, f"peak size: {impact:.1e}, std: {std:.1e}")
        plt.legend(title="peak position")
        plt.tight_layout()
        plt.savefig(
            config.root
            + "icae/results/01-example-plots/gauss-"
            + f"{impact:.2},{std:.1},{mean:.1}.png"
        )
        plt.clf()
# %%
batch_size = int(base_size * noise_factor / 2 / count_variations)
assert batch_size > 0

gen.tqdm = False
for impact in tqdm(
    impacts, f"Generating {batch_size}*{count_variations} noise examples…"
):
    for std in tqdm(stds):
        for mean in means:
            gen.generate_gaussian_reshaped(batch_size, mean, std, impact)


# %%
# plot examples for noise
print("Double peak noise:")

example_params = {"d": 25, "η": 0.1}
valid = gen.pdf_valid(**example_params)
shifts = np.linspace(0.01, 0.4, num=10) * 4e-7
impacts = np.array(list(np.linspace(0.1, 0.5, num=8)) + [1, 2])
count_variations = len(shifts) * len(impacts)

for impact in tqdm(impacts, "Plotting examples", disable=True):
    for shift in shifts:
        times, pdf = gen.pdf_double_pulse(
            **example_params, shift=shift, photon_count_difference=impact,
        )
        plt.plot(times, pdf, label=f"{shift:.1}")

    plt.fill_between(*valid, alpha=0.3, label="valid")

    plt.figtext(0, 0, f"peak size: {impact:.1e}")
    plt.legend(title="separation")
    plt.tight_layout()
    # plt.show()
    plt.savefig(
        config.root
        + "icae/results/01-example-plots/peak-"
        + f"{impact:.2},{shift:.1}.png"
    )
    plt.clf()

# %%
batch_size = int(base_size * noise_factor / 2 / count_variations)
assert batch_size > 0

gen.tqdm = False
for impact in tqdm(
    impacts, f"Generating {batch_size}*{count_variations} noise examples…"
):
    for shift in shifts:
        gen.generate_double_pulse(batch_size, shift, impact)

# %%
gen.save(out_file, "train/")
del gen
#%%
print("Generating Betty (validation set):")
gen = toy.Generator(distances_to_DOM, angles_to_DOM, photons_per_event)
base_size = int(1e4)
noise_factor = 0.5

gen.generate_valid(base_size)

batch_size = int(base_size * noise_factor / 2 / count_variations)
assert batch_size > 0

gen.tqdm = False
for impact in tqdm(
    impacts, f"Generating {batch_size}*{count_variations} noise examples…"
):
    for std in tqdm(stds):
        for mean in means:
            gen.generate_gaussian_reshaped(batch_size, mean, std, impact)

batch_size = int(base_size * noise_factor / 2 / count_variations)
assert batch_size > 0

gen.tqdm = False
for impact in tqdm(
    impacts, f"Generating {batch_size}*{count_variations} noise examples…"
):
    for shift in shifts:
        gen.generate_double_pulse(batch_size, shift, impact)

gen.save(out_file, "val/")
del gen
# %%
print("Finished. Testing generated file:")
import pandas as pd

df = pd.read_hdf(out_file, "train/parameters")
print(f"Produced and saved {df['count'].sum()} training waveforms.")

df = pd.read_hdf(out_file, "val/parameters")
print(f"Produced and saved {df['count'].sum()} validation waveforms.")
# %%
df = pd.read_hdf(out_file, "train/waveforms")
print(f"Data takes {df.values.nbytes/1e9:.3e} GB of RAM")

print("Script finished.")
