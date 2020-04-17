#%%
from icae.results.n01_toy.n01_generate_data import *

print("Generating Conrad")
gen = toy.Generator(distances_to_DOM, angles_to_DOM, photons_per_event)
base_size = int(1e4)
#%%
gen.generate_valid(base_size)
# %%
# plot examples for noise
print("Double peak noise:")

example_params = {"d": 25, "η": 0.1}
valid = gen.pdf_valid(**example_params)
shifts = np.linspace(0.005, 0.15, num=30) * 4e-7
impacts = [0.5]  # np.array(list(np.linspace(0.1, 0.5, num=8)) + [1, 2])
count_variations = len(shifts) * len(impacts)

for impact in tqdm(impacts, "Plotting examples", disable=True):
    for shift in shifts:
        times, pdf = gen.pdf_double_pulse(
            **example_params, shift=shift, photon_count_difference=impact,
        )
        plt.plot(times, pdf, label=f"{shift:.1}")

    plt.fill_between(*valid, alpha=0.3, label="valid")

    plt.figtext(0, 0, f"peak size: {impact:.1e}")
    # plt.legend(title="separation")
    plt.xlabel("time in s")
    plt.ylabel("PDF")
    plt.tight_layout()
    plt.show()
    plt.savefig(
        config.root
        + "icae/results/01-example-plots/conrad_peak-"
        + f"{impact:.2},{shift:.1}.png"
    )
    plt.clf()

# %%
batch_size = int(base_size)
assert batch_size > 0

gen.tqdm = False
for impact in impacts:
    for shift in tqdm(
        shifts, f"Generating {batch_size}*{count_variations} noise examples…"
    ):
        gen.generate_double_pulse(batch_size, shift, impact)

# %%
gen.save(out_file, "only_peaks/")


# %%
