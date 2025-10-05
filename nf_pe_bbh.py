import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

from tqdm import tqdm
import multiprocessing as mp
import bilby
import numpy as np
from pycbc.waveform import get_td_waveform, taper_timeseries
from gwmat import point_lens
import matplotlib.pyplot as plt
import gwmat
import os
import sys
from pycbc.detector.ground import Detector
import pycbc
import bilby
from bilby.gw.prior import BBHPriorDict

import corner
from modules.gw_utils import scale_signal
import torch
from torch.utils.data import Dataset, DataLoader

from gwtorch.modules.gw_utils import inject_noise_with_target_SNR

from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from nflows.nn.nets import ResidualNet
from nflows.transforms import CompositeTransform, RandomPermutation

from modules.pp_plot_code import calculate_pp_values, compute_pp_statistics, plot_overlay_pp_plot

from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif')
rc('font', serif='times')
rc('mathtext', default='sf')
rc("lines", markeredgewidth=1)
rc("lines", linewidth=1)
rc('axes', labelsize=20)
rc("axes", linewidth=0.5)
rc('xtick', labelsize=10)
rc('ytick', labelsize=10)
rc('legend', fontsize=10)
rc('ytick', right=True, direction='in')
rc('xtick', top=True, direction='in')
rc('xtick.major', pad=15)
rc('ytick.major', pad=15)
rc('xtick.major', size=12)
rc('ytick.major', size=12)
rc('xtick.minor', size=7)
rc('ytick.minor', size=7)

num_samples = 6000
f_lower = 20.0       

priors = BBHPriorDict()

for key in ['a_1', 'a_2', 'tilt_1', 'tilt_2', 'theta_jn', 'phi_12', 'phi_jl', 'luminosity_distance', 'psi', 'phase']:
    priors.pop(key, None)

priors['mass_1'].minimum = 10
priors['mass_2'].minimum = 10
priors['chirp_mass'] = bilby.gw.prior.UniformInComponentsChirpMass(name='chirp_mass', minimum=10, maximum=100)
priors['mass_ratio'] = bilby.gw.prior.UniformInComponentsMassRatio(name='mass_ratio', minimum=0.1, maximum=1)
priors['spin1z'] = bilby.core.prior.Uniform(name='spin1z', minimum=-0.9, maximum=0.9)
priors['spin2z'] = bilby.core.prior.Uniform(name='spin2z', minimum=-0.9, maximum=0.9)
priors['coa_phase'] = bilby.core.prior.Uniform(name='coa_phase', minimum=0.0, maximum=2 * np.pi)
priors['polarization'] = bilby.core.prior.Uniform(minimum=0., maximum=np.pi, boundary="periodic")
priors['Log_Mlz'] = bilby.core.prior.Uniform(minimum=2, maximum=5)
priors['yl'] = bilby.core.prior.PowerLaw(alpha=1, minimum=0.01, maximum=1.0)

parameters_list = priors.sample(num_samples)

samples = [
    {key: parameters_list[key][i] for key in parameters_list}
    for i in range(num_samples)
]

print(f"Length of parameters_list: {len(samples)}")


def waveform(num):
    parameters = samples[num].copy()

    # Convert masses
    mass1, mass2 = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(
        parameters['chirp_mass'], parameters['mass_ratio']
    )

    m_lens = np.power(10., parameters.pop("Log_Mlz"))
    y_lens = parameters.pop("yl")

    # Generate unlensed waveform
    sp, sc = get_td_waveform(
        approximant='SEOBNRv4_opt',
        mass1=mass1,
        mass2=mass2,
        spin1z=parameters['spin1z'],
        spin2z=parameters['spin2z'],
        distance=300,
        delta_t=1.0 / 4096,
        coa_phase=parameters['coa_phase'],
        f_lower=f_lower,
    )

    # Frequency-domain lensing
    sp_freq = sp.to_frequencyseries(delta_f=sp.delta_f)
    sc_freq = sc.to_frequencyseries(delta_f=sc.delta_f)
    fs1 = sp_freq.sample_frequencies

    # Ffs_sp = np.vectorize(lambda f: gwmat.cythonized_point_lens.Ff_effective(f, ml=m_lens, y=y_lens))(fs1)
    Ffs = [gwmat.cythonized_point_lens.Ff_effective(f, ml=m_lens, y=y_lens) for f in fs1]
    time_Delay = point_lens.time_delay(ml=m_lens, y=y_lens)

    sp_lensed = pycbc.types.FrequencySeries(np.conj(Ffs) * np.asarray(sp_freq),
                                            delta_f=sp_freq.delta_f).cyclic_time_shift(-1 * (0.1 + time_Delay))
    sc_lensed = pycbc.types.FrequencySeries(np.conj(Ffs) * np.asarray(sc_freq),
                                            delta_f=sc_freq.delta_f).cyclic_time_shift(-1 * (0.1 + time_Delay))

    sp_lensed = sp_lensed.to_timeseries(delta_t=sp_lensed.delta_t)
    sc_lensed = sc_lensed.to_timeseries(delta_t=sc_lensed.delta_t)

    # Detector projection
    detector = Detector('H1')
    lensed_signal = detector.project_wave(
        sp_lensed, sc_lensed,
        ra=parameters['ra'], dec=parameters['dec'], polarization=parameters['polarization']
    )
    lensed_signal = taper_timeseries(lensed_signal, tapermethod="TAPER_STARTEND", return_lal=False)

    # Inject noise with target SNR
    target_snr = np.random.randint(15, 100)
    lensed_noisy, _, distance, _ = inject_noise_with_target_SNR(
        lensed_signal, parameters, mass1, mass2, m_lens, y_lens,
        target_snr, num, gw_signal_type='lensed',
        peak_window=(2.0, 2.2), detector=detector
    )

    # Update local copy of parameters
    parameters['distance'] = distance
    parameters['Log_Mlz'] = np.log10(m_lens)
    parameters['yl'] = y_lens

    lensed_noisy = lensed_noisy.crop(left=24, right=0)

    return np.array(lensed_noisy), parameters


def simulator(num):
    waveform_array, updated_params = waveform(num)
    return updated_params, waveform_array

def simulate_one(ii):
    theta_val, y_val = simulator(ii)
    return theta_val, y_val


num_simulations = num_samples

if __name__ == "__main__":
    with mp.Pool(processes=mp.cpu_count(), maxtasksperchild=500) as pool:
        results = pool.map(simulate_one, range(num_simulations))

    # Unzip results
    updated_samples, data_vals = zip(*results)
    updated_samples = list(updated_samples)
    data_vals = list(data_vals)

    # Replace old samples with updated ones
    samples = updated_samples

    theta_vals, data_vals = zip(*results)

    theta_vals = [list(d.values()) for d in theta_vals]

# convert to torch tensors
theta_vals = torch.from_numpy(np.array(theta_vals)).to(torch.float32)
data_vals = torch.from_numpy(np.array(data_vals)).to(torch.float32)

# create dataset
class DataGenerator(Dataset):
    def __len__(self):
        return num_simulations

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return theta_vals[idx], data_vals[idx]

dataset = DataGenerator()

# create dataloaders - 80/10/10 split
train_set_size = int(0.8 * num_simulations)
val_set_size = int(0.1 * num_simulations)
test_set_size = int(0.1 * num_simulations)

train_data, val_data, test_data = torch.utils.data.random_split(
    dataset, [train_set_size, val_set_size, test_set_size])

train_data_loader = DataLoader(
    train_data, batch_size=256,
    shuffle=True
)

val_data_loader = DataLoader(
    val_data, batch_size=256,
    shuffle=True
)

test_data_loader = DataLoader(
    test_data, batch_size=1,
    shuffle=False
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

num_transforms = 24
num_blocks = 24
hidden_features = 50

context_features = data_vals[0].shape[0]
parameter_space_dim = len(samples[0].keys())

base_dist = StandardNormal([parameter_space_dim])  

transforms = []

for _ in range(num_transforms):
    block = [
        MaskedAffineAutoregressiveTransform(
                features=parameter_space_dim, 
                hidden_features=hidden_features,
                context_features=context_features,
                num_blocks=num_blocks,
                activation=torch.tanh,
                use_batch_norm=True,
                use_residual_blocks=False,
        ),
        RandomPermutation(features=parameter_space_dim)
    ]
    transforms += block

transform = CompositeTransform(transforms)

flow = Flow(transform, base_dist).to(device)

optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

for i in range(200):
    flow.train()
    train_loss_total = 0.0

    train_loader = tqdm(train_data_loader, desc=f"Epoch {i+1} [Train]", leave=False)
    for idx, val in enumerate(train_loader):
        theta, data = val
        theta = theta.to(device)
        data = data.to(device)

        optimizer.zero_grad()
        loss = -flow.log_prob(theta, context=data).mean()
        loss.backward()
        optimizer.step()
        
        train_loss_total += loss.item()
        train_loader.set_postfix(loss=loss.item())

    scheduler.step()
    train_loss_avg = train_loss_total / len(train_data_loader)

    flow.eval()
    val_loss_total = 0.0
    with torch.no_grad():
        val_loader = tqdm(val_data_loader, desc=f"Epoch {i+1} [Val]", leave=False)
        for idx, val in enumerate(val_loader):
            theta, data = val
            theta = theta.to(device)
            data = data.to(device)

            val_loss_batch = -flow.log_prob(theta, context=data).mean()
            val_loss_total += val_loss_batch.item()
            val_loader.set_postfix(loss=val_loss_batch.item())

    val_loss_avg = val_loss_total / len(val_data_loader)

    if i == 0 or (i+1) % 10 == 0:
        print(f"[Epoch {i+1}] Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss_avg:.4f}")

torch.save(flow.state_dict(), "flow_bbh.pth")

theta_test, data_test = next(iter(test_data_loader))
theta_test = theta_test.to(device)
data_test = data_test.to(device)
with torch.no_grad():
        posterior_samples = flow.sample(5000, context=data_test.reshape(tuple(data_test.shape)))

theta_test = theta_test.squeeze(0)


data = posterior_samples.squeeze(0).cpu().numpy()

figure = corner.corner(
    data,
    bins=30,  # Number of bins
    labels=list(samples[0].keys()),
    quantiles=[0.16, 0.5, 0.84], 
    show_titles=True,
    title_kwargs={"fontsize": 12},
    truths=theta_test.cpu().numpy(),
    truth_color="red",
)

def set_tick_sizes_corner(fig, major=12, minor=7):
    for ax in fig.get_axes():
        for line in ax.xaxis.get_ticklines() + ax.yaxis.get_ticklines():
            line.set_markersize(major)
        for tick in ax.xaxis.get_minor_ticks() + ax.yaxis.get_minor_ticks():
            tick.tick1line.set_markersize(minor)
            tick.tick2line.set_markersize(minor)

set_tick_sizes_corner(figure)

figure.savefig("Injection_PE.pdf", bbox_inches='tight')
plt.show()

# flow.load_state_dict(torch.load('trained_flow_model.pth'))

pp_values, parameter_names = calculate_pp_values(
    flow, test_data_loader, samples[0], device, num_posterior_samples=5000
)

print("Generating overlaid P-P plot...")
plot_overlay_pp_plot(pp_values, parameter_names, confidence_level=0.95)

compute_pp_statistics(pp_values, parameter_names)

np.save('pp_values.npy', pp_values)
np.save('parameter_names.npy', parameter_names)

print("\nP-P analysis completed!")