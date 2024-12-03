import utils
import os
import time
import h5py as h5
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import argparse
import yaml 
import energyflow as ef


if __name__ == "__main__":


    parser = argparse.ArgumentParser()    
    parser.add_argument('--config', default='config.yaml', help='Config file with training parameters')
    parser.add_argument('--file_name', default='processed_data_background_rel.h5', help='File to load')
    parser.add_argument('--output_folder', default='/pscratch/sd/d/dimathan/LHCO/Data', help='Folder to save the output')
    parser.add_argument('--output_name', default='sampled_events.h5', help='Name of the output file')
    parser.add_argument('--npart', default=279, type=int, help='Maximum number of particles')
    parser.add_argument('--n_events', default = 1000000, type=int, help='How many events to load')
    parser.add_argument('--n_sample', default= 200000, type=int, help='How many events to sample')
    parser.add_argument('--data_path', default='/pscratch/sd/d/dimathan/LHCO/Data', help='Path containing the training files')


    flags = parser.parse_args()
    with open(flags.config, 'r') as stream:
        config = yaml.safe_load(stream)
      
    
    # load the DATA on the sidebands from the LHCO dataset
    with h5.File(os.path.join(flags.data_path,flags.file_name),"r") as h5f:
        nevts = min(flags.n_events, h5f['jet_data'][:].shape[0])          # number of events
        particles = h5f['constituents'][:nevts]
        jets = h5f['jet_data'][:nevts]
        mask = h5f['mask'][:nevts]
        particles = np.concatenate([particles,mask],-1)

    print('Loaded particles with shape:',particles.shape)

    p4_jets = ef.p4s_from_ptyphims(jets)
    # get mjj from p4_jets
    sum_p4 = p4_jets[:, 0] + p4_jets[:, 1]
    mjj = ef.ms_from_p4s(sum_p4)
    jets = np.concatenate([jets,np.sum(mask,-2)],-1)

    sr_events = np.sum((mjj >= 3300) & (mjj <= 3700) )
    sb_events = np.sum( ( (mjj < 3300) & (mjj > 2300 ) ) | ((mjj > 3700) & (mjj < 5000) ) ) 
    print(f'# of events in the signal region: {sr_events}/{len(mjj)}')
    print(f'# of events in the side band: {sb_events}/{len(mjj)}')
    print()


    mask_region = utils.get_mjj_mask(mjj,use_SR=False,mjjmin=2300,mjjmax=5000)
    mask_mass = (np.abs(jets[:,0,0])>0.0) & (np.abs(jets[:,1,0])>0.0)

    #particles = particles[(mask_region) & (mask_mass)]
    #mjj = mjj[(mask_region) & (mask_mass)]
    #jets = jets[(mask_region) & (mask_mass)]
    #jets[:,:,-1][jets[:,:,-1]<0] = 0.

    #sr_events = np.sum((mjj >= 3300) & (mjj <= 3700) )
    #sb_events = np.sum( ( (mjj < 3300) & (mjj > 2300 ) ) | ((mjj > 3700) & (mjj < 5000) ) ) 
    #print(f'# of events in the signal region: {sr_events}/{len(mjj)}')
    #print(f'# of events in the side band: {sb_events}/{len(mjj)}')
    #print()


    # Define the KernelDensity model
    t_start = time.time()
    kde = KernelDensity(kernel='gaussian', bandwidth=25)  # Adjust bandwidth as needed

    # Fit the model to sideband data
    kde.fit(mjj[:, None])

    # Generate m values across the full range
    m_values = np.linspace(2700, 4300, 20000)[:, None]

    # Compute the log density
    log_density = kde.score_samples(m_values)

    # Convert log density to density
    density = np.exp(log_density)
    print(f'KDE took {time.time() - t_start:.2f} seconds')


    plt.figure(figsize=(8, 6))
    plt.hist(mjj, bins=100, density=True, alpha=0.5, label='Sideband Data')
    plt.plot(m_values, density, label='Interpolated Density', color='red')
    plt.axvline(3300, color='green', linestyle='--', label='Signal Region Start')
    plt.axvline(3700, color='green', linestyle='--', label='Signal Region End')
    plt.xlabel('m')
    plt.ylabel('Density')
    plt.legend()
    plt.title('Kernel Density Estimation for Sideband and Signal Region')

    # save the plot as a pdf 
    plt.savefig('kde.pdf')

    # sample events in the SR based on the KDE
    sr_mask = (m_values[:, 0] >= 3300) & (m_values[:, 0] <= 3700)
    sr_values = m_values[sr_mask]
    sr_density = density[sr_mask]

    # Normalize the density in the SR
    sr_density /= np.trapz(sr_density, sr_values[:, 0])  # Normalize using the trapezoidal rule

    # Sample 10,000 events in the SR region
    sr_samples = np.random.choice(sr_values[:, 0], size=flags.n_sample, p=sr_density / sr_density.sum())

    # Restrict the plotting range to 3000 < m < 4000
    #plot_mask = (m_values[:, 0] >= 3100) & (m_values[:, 0] <= 3900)
    #plot_values = m_values[plot_mask]
    #plot_density = sr_density[plot_mask]

    # Plot KDE and SR samples
    plt.figure(figsize=(10, 6))
    plt.plot(sr_values, sr_density, label='KDE Density', color='red')
    plt.hist(sr_samples, bins=400, density=True, alpha=0.6, label='Sampled Events')
    plt.axvline(3300, color='green', linestyle='--', label='SR Start')
    plt.axvline(3700, color='green', linestyle='--', label='SR End')
    plt.xlim(3000, 4000)  # Restrict x-axis
    plt.xlabel('m')
    plt.ylabel('Density')
    plt.title(f'Sampling {flags.n_sample} Events in the Signal Region (Scaled and Restricted)')
    plt.legend()

    plt.savefig('sampled_events.pdf')

    # Save the sampled events to an h5 file
    with h5.File(os.path.join(flags.output_folder, flags.output_name), 'w') as h5f:
        h5f.create_dataset('mjj', data=sr_samples)
    
    print(f"Saved samples to {os.path.join(flags.output_folder, flags.output_name)}")

    