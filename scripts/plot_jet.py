import numpy as np
import yaml
import matplotlib.pyplot as plt
import argparse
import h5py as h5
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import utils
import tensorflow as tf
from GSGM_lhco import GSGM
import energyflow as ef

import sys
from plot_class import PlottingConfig

def get_mjj(particle,jet):
    # Recover the particle information
    
    new_p = np.copy(particle)
    new_p[:,:,:,0]*=np.expand_dims(jet[:,:,0],-1)
    new_p[:,:,:,1]+=np.expand_dims(jet[:,:,1],-1)
    new_p[:,:,:,2]+=np.expand_dims(jet[:,:,2],-1)
    
    #fix phi
    # new_p[:,:,:,2][new_p[:,:,:,2]>np.pi] -= 2*np.pi
    # new_p[:,:,:,2][new_p[:,:,:,2]<-np.pi] += 2*np.pi
    mask = np.expand_dims(new_p[:,:,:,0]!=0,-1)
    new_p*=mask
    #Flatten particles
    #new_p = np.reshape(new_p,(-1,new_p.shape[1]*new_p.shape[2],new_p.shape[-1]))

    new_p = ef.p4s_from_ptyphims(new_p)
    mjj = ef.ms_from_p4s(np.sum(new_p,(1,2)))
    
    return mjj
    
def plot(jet1,jet2,nplots,title,plot_folder):
    for ivar in range(nplots):
        config = PlottingConfig(title,ivar)
                    
        feed_dict = {
            'true':jet1[:,ivar],
            'gen': jet2[:,ivar]
        }
            

        fig,gs,_ = utils.HistRoutine(feed_dict,xlabel=config.var,
                                     #binning=config.binning,
                                     plot_ratio=True,
                                     reference_name='true',
                                     ylabel= 'Normalized entries',logy=config.logy)
        
        ax0 = plt.subplot(gs[0])     
        # ax0.set_ylim(top=config.max_y)
        if config.logy == False:
            yScalarFormatter = utils.ScalarFormatterClass(useMathText=True)
            yScalarFormatter.set_powerlimits((100,0))
            ax0.yaxis.set_major_formatter(yScalarFormatter)

        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
        fig.savefig('{}/{}_{}.pdf'.format(plot_folder,title,ivar),bbox_inches='tight')


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


    utils.SetStyle()


    parser = argparse.ArgumentParser()

    parser.add_argument('--data_folder', default='/pscratch/sd/d/dimathan/LHCO/Data', help='Folder containing data and MC files')    
    parser.add_argument('--plot_folder', default='../plots', help='Folder to save results')
    parser.add_argument('--file_name', default='processed_data_background_rel.h5', help='File to load')
    parser.add_argument('--npart', default=279, type=int, help='Maximum number of particles')
    parser.add_argument('--config', default='config.yaml', help='Training parameters')
    parser.add_argument('--nsplit', default=20,type = int, help='Number of batches to generate')
    
    parser.add_argument('--sample', action='store_true', default=False,help='Sample from the generative model')
    parser.add_argument('--test', action='store_true', default=False,help='Test if inverse transform returns original data')
    parser.add_argument('--SR', action='store_true', default=False,help='Load signal region background events')
    parser.add_argument('--hamb', action='store_true', default=False,help='Load hamburg team files')
    parser.add_argument('--large', action='store_true', default=False,help='Train with a large model')

    flags = parser.parse_args()
    with open(flags.config, 'r') as stream:
        config = yaml.safe_load(stream)
    
    # create the plot folder if it does not exist
    if not os.path.exists(flags.plot_folder):
        os.makedirs(flags.plot_folder)
        print(f'Created folder {flags.plot_folder}')

    model_name = config['MODEL_NAME']
    if flags.large:
        model_name+='_large'

    sample_name = model_name
    if flags.SR:
        sample_name += '_SR'
    if flags.hamb:
        sample_name += '_Hamburg'
    n_events_sample = config['n_events_sample']

    particles, jets, logmjj, _ = utils.DataLoader(flags.data_folder,
                                                 flags.file_name,
                                                 npart=flags.npart,
                                                 n_events = config['n_events'],
                                                 n_events_sample = n_events_sample, 
                                                 norm=config['NORM'],
                                                 make_tf_data=False,use_SR=flags.SR)
    

    print('After Loading')
    print(f'particles shape: {particles.shape}')
    print(f'jet shape: {jets.shape}')
    print() 

    if flags.test:
        particles_gen, jets_gen, mjj_gen = utils.SimpleLoader(flags.data_folder,
                                                              flags.file_name,
                                                              use_SR=flags.SR,
                                                              npart=flags.npart)
    else:        
        if flags.sample:            
            model = GSGM(config=config,npart=flags.npart)
            checkpoint_folder = '../checkpoints_{}/checkpoint'.format(model_name)
            model.load_weights('{}'.format(checkpoint_folder)).expect_partial()

            particles_gen = []
            jets_gen = []

            if flags.SR:
                logmjj_g = utils.LoadMjjFile(flags.data_folder,'generated_data_all_200k.h5',use_SR=flags.SR)[:n_events_sample]
            else: 
                logmjj_g = logmjj

            for i,split in enumerate(np.array_split(logmjj_g, flags.nsplit)):     # split the logmjj_g array into nsplit batches. Each GPU can only hold ~100 events
                if split.size == 0: break                                         # if split is empty  
                print(f'Generating split {i+1}/{flags.nsplit}')
                p,j = model.generate(split)
                particles_gen.append(p)
                jets_gen.append(j)
                p_aux, j_aux = utils.ReversePrep(p,j,mjj = utils.revert_mjj(split), npart=flags.npart, norm=config['NORM'])
                mjj_aux = get_mjj(p_aux,j_aux)
                # check how many events are in the signal region, i.e. 3300 < mjj < 3700
                sr_events = np.sum((mjj_aux >= 3300) & (mjj_aux <= 3700) )
                sb_events = np.sum( ( (mjj_aux < 3300) & (mjj_aux > 2300 ) ) | ((mjj_aux > 3700) & (mjj_aux < 5000) ) ) 
                print(f'# of events in the signal region: {sr_events}/{len(mjj_aux)}')
                print(f'# of events in the side band: {sb_events}/{len(mjj_aux)}')
                print()


            particles_gen = np.concatenate(particles_gen)
            jets_gen = np.concatenate(jets_gen)
            
            particles_gen, jets_gen = utils.ReversePrep(particles_gen,
                                                        jets_gen,
                                                        mjj = utils.revert_mjj(logmjj_g), # we pass logmjj_g in order to scale the generated particles and jets properly
                                                        npart=flags.npart,
                                                        norm=config['NORM'], )
            mjj_created = get_mjj(particles_gen, jets_gen) # this does not equal mjj_gen, since it is calculated from the generated particles and jets

            # WARNING: VM named it mjj_gen, but it is not the generated (created) mjj, it is the mjj that's used as a condition for the generation
            mjj_gen = utils.revert_mjj(logmjj_g)

            # keep only the events in the SB region
            only_SB = False  
            if only_SB:
                mask_region = utils.get_mjj_mask(mjj_created, flags.SR, mjjmin = config['MJJMIN'], mjjmax = config['MJJMAX'])
                passed = np.sum(mask_region)
                print('Keeping only SideBand Events')
                print(f'# of Generated Events In the SideBand: {passed}/{len(mask_region)}')
                particles_gen = particles_gen[mask_region]
                jets_gen = jets_gen[mask_region]
                mjj_created = mjj_created[mask_region]

            
            with h5.File(os.path.join(flags.data_folder,sample_name+'.h5'),"w") as h5f:
                dset = h5f.create_dataset("particle_features", data=particles_gen)
                dset = h5f.create_dataset("jet_features", data=jets_gen)
                dset = h5f.create_dataset("mjj", data=mjj_gen)
                
        elif flags.hamb:
            assert flags.SR, "ERROR: Hamburg files available only at SR"
            with h5.File(os.path.join(flags.data_folder,'generated_data_datacond_both_jets.h5'),"r") as h5f:
                particles_gen = np.stack([
                    h5f['particle_data_rel_x'][:],
                    h5f['particle_data_rel_y'][:]],1)

                jets_gen = np.stack([
                    h5f['jet_features_x'][:],
                    h5f['jet_features_y'][:]],1)

                mjj_gen = h5f['mjj'][:]
                
        else:  # if the sample flag is False, load from the generated file
            with h5.File(os.path.join(flags.data_folder,sample_name+'.h5'),"r") as h5f:
                jets_gen = h5f['jet_features'][:]
                particles_gen = h5f['particle_features'][:]
                mjj_gen = h5f['mjj'][:]
    
    print()
    print(f'particles shape: {particles.shape}')
    print(f'jet shape: {jets.shape}')


    particles, jets = utils.ReversePrep(particles,jets, mjj = utils.revert_mjj(logmjj),
                                       npart=flags.npart, norm=config['NORM'])


    feed_dict = {
        'true': get_mjj(particles, jets),               # equal to exp(logmjj)
        'gen':  get_mjj(particles_gen, jets_gen)        # close to exp(logmjj_g) if the model is good
    }                                                   # logmjj does not have to be equal to logmjj_g
    
    fig,gs,_ = utils.HistRoutine(feed_dict,xlabel="mjj GeV",
                                 binning=np.linspace(2800, 4200, 50),
                                 plot_ratio=True,
                                 reference_name='true',
                                 ylabel= 'Normalized entries',logy=True)
        
    fig.savefig('{}/mjj_{}.pdf'.format(flags.plot_folder,sample_name),bbox_inches='tight')

    # Flatten and plot other features
    
    jets = jets.reshape(-1,config['NUM_JET'])
    jets_gen = jets_gen.reshape(-1,config['NUM_JET'])
    title = 'jet' if flags.SR==False else 'jet_SR'
    if flags.hamb: title+='_Hamburg'
    
    plot(jets,jets_gen,title=title,
         nplots=config['NUM_JET'],plot_folder=flags.plot_folder)
    
        
    particles_gen=particles_gen.reshape((-1,config['NUM_FEAT']))
    mask_gen = particles_gen[:,0]!=0.
    particles_gen=particles_gen[mask_gen]
    particles=particles.reshape((-1,config['NUM_FEAT']))
    mask = particles[:,0]!=0.
    particles=particles[mask]
    title = 'part' if flags.SR==False else 'part_SR'
    if flags.hamb: title+='_Hamburg'
    
    plot(particles,particles_gen,
         title=title,
         nplots=config['NUM_FEAT'],
         plot_folder=flags.plot_folder)
