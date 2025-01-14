import numpy as np
import matplotlib.pyplot as plt
import argparse
import h5py as h5
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print()
import utils
import tensorflow as tf
from deepsets_cond import DeepSetsClass

import sys
import horovod.tensorflow.keras as hvd

from sklearn.metrics import roc_curve, auc
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras import  Input
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
import energyflow as ef

def recover_jet(jet,part):
    new_j = np.copy(jet)
    new_p = np.copy(part)
    new_p[:,:,:,0]*=np.expand_dims(jet[:,:,0],-1)
    new_p[:,:,:,1]+=np.expand_dims(jet[:,:,1],-1)
    new_p[:,:,:,2]+=np.expand_dims(jet[:,:,2],-1)

    #fix phi
    new_p[:,:,:,2] = np.clip(new_p[:,:,:,2],-np.pi,np.pi)
    # new_p[:,:,:,2][new_p[:,:,:,2]>np.pi] -= 2*np.pi
    # new_p[:,:,:,2][new_p[:,:,:,2]<-np.pi] += 2*np.pi
    
    mask = np.expand_dims(new_p[:,:,:,0]!=0,-1)
    new_p*=mask

    new_p = ef.p4s_from_ptyphims(new_p)
    jets = np.sum(new_p,2)
    jets = ef.ptyphims_from_p4s(jets)
    new_j[:,:,0] = jets[:,:,0]
    new_j[:,:,1] = ef.etas_from_p4s(np.sum(new_p,2))
    new_j[:,:,2] = np.clip(jets[:,:,2] - np.pi,-np.pi,np.pi)
    new_j[:,:,3] = jets[:,:,3]
    mjj = ef.ms_from_p4s(np.sum(new_p,(1,2)))
    
    return new_j.reshape((-1,jet.shape[-1])), mjj
    
def apply_mjj_cut(j,p,mjj,use_SR,mjjmin,mjjmax):
    mask = utils.get_mjj_mask(mjj,use_SR,mjjmin,mjjmax)
    return j[mask],p[mask],mjj[mask]

def combine_part_jet(jet,particle,mjj,npart,jet_from_cond=True):

    if jet_from_cond:
        new_j = np.copy(jet).reshape((-1,jet.shape[-1]))        
    else:
        new_j,mjj = recover_jet(jet,particle)

    
    new_p = np.copy(particle).reshape((-1,particle.shape[-1]))    
    mask = new_p[:,0]!=0

    #Apply the same transformations used during training
    mjj_tile = np.expand_dims(mjj,1)
    mjj_tile = np.reshape(np.tile(mjj_tile,(1,2)),(-1))
    new_j[:,0] = np.log(new_j[:,0]/mjj_tile)
    new_j[:,2] = np.clip(new_j[:,2] - np.pi,-np.pi,np.pi) #clip phi
    new_j[:,3] = np.ma.log(new_j[:,3]/mjj_tile).filled(0)
    new_p[:,0] = np.ma.log(1.0 - new_p[:,0]).filled(0)
    
    data_dict = utils.LoadJson('preprocessing_{}.json'.format(npart))        
    new_j = np.ma.divide(new_j-data_dict['mean_jet'],data_dict['std_jet']).filled(0)    
    new_p = np.ma.divide(new_p-data_dict['mean_particle'],data_dict['std_particle']).filled(0)
    
    new_p *=np.expand_dims(mask,-1)

    # print("Mean jet: {}, std jet: {}".format(np.mean(new_j,0),np.std(new_j,0)))
    # print("Mean particle: {}, std particle: {}".format(np.mean(new_p,0),np.std(new_p,0)))
    # input()
    #Reshape it back
    new_j = np.reshape(new_j,jet.shape)
    new_p = np.reshape(new_p,particle.shape)
    
    return new_j, new_p, mjj



def class_loader(data_path,
                 file_name,
                 npart,
                 use_SR=False,
                 nsig=15000,
                 nbkg=60671,
                 mjjmin=2300,
                 mjjmax=5000
                 
):

    if not use_SR:
        nsig = 0

    parts_bkg,jets_bkg,mjj_bkg = utils.SimpleLoader(data_path,file_name,
                                                    use_SR=use_SR,
                                                    npart=npart)
    
    
    #flatten particles
    parts_bkg = parts_bkg[hvd.rank():nbkg:hvd.size()]
    mjj_bkg = mjj_bkg[hvd.rank():nbkg:hvd.size()]
    jets_bkg = jets_bkg[hvd.rank():nbkg:hvd.size()]
    
    if nsig>0:
        parts_sig,jets_sig,mjj_sig = utils.SimpleLoader(data_path,
                                                        'processed_data_signal_rel.h5',
                                                        use_SR=use_SR,
                                                        npart=npart)
        
        parts_sig = parts_sig[hvd.rank():nsig:hvd.size()]
        mjj_sig = mjj_sig[hvd.rank():nsig:hvd.size()]
        jets_sig = jets_sig[hvd.rank():nsig:hvd.size()]
    
        labels = np.concatenate([np.zeros_like(mjj_bkg),np.ones_like(mjj_sig)])
        particles = np.concatenate([parts_bkg,parts_sig],0)
        jets = np.concatenate([jets_bkg,jets_sig],0)
        mjj = np.concatenate([mjj_bkg,mjj_sig],0)

    else:
        labels = np.zeros_like(mjj_bkg)
        particles = parts_bkg
        jets = jets_bkg
        mjj = mjj_bkg
    return jets,particles,mjj,labels

def compile(model,max_epoch,batch_size,learning_rate,nevts):
    
    lr_schedule = keras.experimental.CosineDecay(
        initial_learning_rate=learning_rate*hvd.size(),
        decay_steps=max_epoch*nevts/batch_size
    )
    
    opt = keras.optimizers.Adamax(learning_rate=lr_schedule)
    opt = hvd.DistributedOptimizer(
        opt, average_aggregated_gradients=True)


    model.compile(            
        optimizer=opt,
        #run_eagerly=True,
        loss="binary_crossentropy",
        experimental_run_tf_function=False,
        weighted_metrics=[])


def get_classifier(SR=False):
    #Define the model
    inputs_mask = Input((2,None,1))
    inputs_jet = Input((None,5))
    inputs_particle = Input((2,None,3))
    if SR:
        outputs = DeepSetsClass(
            inputs_jet,
            inputs_particle,
            num_heads = 2,
            num_transformer = 4,
            projection_dim = 64,
            mask = inputs_mask,
        )

        model = keras.Model(inputs=[inputs_jet,inputs_particle,inputs_mask],
                            outputs=outputs)
    else:
        inputs_cond = Input((1))
        outputs = DeepSetsClass(
            inputs_jet,
            inputs_particle,
            num_heads = 2,
            num_transformer = 4,
            projection_dim = 64,
            mask = inputs_mask,
            use_cond=True,
            cond_embedding = inputs_cond,
        )
        model = keras.Model(inputs=[inputs_jet,inputs_particle,inputs_mask,inputs_cond],
                            outputs=outputs)
    
    return model


if __name__ == "__main__":
    hvd.init()
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    utils.SetStyle()


    parser = argparse.ArgumentParser()

    parser.add_argument('--data_folder', default='/pscratch/sd/d/dimathan/LHCO/Data/', help='Folder containing data and MC files')    
    parser.add_argument('--plot_folder', default='../plots', help='Folder to save results')
    parser.add_argument('--file_name', default='processed_data_background_rel.h5', help='File to load')
    parser.add_argument('--test', action='store_true', default=False,help='Test if inverse transform returns original data')
    parser.add_argument('--npart', default=279, type=int, help='Maximum number of particles')
    parser.add_argument('--config', default='config.yaml', help='Training parameters')    
    parser.add_argument('--SR', action='store_true', default=False,help='Load signal region background events')
    parser.add_argument('--hamb', action='store_true', default=False,help='Load Hamburg team dataset')
    parser.add_argument('--reweight', action='store_true', default=False,help='Apply mjj based reweighting to SR events')
    parser.add_argument('--nsig', type=int,default=2500,help='Number of injected signal events')
    parser.add_argument('--nbkg', type=int,default=100000,help='Number of injected signal events')
    parser.add_argument('--nid', type=int,default=0,help='Independent training ID')
    parser.add_argument('--large', action='store_true', default=False,help='Train with a large model')
    parser.add_argument('--data_file', default='', help='File to load')
    
    parser.add_argument('--LR', type=float,default=1e-4,help='learning rate')
    parser.add_argument('--MAX-EPOCH', type=int,default=5,help='maximum number of epochs for the training')
    parser.add_argument('--BATCH-SIZE', type=int,default=128,help='Batch size')

    flags = parser.parse_args()
    config = utils.LoadJson(flags.config)
    MAX_EPOCH = flags.MAX_EPOCH
    BATCH_SIZE =flags.BATCH_SIZE
    LR = flags.LR

    # load the data from LHCO (with signal, that depends on nsig). The signal is labelled as 1, the background as 0
    data_j,data_p,data_mjj,labels = class_loader(flags.data_folder,
                                                 flags.file_name,
                                                 npart=flags.npart,
                                                 use_SR=flags.SR,
                                                 nsig = flags.nsig,
                                                 nbkg=flags.nbkg,
                                                 mjjmax=config['MJJMAX'],
                                                 mjjmin=config['MJJMIN']
                                                 )

    data_j,data_p,data_mjj = combine_part_jet(data_j,data_p,data_mjj,npart=flags.npart)
    sample_name = config['MODEL_NAME'] if flags.hamb==False else 'Hamburg'

    if flags.large:
        sample_name+='_large'

    if flags.test:sample_name = 'supervised'
    
    if flags.SR:
        sample_name += '_SR'
        

    if flags.test:
        bkg_p,bkg_j,bkg_mjj = utils.SimpleLoader(flags.data_folder,flags.file_name,
                                                      use_SR=flags.SR,npart=flags.npart)
        bkg_p = bkg_p[hvd.rank()::hvd.size()]
        bkg_j = bkg_j[hvd.rank()::hvd.size()]
        bkg_mjj = bkg_mjj[hvd.rank()::hvd.size()]

    elif flags.hamb:
        with h5.File(os.path.join(flags.data_folder,'generated_data_datacond_both_jets.h5'),"r") as h5f:
            bkg_p = np.stack([
                h5f['particle_data_rel_x'][hvd.rank()::hvd.size()],
                h5f['particle_data_rel_y'][hvd.rank()::hvd.size()]],1)
            bkg_j = np.stack([
                h5f['jet_features_x'][hvd.rank()::hvd.size()],
                h5f['jet_features_y'][hvd.rank()::hvd.size()]],1)
            npart = np.sum(bkg_p[:,:,:,0]>0,2)
            bkg_j[:,:,-1] = npart
            bkg_mjj = h5f['mjj'][hvd.rank()::hvd.size()]
            
    else: # Load the generated background data from plot_jet.py 
        f = flags.data_file if flags.data_file else sample_name+'.h5'
        if hvd.rank()==0: print(f'Loading {f}...')
        with h5.File(os.path.join(flags.data_folder,f),"r") as h5f:
            bkg_p = h5f['particle_features'][hvd.rank()::hvd.size()]
            bkg_j = h5f['jet_features'][hvd.rank()::hvd.size()]
            bkg_mjj = h5f['mjj'][hvd.rank()::hvd.size()]
            


    data_size = int(bkg_j.shape[0] + data_j.shape[0])
    bkg_j,bkg_p,bkg_mjj = combine_part_jet(bkg_j,bkg_p,bkg_mjj,npart=flags.npart)
    #Using recalculated values of mjj, let's apply the sideband/signal region cuts again
    bkg_j,bkg_p,bkg_mjj = apply_mjj_cut(bkg_j,bkg_p,bkg_mjj,flags.SR,
                                        mjjmin=config['MJJMIN'],mjjmax=config['MJJMAX'])
    

    if hvd.rank()==0:
        print("Loading {} generated samples and {} data samples".format(bkg_j.shape[0],data_j.shape[0]))

    # semi_labels = 0 for generated background, 1 for data (including background + signal)
    semi_labels = np.concatenate([np.zeros(bkg_j.shape[0]),np.ones(data_j.shape[0])],0)
    sample_j = np.concatenate([bkg_j,data_j],0)
    sample_p = np.concatenate([bkg_p,data_p],0)
    sample_mjj = np.concatenate([bkg_mjj,data_mjj],0)
    sample_mjj = utils.prep_mjj(sample_mjj,mjjmin=config['MJJMIN'],mjjmax=config['MJJMAX'])
    

    mask = sample_p[:,:,:,0]!=0        
    model = get_classifier(flags.SR)
    compile(model,MAX_EPOCH,BATCH_SIZE,LR,int(0.9*data_size))
    callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0),
                 hvd.callbacks.MetricAverageCallback(),
                 EarlyStopping(patience=7,restore_best_weights=True)]

    if hvd.rank()==0:
        if flags.SR:
            callbacks.append(ModelCheckpoint('../checkpoints/{}_nsig_{}_nbkg_{}_nid{}/checkpoint'.format(sample_name,flags.nsig,flags.nbkg,flags.nid),
                                             mode='auto',save_best_only=True,
                                             period=1,save_weights_only=True))
        else:
            callbacks.append(ModelCheckpoint('../checkpoints/{}_class/checkpoint'.format(config['MODEL_NAME']),mode='auto',
                                             save_best_only=True,
                                             period=1,save_weights_only=True))

    if flags.test or flags.hamb or flags.SR==False:
        weights = np.ones(sample_j.shape[0])
            
    elif flags.SR:
        if flags.reweight:
            if hvd.rank()==0:print("Loading weights...")
            model_weight = get_classifier(SR=False)
            model_weight.load_weights('../checkpoints/{}_class/checkpoint'.format(config['MODEL_NAME'])).expect_partial()
            weights = utils.reweight(bkg_j,bkg_p,model_weight,
                                     utils.prep_mjj(bkg_mjj,
                                                    mjjmin=config['MJJMIN'],
                                                    mjjmax=config['MJJMAX']),
                                     )
            weights = np.concatenate([weights,np.ones(data_j.shape[0])])
        else:
            weights = np.ones(sample_j.shape[0])
        
    sample_j,sample_p,semi_labels,mask,mjj,weights = shuffle(sample_j,sample_p,
                                                             semi_labels,mask,
                                                             sample_mjj,weights,
                                                             random_state=10)

    model.fit([sample_j,sample_p,mask] if flags.SR else [sample_j,sample_p,mask,mjj],
              semi_labels,
              batch_size=BATCH_SIZE,
              validation_split = 0.1,
              callbacks=callbacks,
              sample_weight = weights,
              epochs=MAX_EPOCH,shuffle=True,
              verbose=hvd.rank()==0,
              steps_per_epoch=int(data_size*0.9/BATCH_SIZE),
              validation_steps=int(data_size*0.1/BATCH_SIZE)
              )

    # The model has learned to differentiate between data (w/ and w/out signal and generated background)
    # The whole point is that it learns something meaninful about the signal events, since bkg and gen bkg are very similar 


    if hvd.rank() == 0:
        # As a x-check lets see how well it does in differentiating between data (w/ signal) and generated background
        pred_data = model.predict([sample_j,sample_p,mask]) 
        fpr_data, tpr_data, _ = roc_curve(semi_labels,pred_data, pos_label=1)
        auc_res_data = auc(fpr_data, tpr_data)
        print()
        print("AUC on data (w/ signal) vs gen bkg: {}".format(auc_res_data))
        print()
        if flags.SR: # check if the model can differentiate between signal and real background
            pred = model.predict([data_j,data_p,data_p[:,:,:,0]!=0])
            fpr, tpr, _ = roc_curve(labels,pred, pos_label=1)            
            auc_res =auc(fpr, tpr)
            print("AUC: {}".format(auc_res))
        else:
            pred = model.predict([sample_j,sample_p,mask,mjj])
            fpr, tpr, _ = roc_curve(semi_labels,pred, pos_label=1)
            auc_res =auc(fpr, tpr)
            print("AUC: {}".format(auc_res))                        
