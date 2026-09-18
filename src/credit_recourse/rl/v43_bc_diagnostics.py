"""Read-only Stage4 label, weighted loss and final actor diagnostics."""
import numpy as np
import pandas as pd
import torch
from credit_recourse.rl.contracts.v43_encoder import ACTION_IDS
from credit_recourse.rl.v43_one_pass_contract import write_json

def save_bc_diagnostics(consumer,model,states,targets,fit,cw,fw,weights,folder):
    model.eval();index=np.flatnonzero(fit);logits=[]
    with torch.no_grad():
        for start in range(0,len(index),512):
            logits.append(model.forward_from_state(states[index[start:start+512]])['actor_logits'].cpu())
    logits=torch.cat(logits);prob=logits.softmax(1);logp=logits.log_softmax(1)
    target=targets[index].cpu();weight=weights.cpu()
    hard=consumer.rows(4).loc[fit,'candidate_id'].value_counts()
    mass=target.sum(0).numpy();weighted=(target*weight).sum(0).numpy()
    loss_mass=-(target*weight*logp).sum(0).numpy()
    chosen=prob.argmax(1).numpy();counts=np.bincount(chosen,minlength=9)
    entropy=-(prob*logp).sum(1).numpy()
    table=pd.DataFrame({'candidate_id':ACTION_IDS,'hard_count':[int(hard.get(a,0)) for a in ACTION_IDS],
        'soft_mass':mass,'class_weight':cw.cpu().numpy(),'family_weight':fw.cpu().numpy(),
        'combined_effective_BC_weight':weight.numpy(),'weighted_target_mass':weighted,
        'weighted_loss_mass':loss_mass,'actor_argmax_count':counts,
        'mean_actor_probability':prob.mean(0).numpy(),
        'actor_entropy_mean_when_chosen':[float(entropy[chosen==i].mean()) if (chosen==i).any() else None for i in range(9)]})
    for column in ('hard_count','soft_mass','weighted_target_mass','weighted_loss_mass','actor_argmax_count'):
        table[column+'_share']=table[column]/table[column].sum()
    table.to_csv(folder/'action_diagnostics.csv',index=False)
    shares=counts/len(index);positive=shares[shares>0]
    write_json(folder/'action_diagnostics.json',{'status':'PASS','rows':len(index),'population':'eligible Stage4 training decision states only',
        'future_rows':0,'actor_probability_entropy_mean':float(entropy.mean()),
        'argmax_distribution_entropy':float(-(positive*np.log(positive)).sum()),
        'weighted_final_BC_loss':float(loss_mass.sum()/len(index)),
        'weighted_loss_definition':'sum over training states of -combined_weight * soft_target * log(final actor probability)',
        'actions':table.astype(object).where(table.notna(),None).to_dict('records')})
