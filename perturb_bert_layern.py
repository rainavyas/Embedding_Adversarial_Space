'''
Try to identify an adversarial subspace using the PCA basis.

Perform perturbations in the input embedding space for a selected token position.

Currently only for BERT model trained on IMDB dataset
'''

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from data_prep import get_train, get_test
import sys
import os
import argparse
from models import BertSequenceClassifier
from pca_tools import get_covariance_matrix, get_e_v
import matplotlib.pyplot as plt
from tools import fooling_rate, AverageMeter
from layer_handler import Bert_Handler


def make_attack(vec, epsilon):
    attack_signs = torch.sign(vec)
    attackA = attack_signs*epsilon
    attackB = -1*attackA
    return attackA, attackB

def get_perturbation_impact(handler, v, input_ids, mask, labels, model, epsilon, stepsize=1, token_pos=0, batch_size=32):
    ranks  = []
    fools = []

    # batch the data
    ds = TensorDataset(input_ids, mask, labels)
    dl = DataLoader(ds, batch_size=batch_size)
    model.eval()

    for i in range(0, v.size(0), stepsize):
        print("On rank", i)
        ranks.append(i)
        curr_v = v[i]
        fool = AverageMeter()
        with torch.no_grad():
            for id, m, lab in dl:
                original_logits = model(id, m)
                attack, _ = make_attack(curr_v, epsilon)
                layer_embeddings = handler.get_layern_outputs(id, m)
                layer_embeddings[:,token_pos,:] = layer_embeddings[:,token_pos,:] + attack

                # Pass through rest of model
                attacked_logits = handler.pass_through_rest(layer_embeddings, m)
                curr_fool = fooling_rate(original_logits, attacked_logits)
                fool.update(curr_fool.item(), id.size(0))
        fools.append(fool.avg)
    return ranks, fools

def plot_data_vs_rank(ranks, data, yname, filename):

    plt.plot(ranks, data)
    plt.xlabel("Eigenvalue Rank")
    plt.ylabel(yname)
    plt.savefig(filename)
    plt.clf()

if __name__ == '__main__':

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODEL', type=str, help='trained .th model')
    commandLineParser.add_argument('DIR', type=str, help='data base directory')
    commandLineParser.add_argument('--epsilon', type=float, default=0.1, help='l-inf perturbation size')
    commandLineParser.add_argument('--token_pos', type=int, default=0, help="token position to perturb")
    commandLineParser.add_argument('--layer_num', type=int, default=1, help="BERT layer to perturb")
    commandLineParser.add_argument('--stepsize', type=int, default=1, help="ranks step size for plot")
    commandLineParser.add_argument('--num_points', type=int, default=25000, help="number of data points to use")
    commandLineParser.add_argument('--batch_size', type=int, default=64, help="batch size")


    args = commandLineParser.parse_args()
    model_path = args.MODEL
    base_dir = args.DIR
    epsilon = args.epsilon
    token_pos = args.token_pos
    layer_num = args.layer_num
    stepsize = args.stepsize
    num_points = args.num_points
    batch_size = args.batch_size

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/perturb_bert_layern.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Load the model
    model = BertSequenceClassifier()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Create model handler
    handler = Bert_Handler(model, layer_num=layer_num)

    # Use training data to get eigenvector basis
    input_ids, mask, _ = get_train('bert', base_dir)
    input_ids = input_ids[:num_points]
    mask = mask[:num_points]
    with torch.no_grad():
        hidden_states = handler.get_layern_outputs(input_ids, mask)
        cov = get_covariance_matrix(hidden_states[:,token_pos,:])
        e, v = get_e_v(cov)

    # Get test data
    input_ids, mask, labels = get_test('bert', base_dir)
    input_ids = input_ids[:num_points]
    mask = mask[:num_points]
    labels = labels[:num_points]

    # Perturb in each eigenvector direction vs rank
    ranks, fools = get_perturbation_impact(handler, v, input_ids, mask, labels, model, epsilon, stepsize=stepsize, token_pos=token_pos, batch_size=batch_size)

    # Plot the data
    filename = 'fools_eigenvector_perturb_layer'+str(layer_num)+'_tokenpos'+str(token_pos)+'_epsilon'+str(epsilon)+'.png'
    yname = 'Fooling Rate'
    plot_data_vs_rank(ranks, fools, yname, filename)
