import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import sys
import os
import argparse
from tools import AverageMeter, accuracy_topk, get_default_device, fooling_rate, kl_avg
from models import ElectraSequenceClassifier, BertSequenceClassifier, RobertaSequenceClassifier, XlnetSequenceClassifier
from attack_models import Attack_handler
from data_prep import get_test
from pca_tools import get_covariance_matrix, get_e_v
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Subset

def load_model(arch, model_path, device):
    if arch == 'electra':
        model = ElectraSequenceClassifier()
    elif arch == 'bert':
        model = BertSequenceClassifier()
    elif arch == 'roberta':
        model = RobertaSequenceClassifier()
    elif arch == 'xlnet':
        model = XlnetSequenceClassifier()
    else:
        raise Exception("Unsupported architecture")
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model

def get_embedding_X(dataloader, attack_handler, device):
    Xs = []

    for id, mask, target in dataloader:

        id = id.to(device)
        mask = mask.to(device)

        with torch.no_grad():
            curr_X = attack_handler.get_sentence_embedding(id, mask)
        curr_X = curr_X.cpu()
        Xs.append(curr_X)

    X = torch.cat(Xs)
    return X

def get_embedding_cov(dataloader, attack_handler, device):
    X = get_embedding_X(dataloader, attack_handler, device)
    cov = get_covariance_matrix(X)
    return cov

def get_r_f_e_kl(dataloader, model, attack_handler, e, v, stepsize, epsilon, device):
    ranks = []
    fools = []
    eigenvalues = []
    kls = []

    for i in range(0, e.size(0), stepsize):
        ranks.append(i)
        eigenvalues.append(e[i])
        attack_direction = v[i]

        attack_signs = torch.sign(attack_direction)
        attack = attack_signs * epsilon # can multiply by -1 to reverse direction
        attack = attack.to(device)

        fool = AverageMeter()
        kl = AverageMeter()

        for id, mask, target in dataloader:
            id = id.to(device)
            mask = mask.to(device)

            with torch.no_grad():
                original_logits = model(id, mask)
                attacked_logits = attack_handler.attack(id, mask, attack)
                curr_fool = fooling_rate(original_logits, attacked_logits)
                curr_kl = kl_avg(original_logits, attacked_logits)
            fool.update(curr_fool.item(), id.size(0))
            kl.update(curr_kl.item(), id.size(0))
        fools.append(fool.avg)
        kls.append(kl.avg)
    return ranks, fools, eigenvalues, kls

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODEL', type=str, help="trained model path")
    commandLineParser.add_argument('ARCH', type=str, help='electra, bert, roberta, xlnet')
    commandLineParser.add_argument('--stepsize', type=int, default=20, help="Plot stepsize")
    commandLineParser.add_argument('--B', type=int, default=16, help="Specify batch size")
    commandLineParser.add_argument('--epsilon', type=float, default=0.1, help='Attack size l-inf norm')
    commandLineParser.add_argument('--S', type=int, default=5000, help='Subset size')

    args = commandLineParser.parse_args()
    model_path = args.MODEL
    arch = args.ARCH
    stepsize = args.stepsize
    batch_size = args.B
    epsilon = args.epsilon
    subset_size = args.S

    np.random.seed(0)

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/eigenvector_perturb.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get the device
    device = get_default_device()

    # Load the test data - keep only subset
    input_ids, mask, labels = get_test(arch)
    indices = torch.from_numpy(np.random.choice(input_ids.size(0), size=subset_size, replace=False))
    ds = Subset(TensorDataset(input_ids, mask, labels), indices)
    dl = DataLoader(ds, batch_size=batch_size)

    # Load the trained model
    model = load_model(arch, model_path, device)

    # Initialise the attack handler
    attack_handler = Attack_handler(model, arch)

    # Get embedding space eigenvalues and eigenvectors
    cov = get_embedding_cov(dl, attack_handler, device)
    e,v = get_e_v(cov)

    # Get the ranks, eigenvalues and fooling rates
    rank, fool, eig, kl = get_r_f_e(dl, model, attack_handler, e, v, stepsize, epsilon, device)

    # Plot fooling rate against rank
    plt.plot(rank, fool)
    plt.xlabel("Eigenvalue Rank")
    plt.ylabel("Fooling Rate")
    plt.title("Perturbation in embedding space of "+arch)
    plt.savefig("fool_vs_rank_"+arch+".png")
    plt.clf()

    # Plot eigenvalue size against rank
    plt.plot(rank, eig)
    plt.yscale("log")
    plt.xlabel("Eigenvalue Rank")
    plt.ylabel("Eigenvalue")
    plt.title("Embedding space of "+arch)
    plt.savefig("eig_vs_rank_"+arch+".png")
    plt.clf()

    # Plot kl against rank
    plt.plot(rank, kl)
    plt.xlabel("Eigenvalue Rank")
    plt.ylabel("KL Divergence")
    plt.title("Perturbation in embedding space of "+arch)
    plt.savefig("kl_vs_rank_"+arch+".png")
    plt.clf()
