import csv
import random
import time
from collections import deque
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from data import Scene, PIE, COIL100, MSRCV, Webkb, sources
from model import SAEML,get_loss_fisher
np.set_printoptions(precision=4, suppress=True)

print(torch.__version__)
print(torch.cuda.is_available())


def run(args,A,B,fisher_c,ulossweight):

    if args.DatasetName == 'PIE':
        dataset = PIE()
    elif args.DatasetName == 'Scene':
        dataset = Scene()
    elif args.DatasetName == 'sources':
        dataset = sources()
    elif args.DatasetName == 'Webkb':
        dataset = Webkb()
    elif args.DatasetName == 'COIL100':
        dataset = COIL100()
    elif args.DatasetName == 'MSRCV':
        dataset = MSRCV()
    else:
        print('Dataset not recognized.')


    num_samples = len(dataset)
    num_classes = dataset.num_classes
    num_views = dataset.num_views
    dims = dataset.dims
    index = np.arange(num_samples)
    np.random.shuffle(index)
    train_index, test_index = index[:int(0.8 * num_samples)], index[int(0.8 * num_samples):]

    if args.mode == 'conflict':
        dataset.postprocessing(test_index, addNoise=True, sigma=0.5, ratio_noise=0.1, addConflict=True, ratio_conflict=0.4)

    train_loader = DataLoader(Subset(dataset, train_index), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(Subset(dataset, test_index), batch_size=args.batch_size, shuffle=False)

    model = SAEML(num_views, dims, num_classes)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_losses = []
    test_losses = []
    acc_max = 0

    # Early stopp
    patience = 3
    delta = 0.01
    acc_history = deque(maxlen=patience + 1)

    for epoch in range(1, args.epochs + 1):

        train_loss_epoch = 0
        test_loss_epoch = 0
        num_correct_train, num_sample_train = 0, 0
        num_correct_test, num_sample_test = 0, 0
        for phase in ["train", "test"]:
            if phase == "train":
                evas = []
                Ys = []
                model.train()
                for X, Y, indexes in train_loader:
                    for v in range(num_views):
                        X[v] = X[v].to(device)
                    Y = Y.to(device)

                    evidences, evidence_a, u_category = model(X)

                    _, Y_pre = torch.max(evidence_a, dim=1)
                    evas.append(evidence_a.cpu().detach().numpy())
                    Ys.append(Y.cpu().detach().numpy())
                    num_correct_train += (Y_pre == Y).sum().item()
                    num_sample_train += Y.shape[0]

                    train_loss = get_loss_fisher(evidences, evidence_a, Y, epoch, u_category, A, B, fisher_c, ulossweight)

                    optimizer.zero_grad()
                    train_loss.backward()
                    optimizer.step()
                    train_loss_epoch += train_loss.item() * Y.shape[0]
                acc_train = num_correct_train / num_sample_train
            elif phase == "test":
                model.eval()
                Y_all, model_pred_all = [], []
                for X, Y, indexes in test_loader:
                    for v in range(num_views):
                        X[v] = X[v].to(device)
                    Y = Y.to(device)
                    Y_all.append(Y)

                    with torch.no_grad():
                        evidences, evidence_a, u_category = model(X)
                        evi_alp = evidence_a + 1
                        model_pred_all.append(evi_alp.to(device))

                        _, Y_pre = torch.max(evidence_a, dim=1)
                        num_correct_test += (Y_pre == Y).sum().item()
                        num_sample_test += Y.shape[0]

                        test_loss = get_loss_fisher(evidences, evidence_a, Y, epoch, u_category, A, B, fisher_c,ulossweight)

                        test_loss_epoch += test_loss.item() * Y.shape[0]

                acc_test = num_correct_test / num_sample_test
                acc_history.append(acc_test)

                if len(acc_history) > patience:
                    should_stop = True
                    for i in range(1, patience + 1):
                        if abs(acc_history[-i] - acc_history[-i - 1]) >= delta:
                            should_stop = False
                            break
                    if should_stop and acc_test >= acc_max:
                        acc_max = acc_test

        train_losses.append(train_loss_epoch / num_sample_train)
        test_losses.append(test_loss_epoch / num_sample_test)

    # #plot loss curve
    # plt.figure(figsize=(12, 6))
    # plt.plot(range(1, args.epochs + 1), train_losses, label='Train Loss', color='blue')
    # plt.plot(range(1, args.epochs + 1), test_losses, label='Test Loss', color='red')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Train and Test Loss over Epochs')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    return acc_max


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training [default: 100]')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--annealing_step', type=int, default=50, metavar='N',
                        help='gradually increase the value of lambda from 0 to 1')
    parser.add_argument('--lr', type=float, default=0.008, metavar='LR',
                        help='learning rate')
    parser.add_argument('--DatasetName', type=str, default='Scene', metavar='N',
                        help='PIE/Scene/sources/Webkb/COIL100/MSRCV')
    parser.add_argument('--mode', type=str, default='conflict', metavar='N',
                        help='normal/conflict')
    args = parser.parse_args()

    result_file_path = f'SAEML_results/{args.mode}/{args.DatasetName}.csv'
    with open(result_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        param_header = ['Parameter', 'Value']
        writer.writerow(param_header)
        param_rows = [
            ['batch_size', args.batch_size],
            ['epochs', args.epochs],
            ['annealing_step', args.annealing_step],
            ['lr', args.lr],
            ['DatasetName', args.DatasetName]
        ]
        writer.writerows(param_rows)
        writer.writerow([])
        header = ['A']+['B']+['fisher_c']+['ulossweight'] + [f'acc{i + 1}' for i in range(5)] + ['acc_mean', 'acc_var', 'acc_std']
        writer.writerow(header)
        file.flush()

        for A in [1]:
            for B in [3]:  # 1
                for fisher_c in [0.005]:  # 0.005
                    for ulossweight in [0.04]:  # 0.5
                        accs = []
                        start_time0 = time.time()
                        for i in range(5):
                            start_time = time.time()
                            acc = run(args,A, B, fisher_c, ulossweight)
                            accs.append([acc])
                            end_time = time.time()
                            elapsed_time = end_time - start_time
                            print("running time：", elapsed_time, "second", " acc: ", acc)

                        accs_aver = np.mean(accs)
                        accs_var = np.var(accs)
                        accs_stdv = np.std(accs)
                        accs.append([accs_aver])
                        accs.append([accs_var])
                        accs.append([accs_stdv])

                        flat_accs = [item for sublist in accs for item in sublist]
                        writer.writerow([A]+[B]+[fisher_c]+[ulossweight]+flat_accs)
                        file.flush()
                        end_time0 = time.time()
                        elapsed_time0 = end_time0 - start_time0
                        print("running time：", elapsed_time0, "second", '   best_acc:', accs)