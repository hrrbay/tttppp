import argparse

import numpy as np
import torch
import torchvision

import data_config
import data_loader

import matplotlib.pyplot as plt
from tqdm import tqdm

def get_label(idx, wins_per_vid, vids):
    vid_idx = 0
    for i, wins in enumerate(wins_per_vid):
        if idx < wins:
            break
        idx -= wins
        vid_idx += 1

    vid = vids[vid_idx]
    # print(f'vid_idx: {vid_idx}')
    seq_idx = 0
    for i, wins in enumerate(vid.wins_per_seq):
        if idx < wins:
            break
        idx -= wins
        seq_idx += 1

    return vid.next_points[seq_idx]

def inspect_data(trn_loader, val_loader, tst_loader):
    train_ds = trn_loader.dataset.dataset
    val_ds = val_loader.dataset.dataset
    test_ds = tst_loader.dataset

    train_num_sequences = []
    train_avg_sequence_length = []
    for vid in train_ds.vids:
        train_num_sequences.append(len(vid.sequences))
        vid_frame_num = 0
        for seq in vid.sequences:
            vid_frame_num += seq.shape[0]
        train_avg_sequence_length.append(vid_frame_num / len(vid.sequences))

    test_num_sequences = []
    test_avg_sequence_length = []
    for vid in test_ds.vids:
        test_num_sequences.append(len(vid.sequences))
        vid_frame_num = 0
        for seq in vid.sequences:
            vid_frame_num += seq.shape[0]
        test_avg_sequence_length.append(vid_frame_num / len(vid.sequences))

    train_labels = []
    for i in tqdm(trn_loader.dataset.indices):
        train_labels.append(get_label(i, train_ds.wins_per_vid, train_ds.vids))

    val_labels = []
    for i in tqdm(val_loader.dataset.indices):
        val_labels.append(get_label(i, val_ds.wins_per_vid, val_ds.vids))

    test_labels = []
    for i in tqdm(range(len(test_ds))):
        test_labels.append(get_label(i, test_ds.wins_per_vid, test_ds.vids))

    # plot binary class distribution
    trn_labels = np.asarray(train_labels)
    val_labels = np.asarray(val_labels)
    test_labels = np.asarray(test_labels)

    ones_train = np.count_nonzero(trn_labels)
    labels = ['Left train', 'Right train', 'Left val', 'Right val', 'Left test', 'Right test']
    label_counts = [trn_labels.shape[0] - ones_train, ones_train, val_labels.shape[0] - np.count_nonzero(val_labels),
                    np.count_nonzero(val_labels), test_labels.shape[0] - np.count_nonzero(test_labels), np.count_nonzero(test_labels)]
    plt.bar(labels, label_counts, color=['blue', 'orange', 'blue', 'orange', 'blue', 'orange'])
    plt.xlabel('Classes')
    plt.ylabel('Count')
    plt.title('Label Distribution of points')
    plt.show()

    # plot number of sequences per video
    max_length = max(len(train_num_sequences), len(test_num_sequences))
    train_num_sequences += [0] * (max_length - len(train_num_sequences))
    test_num_sequences += [0] * (max_length - len(test_num_sequences))

    # Plotting the bar chart
    width = 0.25  # Width of the bars
    x = range(max_length)

    plt.bar(x, train_num_sequences, width, label='Training Videos')
    plt.bar([i + 2 * width for i in x], test_num_sequences, width, label='Test Videos')

    plt.xlabel('Video')
    plt.ylabel('Sequence Count')
    plt.title('Sequence Counts per Video')
    plt.legend()
    plt.show()

    # plot average sequence length
    max_length = max(len(train_avg_sequence_length), len(test_avg_sequence_length))
    train_avg_sequence_length += [0] * (max_length - len(train_avg_sequence_length))
    test_avg_sequence_length += [0] * (max_length - len(test_avg_sequence_length))

    # Plotting the bar chart
    width = 0.25  # Width of the bars
    x = range(max_length)

    plt.bar(x, train_avg_sequence_length, width, label='Training Videos')
    plt.bar([i + 2 * width for i in x], test_avg_sequence_length, width, label='Test Videos')

    plt.xlabel('Videos')
    plt.ylabel('Average Sequence Length')

    plt.title('Average Sequence Length per Video')
    plt.legend()
    plt.show()



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validation', default=0.1, type=float)
    parser.add_argument('--batch-size', type=int,  default=32)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data-config', type=str, default='base')
    parser.add_argument('--src-fps', type=int, default=120)
    parser.add_argument('--target-fps', type=int, default=30)
    parser.add_argument('--labeled-start', action='store_true', default=False)
    parser.add_argument('--window-size', type=int, default=7)
    parser.add_argument('--gpu', type=int, default='0')

    return parser.parse_args()

def main():

    args = parse_arguments()

    if args.gpu < 0:
        print(f'* GPU < 0. Will use CPU instead.')
        device = 'cpu'
    if not torch.cuda.is_available():
        print('* WARNING: cuda is not available. Will use CPU instead.')
        device = 'cpu'
    elif torch.cuda.device_count() < args.gpu:
        print(f'* WARNING: torch.cuda.device_count() < {args.gpu}. Will use GPU 0 instead.')
        device = 'cuda'
        torch.cuda.device(0)
    else:
        device = 'cuda'
        torch.cuda.device(args.gpu)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_conf = data_config.data_config[args.data_config]
    transforms = data_conf['transforms']
    transforms = [torchvision.transforms.ToTensor()] + transforms
    data_path = data_conf['path']

    trn_loader, val_loader, tst_loader = data_loader.load_data(data_path, args.batch_size, args.src_fps, args.target_fps, args.labeled_start,
                                                               args.window_size, args.seed, transforms=transforms, validation=args.validation)

    inspect_data(trn_loader, val_loader, tst_loader)

if __name__ == '__main__':
    main()