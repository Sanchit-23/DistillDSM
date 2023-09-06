from utils.train_main import train_model
import argparse
def main(args):
    config = {
        'data': args.data,
        'image_path' : args.image_path,
        'mask_path' : args.mask_path,
        'data_split': args.data_split,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'gpu': args.gpu,
        'seg_lr': args.lr,
        'w_decay' : args.w_decay,
        'checkpoint': args.checkpoint,
        'savepath': args.savepath,
    }

    train_model(config)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        """
    Training script.
    Provide the folder paths which contains dataset.
    Provide the folder paths which contains images.                                
    Provide the folder paths which contains labels.
    Provide the folder paths which contains data_split file.
    (Optionally) You may choose to provide batch size (--batch_size), number of epochs (--epoch),
    (Optionally) You may choose to use GPU (--gpu).
    (Optionally) You may choose to provide checkpoint, path for model weights to resume training from(--checkpoint).
    (Optionally) You may choose to provide path to save model weights(--savepath).
    (Optionally) You may choose to provide learning rate(--seg_lr).
    (Optionally) You may choose to provide decay rate(--w_decay)
    """
    )
    parser.add_argument('--data', type=str, required=True,
                        help='Folder path of dataset')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Folder path of images')
    parser.add_argument('--mask_path', type=str, required=True,
                        help='Folder path of labels')
    parser.add_argument('--data_split', type=str, required=True,
                        help='path to the data_split.json file')
    parser.add_argument('-b', '--batch_size', type=int,
                        required=False, default=15, help='Batch size used for training and validation')
    parser.add_argument('-e', '--epochs', type=int, required=False,
                        default=10, help='Max number of epochs to run')
    parser.add_argument('--gpu', type=str, help='Want GPU ?', required=False, default='False')
    parser.add_argument('--checkpoint', type=str,
                        required=False, help='Path of model weights to load for resuming training')
    parser.add_argument('--seg_lr', type=float, required=False, default=0.0001, help='learning rate')
    parser.add_argument('--w_decay', type=float, required=False, default=0.00001, help='decay rate')
    parser.add_argument('--savepath', type=str, required=False, default='model.pth', help='path to save the model')
    arguments = parser.parse_args()

    main(arguments)