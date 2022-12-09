import argparse
import configargparse
def config_parser(cmd=None):
    parser = argparse.ArgumentParser()
    parser = configargparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        default='/home/zhangruiqi/zrq_project/nerf-pytorch/data/nerf_synthetic/lego',
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='blender',
                        choices=['blender', 'llff', 'facebook_NeRV', 'facebook_NeRV1', 'facebook_grid'],
                        help='which dataset to train/val')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[400, 400],
                        help='resolution (img_w, img_h) of the image')
    parser.add_argument('--spheric_poses', default=False, action="store_true",
                        help='whether images are taken in spheric poses (for llff)')

    parser.add_argument('--N_samples', type=int, default=128,
                        help='number of coarse samples')
    parser.add_argument('--N_importance', type=int, default=0,
                        help='number of additional fine samples')
    parser.add_argument('--use_disp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument('--perturb', type=float, default=1.0,
                        help='factor to perturb depth sampling points')
    parser.add_argument('--noise_std', type=float, default=1.0,
                        help='std dev of noise added to regularize sigma')
        
    parser.add_argument('--loss_type', type=str, default='mse',
                        choices=['mse'],
                        help='loss to use')

    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch size')
    parser.add_argument('--chunk', type=int, default=32*1024,
                        help='chunk size to split the input to avoid OOM')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of training epochs')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')

    # parser.add_argument('--ckpt_path', type=str, default="/data1/liufengyi/get_results/hash_table/checkpoints/lego_28_Hash-sh-2048/ckpts/HashTable_3.tar",
    #                     help='pretrained checkpoint path to load')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint path to load')
    parser.add_argument('--prefixes_to_ignore', nargs='+', type=str, default=['loss'],
                        help='the prefixes to ignore in the checkpoint state dict')

    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer type',
                        choices=['sgd', 'adam', 'radam', 'ranger'])
    parser.add_argument('--lr_NeRF', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='learning rate momentum')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--lr_scheduler', type=str, default='steplr',
                        help='scheduler type',
                        choices=['steplr', 'cosine', 'poly'])
    #### params for warmup, only applied when optimizer == 'sgd' or 'adam'
    parser.add_argument('--warmup_multiplier', type=float, default=1.0,
                        help='lr is multiplied by this factor after --warmup_epochs')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Gradually warm-up(increasing) learning rate in optimizer')
    ###########################
    #### params for steplr ####
    parser.add_argument('--decay_step', nargs='+', type=int, default=[20],
                        help='scheduler decay step')
    parser.add_argument('--decay_gamma', type=float, default=0.5,
                        help='learning rate decay amount')
    ###########################
    #### params for poly ####
    parser.add_argument('--poly_exp', type=float, default=0.9,
                        help='exponent for polynomial learning rate decay')
    ###########################

    # parser.add_argument('--exp_name', type=str, default='lego_64_HashMlp-sh-2048-MLP(4*128)',
    #                     help='experiment name')
    
    parser.add_argument('--exp_name', type=str, default='test',
                        help='experiment name')

    #TV loss
    parser.add_argument('--TVLoss_weight_SH', type=float, default=5e-3, help='TV loss设定 SH')
    parser.add_argument('--TVLoss_weight_sigama', type=float, default=5e-4, help='TV loss设定 sigma')

    #增加的grid的超参
    parser.add_argument('--num_voxels', type=int, default=256*256*256, help='网格分辨率')
    # parser.add_argument('--mpi_depth', type=int, default=128, help='网格分辨率的z分量')
    parser.add_argument('--in_features', type=int, default=28, help='hashtable输入维度')
    parser.add_argument('--hidden_features', type=int, default=64, help='MLP w')
    parser.add_argument('--hidden_layers', type=int, default=2, help='MLP层数')
    parser.add_argument('--out_features', type=int, default=1+27, help='hashtable输出维度')
    parser.add_argument('--num_flame', type=int, default=1, help='flame number')
    if cmd is not None:
        return parser.parse_args(cmd)
    else:
        return parser.parse_args()
    return parser.parse_args()
    return parser.parse_args(args=[])
