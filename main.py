import json
import argparse
from trainer import train


def main():
    parser = argparse.ArgumentParser(description='To be continual or to be fair')
    parser.add_argument('--prefix', default='reproduce', type=str, help='')
    parser.add_argument('--dataset', default='cifar100', type=str, help='')
    parser.add_argument('--memory_size', default=1000, type=int, help='')
    parser.add_argument('--memory_per_class', default=10, type=int, help='')
    parser.add_argument('--fixed_memory', action='store_false', default=True)
    parser.add_argument('--shuffle', action='store_false', default=True)
    parser.add_argument('--init_cls', default=50, type=int, help='')
    parser.add_argument('--increment', default=5, type=int, help='')
    parser.add_argument('--model_name', default='ucir_lto', type=str, help='finetune, replay,'
                                                                            ' icarl, bic, ucir, ucir_lto, wa,' 
                                                                            'podnet, podnet_lto')
    parser.add_argument('--convnet_type', default='resnet32', type=str, help='resnet32, cosine_resnet32, '
                                                                             'resnet18, cosine_resnet18, '
                                                                             'wrn40-2, cosine_wrn40-2')
    parser.add_argument('--method', default='baseline', type=str, help='baseline, fair, classaug')

    parser.add_argument('--save_path', default='model_saved/', type=str, help='save files directory')
    parser.add_argument('--device', default=['0'], type=list, help='')
    parser.add_argument('--seed', default=[1993], type=list, help='')

    args = parser.parse_args()
    print(args)
    train(args)


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    return parser


if __name__ == '__main__':
    main()

