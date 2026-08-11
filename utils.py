import argparse

RNG_SEED = 1337


def train_parser():
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/dataset/drive.yml",
        help="Configuration file to use"
    )

    parser.add_argument("--prefix", nargs="?", type=str, default="", help="prefix for this run.")
    parser.add_argument("--device", nargs="?", type=str, default=None, help="Device to use; defaults to CPU")
    parser.add_argument("--model", nargs="?", type=str, default="", help="set the model")
    parser.add_argument("--steps", nargs="?", type=int, default=None, help="Recurrent Steps")
    parser.add_argument("--clip", nargs="?", type=float, default=None, help="gradient clip threshold")
    parser.add_argument("--hidden_size", nargs="?", type=int, default=None, help="hidden size")
    parser.add_argument("--unet_level", nargs="?", type=int, default=None, help="hidden size")
    parser.add_argument("--recurrent_level", nargs="?", type=int, default=None, help="hidden size")
    parser.add_argument("--gate", nargs="?", type=int, default=None, help="GRU gate number, 2 or 3")
    parser.add_argument("--initial", nargs="?", type=int, default=None, help="initial value of hidden state")
    parser.add_argument("--scale_weight", nargs="?", type=float, default=-1., help="loss decay after recurrent steps")
    parser.add_argument("--feature_scale", nargs="?", type=int, default=None, help="scale of feature.")
    parser.add_argument("--lr", nargs="?", type=float, default=-1, help="learning rate")
    parser.add_argument("--loss", nargs="?", type=str, default=None,
                        help="loss decay after recurrent steps")
    parser.add_argument('--structure', nargs="?", type=str, default=None, help='if gru or dru')
    parser.add_argument("--batch_size", nargs="?", type=int, default=0, help="batch size")
    parser.add_argument("--lr_n", nargs="?", type=int, default=0, help="learning rate n")
    parser.add_argument("--lr_exponent", nargs="?", type=int, default=0, help="learning rate exponent")

    parser.add_argument(
        '--benchmark',
        dest='benchmark',
        action='store_true'
    )
    parser.set_defaults(benchmark=False)
    return parser


def validate_parser(parser=None):
    parser = parser if isinstance(parser, argparse.ArgumentParser) else train_parser()
    # add the parser.
    parser.add_argument(
        "--dataset",
        nargs="?",
        type=str,
        default="pascal",
        help="Dataset to use ['pascal, camvid, ade20k etc']",
    )

    parser.add_argument(
        "--is_recurrent",
        dest='is_recurrent',
        action="store_true",
    )
    parser.set_defaults(is_recurrent=False)

    parser.add_argument(
        "--out_path",
        nargs="?",
        type=str,
        default=None,
        help="Path of the output segmap",
    )
    parser.add_argument(
        "--model_path",
        nargs="?",
        type=str,
        default="fcn8s_pascal_1_26.pkl",
        help="Path to the saved model",
    )
    parser.add_argument(
        "--eval_flip",
        dest="eval_flip",
        action="store_true",
        help="Enable evaluation with flipped image |\
                                  True by default",
    )
    parser.add_argument(
        "--no-eval_flip",
        dest="eval_flip",
        action="store_false",
        help="Disable evaluation with flipped image |\
                                  True by default",
    )
    parser.set_defaults(eval_flip=True)


    parser.add_argument(
        "--measure_time",
        dest="measure_time",
        action="store_true",
        help="Enable evaluation with time (fps) measurement |\
                                  True by default",
    )

    parser.add_argument(
        "--no-measure_time",
        dest="measure_time",
        action="store_false",
        help="Disable evaluation with time (fps) measurement |\
                                  True by default",
    )
    parser.set_defaults(measure_time=True)
    return parser


def test_parser(parser=None):
    parser = parser if isinstance(parser, argparse.ArgumentParser) else validate_parser()
    # add the parser.

    parser.add_argument(
        "--img-norm",
        dest="img_norm",
        action="store_true",
        help="Enable input image scales normalization [0, 1] \
                                  | True by default",
    )
    parser.add_argument(
        "--no-img-norm",
        dest="img_norm",
        action="store_false",
        help="Disable input image scales normalization [0, 1] |\
                                  True by default",
    )
    parser.set_defaults(img_norm=True)

    parser.add_argument(
        "--dcrf",
        dest="dcrf",
        action="store_true",
        help="Enable DenseCRF based post-processing | \
                                  False by default",
    )
    parser.add_argument(
        "--no-dcrf",
        dest="dcrf",
        action="store_false",
        help="Disable DenseCRF based post-processing | \
                                  False by default",
    )
    parser.set_defaults(dcrf=False)

    parser.add_argument(
        "--img_path", nargs="?", type=str, default=None, help="Path of the input image"
    )

    return parser
