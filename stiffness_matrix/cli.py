def main():
    import argparse

    parser = argparse.ArgumentParser(prog="element stiffness matrix", description="")

    subparsers = parser.add_subparsers(help="Sub-commands")
    parser_main = subparsers.add_parser("run", help="Execute tool")
    parser_main.set_defaults(handler=default_run)
    parser_clean = subparsers.add_parser("clean", help="Remove previous results")

    parser_main.add_argument(
        "-ele",
        "--ele_type",
        help="Type of element (Q4|Q9)",
        choices=["Q4", "Q9"],
        type=str,
        required=True,
    )

    # Universal arguments
    parser_main.add_argument(
        "-fig",
        "--figout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show plots (use --no-figout to disable). Default: True",
    )
    parser_main.add_argument(
        "-zip",
        "--savezip",
        help="Option flag to compress fig files into zip file",
        action="store_true",
    )
    parser_main.add_argument(
        "-clean",
        "--clean",
        help="Option flag to remove vtk files after creating zip file",
        action="store_true",
    )

    args = parser.parse_args()
    if hasattr(args, "handler"):
        args.handler(args)
    else:
        parser.print_help()

    return


## 나중에 fig들 zip으로 만들고 정리하기 .
def clean_all(args):
    ("구현하지 않았습니다.")
    return


def default_run(args):
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    ele_type = args.ele_type
    fig_out = args.figout
    save_zip = args.savezip
    clean_zip = args.clean

    match ele_type:
        case "Q4":
            dir_save = Path(f"./Q4_elements_mode")
            fname_svg = f"eigenmode"

            from .lib.ReferenceQ4 import Q4_ElementAnalysis

            qea = Q4_ElementAnalysis(dir_save)
            qea.Q4_IsoparametricElement(fig_out=fig_out)

        case "Q9":
            dir_save = Path(f"./Q9_elements_mode")
            fname_svg = f"eigenmode"

            from .lib.ReferenceQ9 import Q9_ElementAnalysis

            qea = Q9_ElementAnalysis(dir_save)
            qea.Q9_IsoparametricElement(fig_out=fig_out)

    if save_zip and clean_zip:
        import shutil

        shutil.rmtree(path=dir_save)

    plt.show()
    return
