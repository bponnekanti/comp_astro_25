##mahdis' note: this is the main file that runs the program. It uses argparse to get input from the command line.
# by command line () daneel -i path_to_parameters.yaml -t we meant:
# hey daneel program which got run in my terminal: please take the input (-i) from the path_to_parameters.yaml
# and do the transit (-t) function.

import datetime
import argparse
from daneel.parameters import Parameters
from daneel.detection import *
from daneel import transit

##mahdis' note: 
# datetime lets us track when the program starts and ends.
# now how this command line would work: we use argparse to define the arguments we want to pass.


def main():
    parser = argparse.ArgumentParser()
    ## mahdis' note: parser means
    parser.add_argument(
        "-i",
        "--input",
        dest="input_file",
        type=str,
        required=True,
        help="Input par file to pass",
    )

    parser.add_argument(
        "-d",
        "--detect",
        dest="detect",
        required=False,
        help="Initialise detection algorithms for Exoplanets",
        action="store_true",
    )

    parser.add_argument(
        "-a",
        "--atmosphere",
        dest="atmosphere",
        required=False,
        help="Atmospheric Characterisazion from input transmission spectrum",
        action="store_true",
    )

    parser.add_argument(
        "-t",
        "--transit",
        dest="transit",
        required=False,
        help="Plot transit light curve from input parameters",
        action="store_true",
    )

    args = parser.parse_args()

    """Launch Daneel"""
    start = datetime.datetime.now()
    print(f"Daneel starts at {start}")

    input_pars = Parameters(args.input_file).params

    if args.transit:
        transit = TransitModel(input_pars['transit'])
        transit.plot_light_curve()
    elif args.detect:
        pass
    elif args.atmosphere:
        pass
    if args.transit:
        transit.transit(input_pars)

    finish = datetime.datetime.now()
    print(f"Daneel finishes at {finish}")

if __name__ == "__main__":
    main()
