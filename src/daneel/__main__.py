##mahdis' note: this is the main file that runs the program. It uses argparse to get input from the command line.
# by command line () daneel -i path_to_parameters.yaml -t we meant:
# hey daneel program which got run in my terminal: please take the input (-i) from the path_to_parameters.yaml
# and do the transit (-t) function.

import datetime
import argparse
from daneel.parameters import Parameters
from daneel.detection import *
from daneel import transit
from daneel.parameters import plot_transits

##mahdis' note: 
# datetime lets us track when the program starts and ends.
# now how this command line would work: we use argparse to define the arguments we want to pass.


def main():
    parser = argparse.ArgumentParser()
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

    # example code for multiple input files:

    input_pars_in = args.input_file.split(",") #seperated by comma (no space)
    input_pars = []
    for ymlfilepath in input_pars_in:
        input_pars.append(Parameters(ymlfilepath).params['transit'])
    
    

    if args.transit:
        transit_list = []
        print("builidng transit list")
        for transit in input_pars:
            transit_list.append(TransitModel(transit))  # was input_pars['transit']
        print("calling plots")
        plot_transits(transit_list)
    elif args.detect:
        pass
    elif args.atmosphere:
        pass

    finish = datetime.datetime.now()
    print(f"Daneel finishes at {finish}")

if __name__ == "__main__":
    main()
