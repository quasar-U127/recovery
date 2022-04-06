import argparse
from typing import Dict
from experiments.alteration import AlterationExperiment
from experiments.alteration_onehot import AlterationOneHotExperiment
from experiments.experiment import Experiment
from experiments.undo import UndoExperiment

def main():
    experiments:dict[str,Experiment] = {
        "undo":UndoExperiment,
        "alteration": AlterationExperiment,
        "alteration_one_hot": AlterationOneHotExperiment
    }

    parser = argparse.ArgumentParser("Recovery Experiments")

    subparser = parser.add_subparsers(dest="exp_type",required=True)

    for exp_type in experiments:
        _parser = subparser.add_parser(exp_type)
        experiments[exp_type].add_parser(_parser)

    args = parser.parse_args()

    exp = experiments[args.exp_type](args)
    print(exp)
    exp.run()

if __name__=="__main__":
    main()