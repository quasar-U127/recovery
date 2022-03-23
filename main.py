from experiments.undo_experiment import UndoExperiment

def main():
    parser = UndoExperiment.parser()
    args = parser.parse_args()
    exp = UndoExperiment(args)
    exp.run()

if __name__=="__main__":
    main()