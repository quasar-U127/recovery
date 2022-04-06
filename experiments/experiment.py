from datetime import datetime
import argparse
import os


class Experiment(object):
    
    def add_parser(parser:argparse.ArgumentParser)->None:
        raise NotImplementedError()

    def __init__(self,root:str) -> None:
        super().__init__()
        time_str = str(datetime.now())
        print("Time: {}".format(time_str))
        self.root = os.path.join(root, time_str)
        os.makedirs(self.root)

    def run(self)->None:
        raise NotImplementedError()

    

