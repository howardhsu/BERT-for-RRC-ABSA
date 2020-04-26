import argparse

from reviewlab import Evaluator

            
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--config", default=None, type=str, required=True)
   
    args = parser.parse_args()
    evaluator = Evaluator()
    evaluator.single_eval(args.config)
