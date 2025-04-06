from argparse import ArgumentParser

def get_args():

    parser = ArgumentParser(prog="main", 
                            description="This will allow you to specify what hyperparamters and other configs you can train your model with")
    
    parser.add_argument('-m','--model_name',required=True, help='This will specifiy what model to use for your model. Its an argment for the %(prog)s  program')

    return parser.parse_args()