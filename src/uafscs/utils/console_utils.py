from argparse import ArgumentParser

def get_args():

    parser = ArgumentParser(prog="main", 
                            description="This will allow you to specify what hyperparamters and other configs you can train your model with")
    
    parser.add_argument('-m' ,'--model_name', type=str,required=True, help='This will specifiy what model to use for your model. Its an argment for the %(prog)s  program')
    parser.add_argument('-wd','--weight_decay',  type=float, help="This ensures that one does not have large weight values which sometimes leads to early overfitting. This should be a floating point value ")
    parser.add_argument('-lr','--lr',default=0.01,type=float,help="Learning rate for you model. The default value is 0.01")
    parser.add_argument('-e','--epochs',type=int,default=4,help="The number of epochs you want your model to train. The default value is 4")
    parser.add_argument('-lf','--loss_fn',type=str,default='crossentropy',help='Loss function for the model. The default value is cross entropy')
    parser.add_argument('-o' , '--optim',type=str,default='sgd',help="optimizer for the model")

    return parser.parse_args()