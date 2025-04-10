import torch
#----------------------------------------------------------------------------------------
#  DEFAULT SETTINGS
#----------------------------------------------------------------------------------------

DESCRIPTION = "UAFS AI Lab"


DEFAULTS = {
	"lrate": 			0.1,
	"epochs":			7,
	"batch_size":		128,
	"dropout":			0.4,
    "input_dims":       28,
	"num_labels":		2,
}


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")