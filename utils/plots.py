import matplotlib.pyplot as plt

def plot_loss(training_loss: list, validation_loss: list) -> None:
    """ Plot the loss measured during the training of the model, here the 
    validation and training loss is shown

    Args:
        training_loss (list): Datapoints for the observed loss during training
        validation_loss (list): Datapoints for the observed loss during validation
    """    
    plt.plot(training_loss, label='Training Loss')
    plt.plot(validation_loss, label='Validity Loss')
    plt.legend()
    plt.show()