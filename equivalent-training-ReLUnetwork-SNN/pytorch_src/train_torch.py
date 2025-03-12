'''
    Module that defines custom training loops for the models
'''

import torch
import torch.nn.functional as F
import config_utils


def train_FC_SNN(model, train_loader, epochs, optimizer, scheduler):
    ''' Training loop for the fully-connected SNN, from scratch using BPTT '''

    # Loop over epochs
    for epoch in range(epochs):
        model.train()       # set model to train mode
        running_loss = 0.0  # loss accumulator
        correct = 0         # count correct labels per epoch
        total_samples = 0
        
        for (inputs, truth_labels) in train_loader:       # load next batch
            # breakpoint()
            optimizer.zero_grad()
            outputs = model(inputs)         # forward pass
            loss = F.cross_entropy(outputs, truth_labels) # compute loss with current weights

            loss.backward()     # compute new gradients
            optimizer.step()    # update parameters

            predicted_labels = torch.argmax(outputs,dim=1)    # max. membrane potential across 1st dimension (for each output)
            correct += (predicted_labels == truth_labels).sum().item()
            total_samples += truth_labels.shape[0]

            running_loss += loss.item()

        scheduler.step()        # update learning rate 

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Acc: {100.*correct  /total_samples: .3f}")
        print("in total predicted: ", total_samples)


def evaluate_FC_SNN(model, test_loader):
    ''' Evaluates the trained FC SNN model on the test set '''
    model.eval()
    correct = 0 
    total = 0
    test_loss = 0.0 
    with torch.no_grad():
        for inputs, truth_labels in test_loader:
            output_logits = model(inputs)
            predicted_labels = torch.argmax(output_logits, dim=1)
            correct += (predicted_labels == truth_labels).sum().item()
            test_loss += torch.nn.functional.cross_entropy(output_logits, truth_labels).item()


    accuracy = 100 * correct / len(test_loader.dataset)
    test_loss /= len(test_loader)

    config_utils.logging.info(f"--- TEST Accuracy: {accuracy:.2f}% | Test Loss: {test_loss:.4f} ---")