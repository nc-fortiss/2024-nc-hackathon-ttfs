'''
    Module that defines custom training loops for the models
'''

import torch
import torch.nn.functional as F
import config_utils

def train_FC_SNN(model, train_loader, epochs, optimizer, scheduler):
    ''' Training loop for the fully-connected SNN, from scratch using BPTT '''
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    # Loop over epochs
    for epoch in range(epochs):
        model.train()       # set model to train mode
        running_loss = 0.0  # loss accumulator
        correct = 0         # count correct labels per epoch
        total_samples = 0
        
        batches = 0
        for (inputs, truth_labels) in train_loader:       # load next batch
            # inputs = inputs.to(device)
            # truth_labels = truth_labels.to(device)
            # breakpoint()
            model.min_spike_times = {}
            optimizer.zero_grad()
            outputs = model(inputs)         # forward pass
            loss = F.cross_entropy(outputs, truth_labels) # compute loss with current weights

            loss.backward()     # compute new gradients
            optimizer.step()    # update parameters

            predicted_labels = torch.argmax(outputs,dim=1)    # max. membrane potential across 1st dimension (for each output)
            correct += (predicted_labels == truth_labels).sum().item()
            total_samples += truth_labels.shape[0]

            running_loss += loss.item()

            # if enabled, t_max will be updated layer-wise as part of the training procedure
            if config_utils.TRAIN_SHIFT:
                
                t_min_prev, t_min, k = 0.0, 1.0, 0
                for layer in model.hidden_layers:
                    layer_name = f"layer_{k}"
                    
                    t_max = t_min + max(layer.t_max - layer.t_min, 1.0*(layer.t_max - model.min_spike_times[layer_name]))
                    layer.t_min_prev = t_min_prev 
                    layer.t_min = t_min 
                    layer.t_max = t_max 

                    t_min_prev, t_min = t_min, t_max 
                    k += 1 
                
                # update t_min and t_max in the output layer accordingly
                model.output_layer.t_min = t_max        # last t_max becomes this t_min
                model.output_layer.t_max = 0            # non-spiking, irrelevant

            batches += 1
            if (batches % 100 == 0): config_utils.logging.info(f"### trained on {batches} batches: - min_ti={model.min_spike_times}")

        scheduler.step()        # update learning rate 

        config_utils.logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Acc: {100.*correct  /total_samples: .3f}")


def evaluate_FC_SNN(model, test_loader):
    ''' Evaluates the trained FC SNN model on the test set '''
    config_utils.logging.info("### Evaluating SNN ###")
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

    config_utils.logging.info(f"--- Test Accuracy: {accuracy:.2f}% | Test Loss: {test_loss:.4f} ---")


def train_FC_ReLU(model,train_data, optimizer,loss_criterion,epochs=5):
        '''
            Train the FC ReLU instance on the input 'train_data'. 
            Adapted from: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#train-the-network (Accessed 24/02/2025)
        '''
        model.train()
        train_acc = 0
        total = 0
        for epoch in range(epochs):  

            running_loss = 0.0
            for batch_idx, (data,target) in enumerate(train_data):

                # breakpoint()
                
                optimizer.zero_grad()

                outputs = model.forward(data)
                loss = loss_criterion(outputs, target)
                loss.backward()
                optimizer.step()

                _, preds  = torch.max(outputs, dim=1)

                train_acc += torch.sum(preds == target)
                total += len(preds)

                # print statistics
                running_loss += loss.item()
                if batch_idx % 100 == 0:    
                    config_utils.logging.info(f'[{epoch + 1}, {batch_idx + 1:5d}] loss: {running_loss / 100:.3f} --- acc: {train_acc / total}')
                    running_loss = 0.0

        config_utils.logging.info('Finished Training')


def evaluate_FC_ReLU(model, test_loader):
    ''' Run trained FC ReLU on testset '''
    model.eval()
    correct = 0 
    test_loss = 0.0 
    with torch.no_grad():
        for inputs, truth_labels in test_loader:
            outputs = model(inputs)
            _, predicted_labels = torch.max(outputs, dim=1)
            correct += (predicted_labels == truth_labels).sum().item()
            test_loss += torch.nn.functional.cross_entropy(outputs, truth_labels).item()


    accuracy = 100 * correct / len(test_loader.dataset)
    test_loss /= len(test_loader)

    config_utils.logging.info(f"--- Test Accuracy: {accuracy:.2f}% | Test Loss: {test_loss:.4f} ---")        