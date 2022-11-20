# Importing required libraries
import sys
import json
import time
# from nni.experiment import Experiment
from model import ResNet, ResBlock, train, test, device, fine_tune
# from torch.optim import Adam
import torch
from nni.compression.pytorch.pruning import L1NormPruner, FPGMPruner
from nni.compression.pytorch.speedup import ModelSpeedup



epochs = 10

if __name__ == '__main__':

    model = ResNet(1, ResBlock, [1, 1], outputs=10).to(device)

    print("ORIGINAL UN-PRUNED MODEL: \n\n", model, "\n\n")

    

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        train(model, optimizer)
        test(model)

    print("Running the test data once to get execution time for it")
    # Starting time for unpruned model
    start_time = time.time()
    test(model)
    # Ending time for unpruned model
    end_time = time.time()

    # The total execution time of unpruned model
    exec_time = end_time - start_time
    print("\nTHE TOTAL EXECUTION TIME OF UNPRUNED MODEL: ", exec_time, "\n\n")

    torch.save(model, f'./unpruned_model.torch')
    print('Unpruned model saved')

    configuration_list = [{
        'sparsity_per_layer': 0.4,
        'op_types': ['Conv2d']
    }, {
        'exclude': True,
        'op_names': ['fc','layer0.0']
    }]
    # Defining the pruner to be used
    pruner = L1NormPruner(model, configuration_list)
    
    print(f"PRUNER WRAPPED MODEL WITH L1 NormPruner: \n\n{model}\n\n")

    # Next, compressing the model and generating masks
    _, masks = pruner.compress()
    # show the masks sparsity
    for name, mask in masks.items():
        print(name, ' sparsity : ', '{:.2}'.format(mask['weight'].sum() / mask['weight'].numel()), "\n")

    # Need to unwrap the model before speeding-up.
    pruner._unwrap_model()

    ModelSpeedup(model, torch.rand(64, 1, 28, 28).to(device), masks).speedup_model()

    print(f"PRUNED MODEL WITH L1 NormPruner: \n\n{model}\n\n")

    
    # Running the pre-training stage with pruned model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        train(model, optimizer)
        test(model)

    print("Running the test data once to get execution time for it")
    # Starting time for pruned model
    start_time = time.time()
    test(model)
    # Ending time for pruned model
    end_time = time.time()

    print(f"\nTHE TOTAL EXECUTION TIME OF PRUNED MODEL: {end_time-start_time}")

    torch.save(model, f'./pruned_model.torch')
    print('Pruned model saved')

    print("\nPerforming distillation\n")
    
    teacher_model = torch.load('./unpruned_model.torch', map_location=device)
    student_model = torch.load('./pruned_model.torch', map_location=device)
    
    models = [student_model, teacher_model]
    optimizer = torch.optim.SGD(student_model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        fine_tune(models,optimizer,5) # Set temperature to 5 for distillKL
        test(models[0])
    
    print("Running the test data once to get execution time for it")
    # Starting time for distilled model
    start_time = time.time()
    test(model)
    # Ending time for distilled model
    end_time = time.time()

    print("Distillation of pruned model done")
    print(f"Distilled and pruned model: \n\n{student_model}\n\n")
    print(f"THE TOTAL EXECUTION TIME OF DISTILLED MODEL: {end_time-start_time}")

    print("Saving distilled model in onnx format for tvm")

    # Exporting the torch model 
    student_model.eval()
    x = torch.randn(64, 1, 28, 28).to(device)
    torch.onnx.export(
        student_model,
        x,
        "distilled_model.onnx",
        export_params=True,
        input_names=["data"],
        output_names=["output"],
    )
