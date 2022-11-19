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



epochs = 2
pruner_used = "L1NormPruner"          # -> (L1NormPruner, FPGMPruner)
config_choice = "config_list_1"     # -> or config_list_2
# device = "cuda" # or cuda

if __name__ == '__main__':

    # print("\nDEVICE BEING USED: ", device, "\n")

    # Defined original unpruned model
    # model = ResNet().to(device)
    model = ResNet(1, ResBlock, [1, 1], outputs=10).to(device)

    print("ORIGINAL UN-PRUNED MODEL: \n\n", model, "\n\n")

    # Starting time for unpruned model
    start_time = time.time()

    # Running the pre-training stage with original unpruned model
    # optimizer = Adam(model.parameters(), lr=1e-3)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        train(model, optimizer)
        test(model)

    # Ending time for unpruned model
    end_time = time.time()

    # The total execution time of unpruned model
    exec_time = end_time - start_time
    print("\nTHE TOTAL EXECUTION TIME OF UNPRUNED MODEL: ", exec_time, "\n\n")

    torch.save(model, f'./pretrained_model.pth')
    print('Pretrained model saved')

    configuration_list = [{
        'sparsity_per_layer': 0.4,
        'op_types': ['Conv2d']
    }, {
        'exclude': True,
        'op_names': ['fc','layer0.0']
    }]
    # Defining the pruner to be used
    pruner = L1NormPruner(model, configuration_list)
    
    print("PRUNER WRAPPED MODEL WITH {}: \n\n".format(pruner_used), model, "\n\n")

    # Next, compressing the model and generating masks
    _, masks = pruner.compress()
    # show the masks sparsity
    for name, mask in masks.items():
        print(name, ' sparsity : ', '{:.2}'.format(mask['weight'].sum() / mask['weight'].numel()), "\n")

    # Need to unwrap the model before speeding-up.
    pruner._unwrap_model()

    ModelSpeedup(model, torch.rand(64, 1, 28, 28).to(device), masks).speedup_model()

    print("\nPRUNED MODEL WITH {}: \n\n".format(pruner_used), model, "\n\n")

    # Starting time for pruned model
    start_time = time.time()

    # Running the pre-training stage with pruned model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        train(model, optimizer)
        test(model)

    # Ending time for pruned model
    end_time = time.time()

    # The total execution time of pruned model
    exec_time = end_time - start_time
    print("\nTHE TOTAL EXECUTION TIME OF PRUNED MODEL: ", exec_time, "\n\n")

    torch.save(model, f'./pruned_model.pth')
    print('Pruned model saved')


    print("\n\nPerforming distillation: loading teacher and student models")

    teacher_model = torch.load('./pretrained_model.pth', map_location=device)
    student_model = torch.load('./pruned_model.pth', map_location=device)
    
    # model_t = teacher_model.eval()
    # model_s = student_model.train()
    models = [student_model, teacher_model]
    optimizer = torch.optim.SGD(student_model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        fine_tune(models,optimizer,5)
        test(models[0])

    student_model.eval()
    x = torch.randn(64, 1, 28, 28).to(device)
    torch.onnx.export(
        student_model,
        x,
        "distilled_pruned_model.onnx",
        export_params=True,
        input_names=["dummy_input"],
        output_names=["dummpy_output"],
    )