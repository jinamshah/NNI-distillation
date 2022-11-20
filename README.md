# Model pruning using Microsoft NNI

### Authors

Jinam Shah </br>
Saumil Shah

## Folder structure

The python file `main.py` contains the codebase to run the different steps of the experiment except running the tvm compilation, tuning and execution job.

The file `model.py` has the model definition and the train, test functions.

We have also included the log file from the execution of the runner shell script.
********

## Code execution

`runner.sh` is used to consolidate all the steps and run them using the slurm command `sbatch` on bridges2. This script can also be run on any server using `./runner.sh` command

> It is important to note here that tvm on bridges2 machines doesn't work with GPUs and hence we have restricted ourseleves to use only the CPU.

**TVM execution details**

```bash
tvmc compile --target "llvm" --input-shapes "data:[64,1,28,28]" --output models/tvm_first_compile.tar models/distilled_model.onnx
```

This command is used to compile the distilled model so that it can be tuned using TVM

Next, we tune the model using the following command

```bash
tvmc tune --target "llvm" --output models/tune_config.json models/distilled_model.onnx
```

Finally, we were able to compile the model again using the tuned configuration and run it on the test dataset. We run it 50 times to ensure that we get fairly accurate execution times.

```bash
tvmc compile --target "llvm" --input-shapes "data:[64,1,28,28]" --tuning-records models/tune_config.json --output models/final_tuned_model.tar models/distilled_model.onnx

tvmc run --inputs test_data.npz --repeat 50 --print-time --output temp_predictions.npz models/final_tuned_model.tar
```

********

## Results

*TODO*

********