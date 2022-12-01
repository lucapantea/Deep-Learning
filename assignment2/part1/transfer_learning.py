# Imports used for this notebook
import matplotlib.pyplot as plt

import argparse
import numpy as np
import torch
import os

# Device business
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f'Device running this notebook: {device}')

DATASET_PATH = "./data"
CHECKPOINT_PATH = "../saved_models/part1"

# Create checkpoint path if it doesn't exist yet
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# Setting the Pytorch environment for saving the models
os.environ['TORCH_HOME'] = CHECKPOINT_PATH

# Method for setting the seed
def set_seed(seed):
    """
    Function for setting the seed for reproducibility & benchmarking.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_model(model_name):
    match model_name:
        case 'vgg11':
            from torchvision.models import vgg11
            return vgg11
        case 'vgg11_bn':
            from torchvision.models import vgg11_bn
            return vgg11_bn
        case 'resnet18':
            from torchvision.models import resnet18
            return resnet18
        case 'resnet34':
            from torchvision.models import resnet34
            return resnet34
        case 'densenet121':
            from torchvision.models import densenet121
            return densenet121
        case 'mobilenet_v3_small':
            from torchvision.models import mobilenet_v3_small
            return mobilenet_v3_small
        case _:
            print('Model name unknown.')

def run_inference(model_name, model_weights, num_runs=1, no_grad=True, batch_size=1):    
    # Cuda events for calculating elapsed time
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    image_size = (batch_size, 3, 224, 224) 

    # Initialize model
    model = get_model(model_name)(weights=model_weights).to(device)
    model.eval()

    # GPU warmup
    for _ in range(3):
        model(torch.rand(size=image_size).to(device))

    # Model outputs
    pred = None
    inference_times = []
    memory_buffer = []
    
    # Perform inference with given context manager
    with torch.no_grad() if no_grad else torch.enable_grad():
        for _ in range(num_runs):
            # Start recording the time
            start.record()
            pred = model(torch.rand(size=image_size).to(device))
            end.record()

            # Sync clocks & record stats
            torch.cuda.synchronize()
            inference_times.append(start.elapsed_time(end))
            memory_buffer.append(torch.cuda.memory_allocated() * 1e-6)
    
    del model
    pred.detach()
    torch.cuda.empty_cache()
    return pred, np.mean(inference_times), np.mean(memory_buffer)

if __name__ == '__main__':
    set_seed(42)

    # Initializing models
    models = {
        'vgg11': 69.02,
        'vgg11_bn': 70.37,
        'resnet18': 69.758,
        'resnet34': 73.314,
        'densenet121': 74.434,
        'mobilenet_v3_small': 67.668,
    }

    # Arguments to run experiment
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default=1, type=int,
                        help='Which experiment int - a):1, b):2, c):3')

    args = parser.parse_args()
    kwargs = vars(args)

    match kwargs.get('experiment'):
        case 1:
            # Question 1.1 a)
            inference_times_no_grad, memory_used_no_grad = [], []

            for model_name in models.keys():
                _, inference_speed, _ = run_inference(model_name, 'IMAGENET1K_V1', num_runs=5, no_grad=True, batch_size=1)
                inference_times_no_grad.append(inference_speed)

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, tight_layout=True, figsize=(18, 5))

            # Inference speed per model
            ax1.set_xlabel('Pre-Trained Models')
            ax1.set_ylabel('Average Inference Speed (ms)')
            ax1.xaxis.set_tick_params(rotation=10)
            ax1.plot(models.keys(), inference_times_no_grad, 'o')

            # Inference speed for each model vs top1 acc
            ax2.set_xlabel('Average Inference Speed (ms)')
            ax2.set_ylabel('Model Top-1 accuracy')
            for i, model_name in enumerate(models.keys()):
                ax2.plot(inference_times_no_grad[i], models[model_name], "s", label=model_name)

            # Inference speed vs number of parameters
            ax3.set_xlabel('Average Inference Speed (ms)')
            ax3.set_ylabel('Model Number of Parameters')
            for i, model_name in enumerate(models.keys()):
                num_params = sum(p.numel() for p in get_model(model_name)(weights='IMAGENET1K_V1').parameters())
                ax3.plot(inference_times_no_grad[i], num_params, "D", label=model_name)

            # Legend business
            box = ax2.get_position()
            ax2.set_position([box.x0, box.y0 + box.height * 0.1,
                                box.width, box.height * 0.9])
            ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
                        fancybox=True, shadow=True, ncol=3, prop={'size': 13})
            plt.savefig(f'./question_1-1_a.png')
        case 2:
            # Question 1.1 b)
            inference_times_no_grad = []
            inference_times_grad = []

            for model_name in models.keys():
                _, inference_speed_no_grad, _ = run_inference(model_name, 'IMAGENET1K_V1', num_runs=5, no_grad=True, batch_size=1)
                _, inference_speed_grad, _ = run_inference(model_name, 'IMAGENET1K_V1', num_runs=5, no_grad=False, batch_size=1)
                inference_times_no_grad.append(inference_speed_no_grad)
                inference_times_grad.append(inference_speed_grad)

            fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(8, 4))

            # Inference speed per model
            ax.set_xlabel('Pre-Trained Models')
            ax.set_ylabel('Average Inference Speed (ms)')
            ax.xaxis.set_tick_params(rotation=10)
            ax.plot(models.keys(), inference_times_no_grad, 'o', label="With torch.no_grad()")
            ax.plot(models.keys(), inference_times_grad, 'o', label="With torch.grad_enabled()")
            plt.legend()
            plt.savefig(f'./question_1-1_b.png')

        case 3:
            # Question 1.1 c)
            fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(8, 4))
            memory_used_no_grad = []
            memory_used_grad = []
            for model_name in models.keys():
                _, _, memory_no_grad = run_inference(model_name, 'IMAGENET1K_V1', num_runs=1, no_grad=True, batch_size=64)
                _, _, memory_grad = run_inference(model_name, 'IMAGENET1K_V1', num_runs=1, no_grad=False, batch_size=64)
                memory_used_no_grad.append(memory_no_grad)
                memory_used_grad.append(memory_grad)

            # Memory usage per model
            ax.set_xlabel('Pre-Trained Models')
            ax.set_ylabel('GPU vRAM usage (MB)')
            ax.xaxis.set_tick_params(rotation=10)
            ax.plot(models.keys(), memory_used_no_grad, 'o', label="With torch.no_grad()")
            ax.plot(models.keys(), memory_used_grad, 'o', label="With torch.grad_enabled()")
            plt.legend()
            plt.savefig(f'./question_1-1_c.png')