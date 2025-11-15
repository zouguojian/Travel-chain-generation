import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='CNN', help='model')
parser.add_argument("--test", action="store_true", help="test program")
args = parser.parse_args()

model = args.model

def main():
    if model == 'CNN':
        if not args.test:  # training
            run = 'python ./baselines/travelTrajectory/CNN/main.py'
            os.system(run)
        else:
            run = 'python ./baselines/travelTrajectory/CNN/main.py --test'
            os.system(run)
    elif model == 'DNN':
        if not args.test:  # training
            run = 'python ./baselines/travelTrajectory/DNN/main.py'
            os.system(run)
        else:
            run = 'python ./baselines/travelTrajectory/DNN/main.py --test'
            os.system(run)
    elif model == 'Transformer':
        if not args.test:  # training
            run = 'python ./baselines/travelTrajectory/Transformer/main.py'
            os.system(run)
        else:
            run = 'python ./baselines/travelTrajectory/Transformer/main.py --test'
            os.system(run)
    elif model == 'RF':
        run = 'python ./baselines/travelTrajectory/RF/main.py'
        os.system(run)


if __name__ == "__main__":
    main()