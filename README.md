# Audio Prediction Example

Demonstrates a simple audio prediction training loop.

## How-To

Requires conda: https://docs.conda.io/projects/miniconda/en/latest/

```
git clone https://github.com/catid/audio_regression
cd audio_regression

conda create -n audio python=3.10 -y && conda activate audio

pip install -r requirements.txt

python train.py
```

## Example Output

```
(audio) ➜  audio_regression git:(master) python train.py
Dataset shape: (36, 500, 12)
Epoch 100/1000, Train Loss: 0.011869, Val Loss: 0.012290:   9%|████▌                                             | 90/1000 [00:00<00:03, 242.98it/s]Epoch 100/1000, Train Loss: 0.011869, Val Loss: 0.012290
Epoch 200/1000, Train Loss: 0.008134, Val Loss: 0.008720:  17%|████████▍                                        | 172/1000 [00:00<00:03, 263.16it/s]Epoch 200/1000, Train Loss: 0.008134, Val Loss: 0.008720
Epoch 300/1000, Train Loss: 0.007304, Val Loss: 0.007716:  28%|█████████████▊                                   | 282/1000 [00:01<00:02, 266.66it/s]Epoch 300/1000, Train Loss: 0.007304, Val Loss: 0.007716
Epoch 400/1000, Train Loss: 0.006708, Val Loss: 0.007312:  39%|███████████████████▏                             | 392/1000 [00:01<00:02, 256.16it/s]Epoch 400/1000, Train Loss: 0.006708, Val Loss: 0.007312
Epoch 500/1000, Train Loss: 0.006814, Val Loss: 0.007051:  50%|████████████████████████▎                        | 496/1000 [00:01<00:01, 253.65it/s]Epoch 500/1000, Train Loss: 0.006814, Val Loss: 0.007051
Epoch 600/1000, Train Loss: 0.006822, Val Loss: 0.006885:  57%|████████████████████████████▏                    | 574/1000 [00:02<00:01, 252.63it/s]Epoch 600/1000, Train Loss: 0.006822, Val Loss: 0.006885
Epoch 700/1000, Train Loss: 0.006370, Val Loss: 0.006737:  68%|█████████████████████████████████▏               | 678/1000 [00:02<00:01, 254.69it/s]Epoch 700/1000, Train Loss: 0.006370, Val Loss: 0.006737
Epoch 800/1000, Train Loss: 0.006186, Val Loss: 0.006666:  78%|██████████████████████████████████████▎          | 782/1000 [00:03<00:00, 247.05it/s]Epoch 800/1000, Train Loss: 0.006186, Val Loss: 0.006666
Epoch 900/1000, Train Loss: 0.006159, Val Loss: 0.006545:  88%|███████████████████████████████████████████▎     | 883/1000 [00:03<00:00, 240.08it/s]Epoch 900/1000, Train Loss: 0.006159, Val Loss: 0.006545
Epoch 1000/1000, Train Loss: 0.006273, Val Loss: 0.006535:  99%|███████████████████████████████████████████████▍| 987/1000 [00:03<00:00, 252.62it/s]Epoch 1000/1000, Train Loss: 0.006273, Val Loss: 0.006535
Epoch 1000/1000, Train Loss: 0.006273, Val Loss: 0.006535: 100%|███████████████████████████████████████████████| 1000/1000 [00:03<00:00, 250.88it/s]
Epoch 1000/1000, Train Loss: 0.006273, Val Loss: 0.006535
```
