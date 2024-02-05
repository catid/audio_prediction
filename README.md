# Audio Prediction Example

Demonstrates a simple audio prediction training loop.

I think people working on biologically plausible artificial neurons should start on this toy problem.  Basically the brain is built up of predictor units, so an artificial neuron should be able to do sequence prediction.  If you're using standard things like RNN or transformer decoders then fine, but if you want to build something to replace them then it should be able to do well on this kind of simple test.  And since it runs very fast on a laptop you can evaluate ideas rapidly.

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
(audio) ➜  audio_regression git:(master) ✗ python train.py
Dataset shape: (36, 500, 12)
Epoch 100/1000, Train Loss: 0.011148, Val Loss: 0.014608:   8%|█████                                                        | 84/1000 [00:00<00:04, 226.01it/s]

Epoch 200/1000, Train Loss: 0.007758, Val Loss: 0.010048:  19%|███████████▏                                                | 186/1000 [00:00<00:03, 246.54it/s]

Epoch 300/1000, Train Loss: 0.006816, Val Loss: 0.008765:  29%|█████████████████▍                                          | 290/1000 [00:01<00:02, 251.08it/s]

Epoch 400/1000, Train Loss: 0.006459, Val Loss: 0.008250:  37%|██████████████████████▎                                     | 372/1000 [00:01<00:02, 261.74it/s]

Epoch 500/1000, Train Loss: 0.006453, Val Loss: 0.007986:  48%|████████████████████████████▉                               | 482/1000 [00:02<00:01, 262.66it/s]

Epoch 600/1000, Train Loss: 0.005863, Val Loss: 0.007812:  59%|███████████████████████████████████▎                        | 589/1000 [00:02<00:01, 257.52it/s]

Epoch 700/1000, Train Loss: 0.005967, Val Loss: 0.007644:  69%|█████████████████████████████████████████▌                  | 693/1000 [00:02<00:01, 247.46it/s]

Epoch 800/1000, Train Loss: 0.005847, Val Loss: 0.007552:  78%|██████████████████████████████████████████████▌             | 775/1000 [00:03<00:00, 258.82it/s]

Epoch 900/1000, Train Loss: 0.005905, Val Loss: 0.007571:  88%|████████████████████████████████████████████████████▋       | 879/1000 [00:03<00:00, 253.67it/s]

Epoch 1000/1000, Train Loss: 0.005661, Val Loss: 0.007447:  99%|██████████████████████████████████████████████████████████▏| 986/1000 [00:03<00:00, 263.55it/s]

Epoch 1000/1000, Train Loss: 0.005661, Val Loss: 0.007447: 100%|██████████████████████████████████████████████████████████| 1000/1000 [00:03<00:00, 251.54it/s]

Final Epoch 1000/1000, Train Loss: 0.005661, Val Loss: 0.007447
```
