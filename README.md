# Text Brush

A project that implements generative machine learning models for text and images. The focus is on handwritten digits
and the core architecture component is the transformer.

## Usage

This chapter contains example commands.

```
python main.py --help
```

### Train the Model

To train the model (neural network) of the application run:
```
python main.py --train <application>
```

To monitor the GPU run:
```
watch -n 1 nvidia-smi
```

### Generate Text

To generate text (1000 characters with prompt "QUEEN") run:
```
python main.py text -p "QUEEN" -n 1000
```

### Classify Images

To classify images (10 images) run:
```
python main.py image -n 10
```

### Visualize the Model

To train the model (neural network) of the application run:
```
python main.py --visualize-model <application>
```

### Run Static Code Analyzers

To check all linters run:
```
make lint
```

## Datasets

The datasets used are Tiny Shakespeare and MNIST.

## Relevant Papers

- Attention Is All You Need: https://arxiv.org/abs/1706.03762
- An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT): https://arxiv.org/abs/2010.11929
- Language Models are Few-Shot Learners (GPT): https://arxiv.org/abs/2005.14165
- Deep Residual Learning for Image Recognition (ResNet): https://arxiv.org/abs/1512.03385
- Gaussian Error Linear Units (GELUs): https://arxiv.org/abs/1606.08415
- Layer Normalization: https://arxiv.org/abs/1607.06450
- Dropout: https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf

## PyTorch

This project uses the PyTorch (Python version of Torch) machine learning framework. It is a very user friendly
framework that integrates well with the core Python principles. There are however some things to think about.

### Device

All tensors involved in the computation must be on the correct device (e.g. CPU or GPU). This must be handled manually
in many cases, e.g. when creating new tensors. This also applies to the models. Note that for a model you do
`model.to(device)` while for a tensor you need to `x = x.to(device)`.

### Reshape

There are several ways to reshape tensors in Torch, with benefits and drawbacks.

Shape

- reshape
- view
- flatten
- unflatten

Dimension decrease/increase

- squeeze
- unsqueeze

Dimension reorder

- transpose
- permute

Expansion

- expand
- repeat

Memory

- contiguous

### Broadcast

Be careful when relying on broadcast, often manual work (e.g. adding dimensions) is required for it to work out
properly.

### Optimization

Do not forget to zero the gradients for each step in the training `optimizer.zero_grad()`. It is also important to have
the model in the correct mode, `model.train()` when doing training and `model.eval()` for almost all other use cases,
e.g. inference.

Training loop functions often becomes very bloated with a huge parameter list. A neat approach is to implement the
training loop as a generator and only keeping the absolute minimum in this generator. This can then easily be extended
as needed. With this approach there are no need for callback functions, logic for storing the models, evaluation
handling, logic for early stopping, etc. All of this can be added outside as needed.

### Detach

To avoid huge memory leaks during training it is very important to detach the loss tensor from the compute graph. This
is done via `.item()` or `.detach()`. Code like this is very bad: `losses.append(loss)`.
