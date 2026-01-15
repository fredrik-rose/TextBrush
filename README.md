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

## Datasets

The datasets used are Tiny Shakespeare and MNIST.

## Relevant Papers

- Attention Is All You Need: https://arxiv.org/abs/1706.03762
- An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT): https://arxiv.org/abs/2010.11929
- Language Models are Few-Shot Learners: https://arxiv.org/abs/2005.14165
- Deep Residual Learning for Image Recognition (ResNet): https://arxiv.org/abs/1512.03385
- Layer Normalization: https://arxiv.org/abs/1607.06450
- Dropout: https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf
