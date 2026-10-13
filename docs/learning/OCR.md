## What is OCR?

Optical Character Recognition, or OCR, converts text represented as image
pixels into machine-readable characters.

## Common OCR recognition strategies

OCR is not one specific algorithm. Several model families can solve the recognition stage.

### Segmentation-based OCR

Older systems attempted to separate a line into individual characters and classify each other independently.

This approach can struggle when characters touch, spacing is irregular, or the image is blurred.

### CNN/RNN with CTC

A convolutional network extracts visual features across the text line. A sequence model processes those features, a Connectionist Temporal Classification (CTC) learns how to align them with characters without requiring a bound box for every character.

CTC recognizers are often fast and effective for line-level text. Their decoding is generally less linguistically generative than an encoder-decoder model.

### Attention-based encoder-decoder OCR

And encoder converts the image into contextual visual representations. A decoder then generates text tokens while attending to those representations.

TrOCR belongs to this family.

### OCR-free document understanding

Models such as Donut consume an entire document an directly generate structured output. They avoid an explicit intermediate OCR stage.


## How TrOCR works

TrOCR means Transformer-based Optical Character Recognition. It combines a vision encoder with an autoregressive text decoder.

### 1. Image Processing
The processor resizes, rescales, and normalizes the RGB crop. It converts the image into a PyTorch tensor with the numerical format expected by the model.

### 2. Image patches
The processed image is divided into fixed-size patches. Each patch is converted into a vector representations.

Instead of initially seeing individual characters, the model sees a sequence of visual regions.

### 3. Vision encoder

The Transformer encoder allows the image patches to exchange contextual information. A patch representing part of a character can therefore be interpreted using surrounding patches.

The encoder produces a sequence of contextual visual representations.

### 4. Autoregressive decoder

The decoder being with a special start token. It predicts the probability of the first text token using the encoded image.

After selecting that token, it predicts the next token using:
- The encoded image;
- The tokens already generated
This continues until the decoder generates an end token or reaches a configured length limit.


### 5. Token Decoding
The model produces token IDs, not a python string. The tokenizer maps those IDs back into readable text.

The processor conveniently contains both the image processor and tokenizer, but the processor is not the neural model itself.


### Key consequence of autoregressive decoding

A decoder-generated token influences later tokens. This can be helpful because language context can resolve ambiguous characters. It can also produce plausible-looking text that is not actually present in the image.

That is why we must preserve raw predictions and compare strictly with ground truth,


## Models and Checkpoints

A model architecture describes the computational structure: layers, hidden
sizes, attention mechanisms, and connections.

A checkpoint contains learned parameter values for that architecture.

Two checkpoints can use the same architecture but behave differently because
they were trained on different data or tasks.

Using a pretrained checkpoint means starting with parameters already learned
from another dataset rather than initializing the model randomly.
