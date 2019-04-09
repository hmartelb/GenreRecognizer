# GenreRecognizer

This project's aim is to build a Convolutional Neural Network to classify songs according to the music genre. The implementation of this network uses Keras with Tensorflow Backend.

## Getting Started


## Dataset

The dataset is composed by the top 50 songs of each genre (as per dataset/genres.txt) downloaded from SoundCloud. The songs are classified in folders according to the genre.

The input data of the network consists of 1 second audio frames, sliced from each song after applying amplitude normalization on the entire piece. The sampling rate of the audios is set to 44.1kHz, resampling any audio that originally came with a different sampling rate. 

```
At the time of writing these lines, the dataset has not been compiled yet.
```

## Network architecture

The network architecture from this work consists of the following stages:

### Input feature extraction
A time-frequency representation is extracted from the raw audio input by computing the spectrogram using Kapre. 

### N-Residual Convolutional blocks
### M-Dense layers

## Results

This section will display the results. This project is still under development so there are no conclusive results yet. 

## Usage
## License

This project is licensed under the MIT License.
```
MIT License

Copyright (c) 2019 Héctor Martel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
