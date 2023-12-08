# ByteRCNN: Enhancing File Fragment Type Identification with Recurrent and Convolutional Neural Networks

ByteRCNN is a novel machine learning model used to identify file fragment types.
Please find our corresponding paper at https://ieeexplore.ieee.org/document/10347203.  
ByteRCNN was traind using the ady-to-use ady-to-use open access datasets at [FFT-75](https://ieee-dataport.org/open-access/file-fragment-type-fft-75-dataset).

## Data
The data folder contains a script used to generate 512-byte fragments in csv format for training and evaluation based on the four datasets:

[2]   A. Khodadadi and M. Teimouri, “Dataset for file fragment classification of audio file formats,” BMC Res. Notes, vol. 12, no. 1, p. 819, Dec. 2019, doi: 10.1186/s13104-019-4856-1.

[3]   N. Sadeghi, M. Fahiminia, and M. Teimouri, “Dataset for file fragment classification of video file formats,” BMC Res. Notes, vol. 13, no. 1, p. 213, Apr. 2020, doi: 10.1186/s13104-020-05037-x.

[4]   F. Mansouri Hanis and M. Teimouri, “Dataset for file fragment classification of textual file formats,” BMC Res. Notes, vol. 12, no. 1, p. 801, Dec. 2019, doi: 10.1186/s13104-019-4837-4.

[5]   R. Fakouri and M. Teimouri, “Dataset for file fragment classification of image file formats,” BMC Res. Notes, vol. 12, no. 1, p. 774, Nov. 2019, doi: 10.1186/s13104-019-4812-0.
The folder also includes sub-sampled fragments obtained from each of those four datasets. They were sub-sampled since their original size is 1024 bytes.

# Citation

Please cite as:

```bibtex
  @ARTICLE{
    10347203,
    author={Skračić, Kristian and Petrović, Juraj and Pale, Predrag},
    journal={IEEE Access}, 
    title={ByteRCNN: Enhancing File Fragment Type Identification with Recurrent and Convolutional Neural Networks}, 
    year={2023},
    volume={},
    number={},
    pages={1-1},
    doi={10.1109/ACCESS.2023.3340441}
}
