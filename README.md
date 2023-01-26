# Learning knot invariants across dimensions
Code to accompany the paper "Learning knot invariants across dimensions". Provides the ability to learn the s-invariant and the slice genus from the Khovanov and Jones polynomials. 

## Setup
Download the files and extract the data files into your working directory. <br>
You will need Python, Tensorflow, and Numpy installed. <br>

## Usage
Run the program and follow the prompts to make a selection of inputs and outputs. <br>
Change the neural net's hyperparameters on lines 8-13 to see how a different setup affects the results. <br>
The trained model is returned from the learning function on line 144. You could use this to do validation testing, try some new data, or any other types of experments.

## Citation
Cite as <br>
@article{craven2021learning, <br>
  title={Learning knot invariants across dimensions}, <br>
  author={Craven, Jessica and Hughes, Mark and Jejjala, Vishnu and Kar, Arjun}, <br>
  journal={arXiv preprint arXiv:2112.00016}, <br>
  year={2021} <br>
}
