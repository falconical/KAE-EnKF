# KAE EnKF

Note: this repository is meant primarily as a code accompaniment to the paper introducing the KAE EnKF.

## Installation

Installation can be performed as follows. 

First clone the repository by running:

`git clone https://github.com/falconical/KAE-EnKF.git`

The code can then be run, either from an existing Python environment with requisite packages installed as needed, which is recommended.

To enable absolute reproducibility, Python version 3.7.6 was used, and all packages in the environment the results were generated in are saved to a requirements.txt file, however many of these packages are surplus to requirment, hence installing in this was is not recommended.

## Example Usage

With all the code and dependencies installed, and can run the KAE EnKF example notebook in your freshly activated Python environment via the command:

`jupyter notebook`

and then clicking on the notebook titled "KAE EnKF example".

This notebook guides you through what an application of the KAE EnKF could look like on a simple synthetic dataset.

Note: The pendulum scripts use video data from https://www.youtube.com/watch?v=MpzaCCbX-z4, which is not included in this repo.

## Authors and acknowledgment
#### Authors:
Stephen A Falconer, David J.B. Lloyd, and Naratip Santitissadeekorn

Department of Mathematics, University of Surrey, Guildford, GU2 7XH, UK

#### Acknowledgments
This work was supported by the UKRI, whose Doctoral Training Partnership Studentship helped fund Stephen Falconers PhD. He would also like to thank Stefan Klus, for providing the pendulum video data used in this work, as well as useful insights. Also thanks for their valuable discussions go out to Nadia Smith and Spencer Thomas from the National Physics Laboratory.
