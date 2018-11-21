# INF5860 Oblig 1

In this exercise, you should implement a KNN-classifier and a Softmax-classifier in the style of a neural net, using the softmax score function, the cross entropy loss function and updating weights by using stochastic gradient descent and taking in a step in the direction of the negative gradient of the loss function with respect to the weight. 

Start with the knn. The exercise is described in the notebooks, which will ask you to fill in the missing code both in the notebook and files in subfolders. 

**Currently we recommend working at  linux-machines at IFI, starting jupyter notebooks from /opt/ifi/anaconda2/bin/jupyter-notebook or using your own installation of python with anaconda. Currently there are some issues with packages at jupyterhub and loading CIFAR will make the kernel stop(we are working on a solution). **

You might want to start by making a copy of the original code, in case you overwrite the original code. 
Use the "make a copy"  option of each notebook. 

Jupyter uses autosave, but it is a good idea to press "Save and checkpoint" regularly during your work. You can then revert to the checkpoint. 


##The Parts to Complete
#### Part 1: Implement a knn classifier and use cross-validation to find the best value of k.
The Jupyter Notebook knn.ipynb will walk you through implementing the KNN classifier.



#### Part 2: Implement a softmax classifier as a multiclass extension to logistic regression.
The Jupyter notebook softmax.ipynb will walk you through the implementation of the softmax classifier.

### Download data:
Once you have the starter code, you will need to download the CIFAR-10 dataset. Run the following from the main directory of the code:

On linux:
cd inf5860/datasets
./get_datasets.sh
 
If you are not on linux, you can download the CIFAR-10 from https://www.cs.toronto.edu/~kriz/cifar.html









By clicking on a notebook, you can start working on an assignment. Start with knn.ipynb.
You can run a cell with shift+enter or ctrl+enter. You can find more keyboard shortcuts here.

If you are unfamiliar with jupyter, you can test it out with try Jupyter. There you can find a simple overview in the Welcome to Python.ipynb notebook. To get a more extensive guide you can go to communities/pyladies/Python 101.ipynb.




Once you are done working, run the collectSubmission.sh script; this will produce a file called INF5860_Oblig1.zip.

Then upload the zip-file file to devilry (devilry.ifi.uio.no). You can make multiple submissions before the deadline.

Good luck!