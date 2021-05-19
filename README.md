Sean Healy
Chris Shafer

How to run:

training the model can be done by running "python train.py"

How to use:

First, if the model is not present, download the model from the link provided in the model.txt file or train your own (will take up to 8 hours).
Next, find the question you want to ask (examples in questionFinal.txt).
Then find the associated line in the contextFinal.txt file for the data you are looking for.
Then run "python interact.py -q "<question here>" -n <line number here>

There is also a smaller test in the directory "smallTest" where the data was shrunk to be more uniform in format. This test was to show how on a small scale, the system can work very well. Run this the same way.
