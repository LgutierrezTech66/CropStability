# CropStability
<br/>
<br/>
Goal:
<br/>
Creating a machine learning model that predicts whether a crop is stable or unstable based on factors like rainfall, temperature, soil PH, and crop type, and presenting via a streamlit web interface.
<br/>
<br/>
Training and Saving the Model:
<br/>

Libraries:
  * Pandas: for handling data
  * sklearn: for training the model
  * pickle: for saving/loading the model
  * os: ensures the model folder exists

<br/>

Dataset:
* Create a simple table of crop data to train the model.

<br/>

Model Training:
* X is the input (Feature)
* Y is the output label (stability)
* Splits the data set into training (80%) and testing (20%) parts.
* Training a Random Forest classifier using your features and labels.
 
<br/>

Saving the Model:
* pickle.dump(model,f) - saves the trained model to disk in the model folder using the pickle libraries.

<br/>

The Streamlit Web App:
* Load Model:
   * pickle.load(f) loads the saved model so it can be used to make predictions
* User Input interface:
   * List of features uses streamlit to let the user enter feature values via form inputs.
 
<br/>

How to run the project:
* In the command line:
   * Train the model first: python train_model.py
   * Run the streamlit app: streamlit run appy.py
 

