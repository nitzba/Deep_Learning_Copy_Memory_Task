# Copy Memory Task 📝

### About the project 📖

---

This project is build out 2 projects. Self implementation of LSRMcell, RNNcell and Linear layers. Self implementation of the copy memory task.<br>
The project is a university assignment in the course of mini-project in deep learning using PyTorch.<br><br>

**LSTM.py** - self implementation of lstm cell<br>
**RNN.py** - self implementation of rnn cell<br>
**MLP.py** - self implementation of linear<br>
**copy_task_base.py** - self implementation of the copy task with 3 different models<br><br>

### Setup ⚙️

---

- Make sure you have python 3.6 or above on your computer.<br>

- Clone this repository or download and extract the zip of this repository (make sure you have all the files).

- Open up terminal in the project directory.<br>In order to setup the relevant libraries for the project use the give **_requirements_script.txt_** file by running:<br>

  ```
  pip install -r requirements_script.txt
  ```

- Run this command to train the model:
  ```
  python copy_task_base.py -t [number_of_blanks] -k [number_of_numbers_to_copy] -m [model_name]
  ```

  **number_of_blanks** - choose how deep is your network will need to remember the numbers to copy.
  **number_of_numbers_to_copy** - choose how many number you will need to copy after the delimiter.
  **model_name** - choose which model do you want to run: RNN, LSTM or MLP.
