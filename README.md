# Predicting NFL Games 🏈

__Interact with this model and view historical performance data for your favorite team <a href="http://promatchpredict.com/" target="_blank" rel="noopener noreferrer">here</a>__

I wanted to combine both my love of sports and love of data science, and what better way than to try predict which NFL team will cover the spread.

I pulled both historical and current data from [nfl_data_py](https://github.com/nflverse/nfl_data_py), and trained a random forest classifier model to predict which team will cover the spread. The model currently sits in a Docker container within an EC2 isntance, and you can view the predictions for your upcoming team by checking the link above!

The training data is available in this folder under 'nfl_training_data.csv'. 

Feel free to reach out with any suggestions to improve my model, or other inputs I should consider
