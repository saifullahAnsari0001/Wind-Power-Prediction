# Wind-Power-Prediction

```markdown
# Wind Turbine Data Analysis and Machine Learning Model Implementation

This project focuses on the analysis of wind turbine data and the implementation of various machine learning models to predict power production. The dataset used contains information about wind speed, wind direction, theoretical power, and active power.

## Project Structure

- **Data Analysis and Visualization**: Initial exploration and visualization of the dataset to understand its characteristics and identify any missing or anomalous data.
- **Data Cleaning and Imputation**: Addressing missing timestamps and null values using the K-Nearest Neighbors (KNN) imputer.
- **Feature Engineering**: Extracting useful features such as month, day, hour, and minute from the timestamp for further analysis.
- **Machine Learning Models**: Implementing and evaluating various machine learning models including Random Forest Regressor, XGBoost Regressor, and Support Vector Regressor (SVR).
- **Model Evaluation**: Comparing the performance of the models using metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (R²).
- **Visualization of Results**: Visualizing the actual vs. predicted power production using scatter plots.

## Dataset

The dataset `T1.csv` contains the following columns:
- `Date/Time`: Timestamp of the data entry.
- `LV ActivePower (kW)`: Active power produced by the turbine in kW.
- `Wind Speed (m/s)`: Wind speed in meters per second.
- `Theoretical_Power_Curve (KWh)`: Theoretical power production in KWh.
- `Wind Direction (°)`: Wind direction in degrees.

## Libraries Used

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- plotly

## Steps to Run the Project

1. **Install the required libraries**:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost plotly
   ```

2. **Load the dataset**:
   ```python
   df = pd.read_csv('T1.csv')
   ```

3. **Preprocess the data**:
   - Convert the `Date/Time` column to datetime format.
   - Set the `Date/Time` column as the index.
   - Rename the columns for easier reference.
   - Handle missing timestamps and null values using KNN imputer.

4. **Visualize the data**:
   - Plot histograms for wind speed, wind direction, theoretical power, and active power.
   - Plot a heatmap to check for missing values.
   - Plot monthly average power production.
   - Plot correlation heatmap and pairplot.

5. **Clean the data**:
   - Identify and replace outliers and physically implausible values (e.g., negative power values) with NaN.
   - Impute the NaN values using KNN imputer.

6. **Train and evaluate machine learning models**:
   - Split the dataset into training and testing sets.
   - Normalize the features using MinMaxScaler.
   - Train and evaluate the Random Forest Regressor, XGBoost Regressor, and SVR models.
   - Compare the performance of the models using MAE, RMSE, and R².

7. **Visualize the results**:
   - Create scatter plots to compare the actual vs. predicted power production for each model.

## Results

The results of the models are summarized in the following table:

| Model                  | Data Set | MAE       | RMSE      | R²       |
|------------------------|----------|-----------|-----------|----------|
| RandomForestRegressor  | Train    | 68.378137 | 110.717079| 0.992462 |
| RandomForestRegressor  | Test     | 69.248295 | 112.217068| 0.992348 |
| XGBRegressor           | Train    | 15.994104 | 31.54174  | 0.999388 |
| XGBRegressor           | Test     | 18.740818 | 41.409775 | 0.998958 |
| SVR                    | Train    | 44.06491  | 84.30114  | 0.99563  |
| SVR                    | Test     | 43.681587 | 83.300693 | 0.995784 |

Scatter plots for actual vs. predicted power production are also included for visual comparison.

## Conclusion

This project demonstrates the process of analyzing wind turbine data, handling missing and anomalous values, and implementing machine learning models to predict power production. The Random Forest Regressor, XGBoost Regressor, and SVR models were evaluated, and their performance was compared to identify the best model for this task.

## Author

Md Saifullah Ansari
- [LinkedIn](https://linkedin.com/in/md-saifullah-ansari-613405309)
