import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Loading datasets
food_data = pd.read_csv('/cleaned_food_data.csv')
exercise_data = pd.read_csv('/cleaned_exercise_data.csv')
user_data = pd.read_excel('/cleaned_user_data.xlsx')
ethnicity_data = pd.read_excel('/cleaned_ethnicity_data.xlsx')

# Preprocessing User Data
user_data['gender'] = user_data['gender'].map({'M': 1, 'F': 0})
user_data['activity_level'] = user_data['activity_level'].map({'Low': 1, 'Moderate': 2, 'Active': 3})
user_data['health_goal'] = user_data['health_goal'].map({'Weight Loss': -500, 'Maintenance': 0, 'Muscle Gain': 500})

# Map ethnicity death rates to user data
def map_ethnicity_death_rate(user_row, ethnicity_data):
    ethnicity = user_row['ethnicity']
    if ethnicity not in ethnicity_data.columns:
        return 0  # Default to 0 if ethnicity is not in dataset
    return ethnicity_data.loc[ethnicity_data['Cause of Death'] == 'All causes', ethnicity].values[0]

user_data['ethnicity_death_rate'] = user_data.apply(map_ethnicity_death_rate, axis=1, ethnicity_data=ethnicity_data)

# prediction for Caloric Needs
caloric_data = user_data[['age', 'gender', 'weight', 'activity_level', 'health_goal', 'height_cm']].copy()
caloric_data.fillna(caloric_data.median(), inplace=True)

X_caloric = caloric_data[['age', 'gender', 'weight', 'activity_level', 'height_cm']]
y_caloric = 10 * caloric_data['weight'] + 6.25 * caloric_data['height_cm'] - 5 * caloric_data['age'] + 5
y_caloric += caloric_data['health_goal']

X_caloric_train, X_caloric_test, y_caloric_train, y_caloric_test = train_test_split(X_caloric, y_caloric, test_size=0.2, random_state=42)
caloric_model = RandomForestRegressor()
caloric_model.fit(X_caloric_train, y_caloric_train)

# Health Risk Classification
risk_data = user_data[['activity_level', 'genetic_predispositions', 'chronic_conditions', 'ethnicity_death_rate']].copy()
risk_data['activity_level'] = risk_data['activity_level'].map({'Low': 1, 'Moderate': 2, 'Active': 3})
risk_data = pd.get_dummies(risk_data, columns=['genetic_predispositions', 'chronic_conditions'])
risk_data.fillna(0, inplace=True)

X_risk = risk_data
y_risk = user_data['stress_level'].map({'Low': 0, 'Moderate': 1, 'High': 2})

X_risk_train, X_risk_test, y_risk_train, y_risk_test = train_test_split(X_risk, y_risk, test_size=0.2, random_state=42)
risk_model = GradientBoostingClassifier()
risk_model.fit(X_risk_train, y_risk_train)

# Train Meal Recommendation Model
def train_meal_recommendation_model():
    meal_data = food_data.copy()
    meal_data['is_recommended'] = 1
    X_meal = meal_data[['calories(kcal)', 'vegan', 'gluten_free', 'keto']]
    y_meal = meal_data['is_recommended']
    X_meal = pd.get_dummies(X_meal, drop_first=True)

    meal_model = DecisionTreeClassifier()
    meal_model.fit(X_meal, y_meal)
    return meal_model

meal_model = train_meal_recommendation_model()

# Recommend Meals Using Strict Filtering
def recommend_meals_with_model(user_preferences):
    user_preferences.setdefault('vegan', False)
    user_preferences.setdefault('vegetarian', False)
    user_preferences.setdefault('gluten_free', False)
    user_preferences.setdefault('keto', False)

    meal_features = food_data[['calories(kcal)', 'vegan', 'gluten_free', 'keto']]
    meal_features = pd.get_dummies(meal_features, drop_first=True)

    food_data['is_recommended'] = meal_model.predict(meal_features)
    recommended_meals = food_data[food_data['is_recommended'] == 1]

    # Apply strict filtering for dietary preferences
    if user_preferences['vegan']:
        recommended_meals = recommended_meals[recommended_meals['vegan'] == 'Yes']
    if user_preferences['vegetarian']:
        non_vegetarian_keywords = ['pork', 'beef', 'chicken', 'fish', 'turkey', 'sausage']
        recommended_meals = recommended_meals[
            ~recommended_meals['name'].str.contains('|'.join(non_vegetarian_keywords), case=False, na=False)
        ]
    if user_preferences['gluten_free']:
        recommended_meals = recommended_meals[recommended_meals['gluten_free'] == 'Yes']
    if user_preferences['keto']:
        recommended_meals = recommended_meals[recommended_meals['keto'] == 'Yes']

    return recommended_meals.sample(3) if not recommended_meals.empty else pd.DataFrame()

# Recommending Exercises Using the Model
def recommend_exercises_with_model(user):
    exercise_data['is_recommended'] = 1  # Placeholder
    recommended_exercises = exercise_data[exercise_data['Calories per kg'] <= 2.5]  # Example condition
    return recommended_exercises.sample(3) if not recommended_exercises.empty else pd.DataFrame()

# Plan for a single user
def generate_plan(user):
    caloric_needs = caloric_model.predict(pd.DataFrame([[user['age'], user['gender'], user['weight'],
                                                         user['activity_level'], user['height_cm']]],
                                                       columns=X_caloric.columns))[0]

    risk_input = {
        'activity_level': user['activity_level'],
        'ethnicity_death_rate': user['ethnicity_death_rate'],
        'genetic_predispositions_' + str(user.get('genetic_predispositions', 'None')): 1,
        'chronic_conditions_' + str(user.get('chronic_conditions', 'None')): 1,
    }
    aligned_risk_input = {col: risk_input.get(col, 0) for col in X_risk.columns}
    risk_input_data = pd.DataFrame([aligned_risk_input])
    risk_level = risk_model.predict(risk_input_data)[0]

    meals = recommend_meals_with_model(user)
    exercises = recommend_exercises_with_model(user)

    output = f"""
    Caloric Needs: {caloric_needs:.2f} kcal (adjusted for user’s weight, activity level, and goal).
    Risk Level: {"High" if risk_level == 2 else "Moderate" if risk_level == 1 else "Low"} (based on genetic predispositions, chronic conditions, and ethnicity death rates).
    """

    if not meals.empty:
        output += "Meals:\n"
        for i, meal in enumerate(meals.to_dict(orient='records'), start=1):
            output += f"    Option {i}: {meal['name']} ({meal['calories(kcal)']} kcal, vegan: {meal['vegan']}, gluten-free: {meal['gluten_free']}, keto: {meal['keto']}).\n"
    else:
        output += "Meals:\n    No suitable meals found.\n"

    if not exercises.empty:
        output += "Exercises:\n"
        for i, exercise in enumerate(exercises.to_dict(orient='records'), start=1):
            output += f"    Option {i}: {exercise['Activity']} (calories per kg: {exercise['Calories per kg']:.2f}).\n"
    else:
        output += "Exercises:\n    No suitable exercises found.\n"

    return output.strip()

def generate_plan_for_user_id(user_id, user_data):
    specific_user = user_data[user_data['id'] == user_id]
    if specific_user.empty:
        return f"No user found with ID: {user_id}"
    user_dict = specific_user.iloc[0].to_dict()
    return generate_plan(user_dict)

# Testing for a specific user ID
user_id_to_test = "BHVH9"  # Replace with user ID from user_data file
print(generate_plan_for_user_id(user_id_to_test, user_data))
