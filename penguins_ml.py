import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


penguin_df = pd.read_csv('penguins.csv')
penguin_df.dropna(inplace=True)
output = penguin_df['species']
features = penguin_df[['bill_length_mm','bill_depth_mm','flipper_length_mm','body_mass_g','island','sex']]
features = pd.get_dummies(features)
output,uniques = pd.factorize(output)
x_train,x_test,y_train,y_test =train_test_split(features,output,test_size=0.8)

rfc = RandomForestClassifier(random_state=15)
rfc.fit(x_train,y_train)
y_pred=rfc.predict(x_test)
score = accuracy_score(y_test,y_pred)
print(f'the accuracy score for this model is: {score:.4f}%')
rf_pickle = open('random_forest_penguin.pickle','wb')
pickle.dump(rfc,rf_pickle)
rf_pickle.close()
output_pickle = open('output_penguin.pickle','wb')
pickle.dump(uniques,output_pickle)
output_pickle.close()
fig,ax = plt.subplots()
ax = sns.barplot(x=rfc.feature_importances_,y=features.columns)
plt.title('which features are most important for species prediction')
plt.xlabel('importance')
plt.ylabel('Feature')
plt.tight_layout()
fig.savefig('feature_importance.png')


rf_pickle = open('random_forest_penguin.pickle','rb')
map_pickle = open('output_penguin.pickle','rb')

rfc= pickle.load(rf_pickle)
unique_penguin_mapping = pickle.load(map_pickle)
rf_pickle.close()
map_pickle.close()


with st.form('user_inputs'):
    island= st.selectbox('Penguin Island',['Biscoe','Dream','Torgerson'])
    sex = st.selectbox('Sex',['Female','Male'])
    bill_length = st.number_input('Bill Length(mm)',min_value=0)
    bill_depth = st.number_input('Bill Depth(mm)',min_value=0)
    flipper_length=st.number_input('Flipper Length(mm)',min_value=0)
    body_mass = st.number_input('Body mass(g)',min_value=0)
    st.form_submit_button()
    island_biscoe,island_dream,island_torgerson=0,0,0
    if island == 'Biscoe':
        island_biscoe=1
    elif island=='Dream':
        island_dream=1
    elif island=='Torgerson':
        island_torgerson=1
    sex_female,sex_male=0,0
    if sex=='Female':
        sex_female=1
    elif sex=='Male':
        sex_male=1

    new_predictions = rfc.predict([[
        bill_length,
        bill_depth,
        flipper_length,
        body_mass,
        island_biscoe,
        island_dream,
        island_torgerson,
        sex_male,
        sex_female
    ]])
    prediction_species = unique_penguin_mapping[new_predictions][0]

    st.subheader('predicting your penguins species: ')
    st.write(f"We predict your penguin is of the {prediction_species} species")

st.image('feature_importance.png')

fig,ax = plt.subplots()
ax = sns.displot(x=penguin_df['bill_length_mm'],hue=penguin_df['species'])
plt.axvline(bill_length)
plt.title('bill length by species')
st.pyplot(ax)

fig,ax = plt.subplots()
ax = sns.displot(x=penguin_df['bill_depth_mm'],hue=penguin_df['species'])
plt.axvline(bill_depth)
plt.title('Bill Depth by Species')
st.pyplot(ax)
fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df['flipper_length_mm'],hue=penguin_df['species'])
plt.axvline(flipper_length)
plt.title('Flipper Length by Species')
st.pyplot(ax)
