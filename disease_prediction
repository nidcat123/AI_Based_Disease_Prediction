# import numpy as np
from flask import Flask, request, jsonify, render_template, redirect
import pickle
import os
import numpy as np
import pandas as pd
#create flask name
app = Flask( name )
picFolder = os.path.join('static','builtin')
app.config['UPLOAD_FOLDER'] = picFolder
#Load the pickle model
#model = pickle.load(open('model.pkl','rb'))
# for i in range(0,len(l1)):
# l2.append(0)
df = pd.read_csv("training.csv")
DF = pd.read_csv('training.csv', index_col='prognosis')
# Replace the values in the imported file by pandas by the inbuilt function
replace in pandas.
df.replace({'prognosis': {'Fungal infection': 0, 'Allergy': 1, 'GERD': 2,
'Chronic cholestasis': 3, 'Drug Reaction': 4,
'Peptic ulcer diseae': 5, 'AIDS': 6, 'Diabetes ':
7, 'Gastroenteritis': 8,
'Bronchial Asthma': 9, 'Hypertension ': 10,
'Migraine': 11, 'Cervical spondylosis': 12,
'Paralysis (brain hemorrhage)': 13, 'Jaundice':
14, 'Malaria': 15, 'Chicken pox': 16,
'Dengue': 17, 'Typhoid': 18, 'hepatitis A': 19,
'Hepatitis B': 20, 'Hepatitis C': 21, 'Hepatitis
D': 22, 'Hepatitis E': 23,
'Alcoholic hepatitis': 24, 'Tuberculosis': 25,
'Common Cold': 26, 'Pneumonia': 27, 'Dimorphic
hemmorhoids(piles)': 28, 'Heart attack': 29,
'Varicose veins': 30, 'Hypothyroidism': 31,
'Hyperthyroidism': 32, 'Hypoglycemia': 33,
'Osteoarthristis': 34, 'Arthritis': 35,
'(vertigo) Paroymsal Positional Vertigo': 36,
'Acne': 37, 'Urinary tract infection': 38,
'Psoriasis': 39,
'Impetigo': 40}}, inplace=True)
l1=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','ye
llow_urine',
'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stom
ach',
'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','thr
oat_irritation',
15
'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','w
eakness_in_limbs',
'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','blood
y_stool',
'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','
swollen_legs',
'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_na
ils',
'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_an
d_tingling_lips',
'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck'
,'swelling_joints',
'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell
_of urine',
'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look
_(typhos)',
'depression','irritability','muscle_pain','altered_sensorium','red_spots_ove
r_body','belly_pain',
'abnormal_menstruation','dischromic
_patches','watering_from_eyes','increased_appetite','polyuria','family_histo
ry','mucoid_sputum',
'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_bloo
d_transfusion',
'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_ab
domen',
'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','promine
nt_veins_on_calf',
'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring
','skin_peeling',
'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister',
'red_sore_around_nose',
'yellow_crust_ooze']
disease=['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis',
'Drug Reaction', 'Peptic ulcer diseae', 'AIDS', 'Diabetes ',
'Gastroenteritis', 'Bronchial Asthma', 'Hypertension ', 'Migraine',
'Cervical spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice',
'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A',
'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E',
'Alcoholic hepatitis', 'Tuberculosis', 'Common Cold', 'Pneumonia',
'Dimorphic hemmorhoids(piles)', 'Heart attack', 'Varicose veins',
'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia',
'Osteoarthristis', 'Arthritis',
'(vertigo) Paroymsal Positional Vertigo', 'Acne',
'Urinary tract infection', 'Psoriasis', 'Impetigo']
l2=[]
X = df[l1]
y = df[["prognosis"]]
16
np.ravel(y)
# Reading the testing.csv file
tr = pd.read_csv("testing.csv")
# Using inbuilt function replace in pandas for replacing the values
tr.replace({'prognosis': {'Fungal infection': 0, 'Drug Reaction': 1, 'GERD':
2, 'Chronic cholestasis': 3, 'Allergy': 4,
'Peptic ulcer diseae': 5, 'AIDS': 6, 'Diabetes ':
7, 'Gastroenteritis': 8,
'Bronchial Asthma': 9, 'Hypertension ': 10,
'Migraine': 11, 'Cervical spondylosis': 12,
'Paralysis (brain hemorrhage)': 13, 'Jaundice':
14, 'Malaria': 15, 'Chicken pox': 16,
'Dengue': 17, 'Typhoid': 18, 'hepatitis A': 19,
'Hepatitis B': 20, 'Hepatitis C': 21, 'Hepatitis
D': 22, 'Hepatitis E': 23,
'Alcoholic hepatitis': 24, 'Tuberculosis': 25,
'Common Cold': 26, 'Pneumonia': 27, 'Dimorphic
hemmorhoids(piles)': 28, 'Heart attack': 29,
'Varicose veins': 30, 'Hypothyroidism': 31,
'Hyperthyroidism': 32, 'Hypoglycemia': 33,
'Osteoarthristis': 34, 'Arthritis': 35,
'(vertigo) Paroymsal Positional Vertigo': 36,
'Acne': 37, 'Urinary tract infection': 38,
'Psoriasis': 39,
'Impetigo': 40}}, inplace=True)
X_test = tr[l1]
y_test = tr[["prognosis"]]
np.ravel(y_test)
# Decision Tree classifier
# def DecisionTree():
from sklearn import tree
clf3 = tree.DecisionTreeClassifier()
clf3 = clf3.fit(X, y)
# y_pred = clf3.predict(X_test)
# print(y_pred)
pickle.dump(clf3, open("model.pkl","wb"))
@app.route("/")
def Home():
pic1 = os.path.join(app.config['UPLOAD_FOLDER'],'img1.webp')
return render_template("index.html", user_image = pic1)
@app.route("/predict", methods =["GET","POST"])
def predict():
option1 = str(request.args.get('options1'))
option2 = str(request.args.get('options2'))
option3 = str(request.args.get('options3'))
pic3 = os.path.join(app.config['UPLOAD_FOLDER'], 'img1.webp')
if option1 == option2:
return render_template("index.html",user_image = pic3)
elif option2 == option3:
return render_template("index.html", user_image = pic3)
elif option1 == option3:
return render_template("index.html", user_image = pic3)
elif option1 == option2 and option2 == option3:
return render_template("index.html", user_image = pic3)
17
# return redirect(url_for('index'))
# arr = np.array([option1,option2,option3]).reshape(1,-1)
# print(option1,option2,option3)
for k in range(0, len(l1)):
if (l1[k] == option1):
l2.append(1)
elif (l1[k] == option2):
l2.append(1)
elif (l1[k] == option3):
l2.append(1)
else:
l2.append(0)
# print(l2)
inputtest = [l2]
predict = clf3.predict(inputtest)
predicted = predict[0]
pred1=""
h = 'no'
for a in range(0, len(disease)):
if (predicted == a):
h = 'yes'
break
if (h == 'yes'):
pred1 = disease[a]
else:
pred1 = "Not Found"
option1 = ""
option2 = ""
option3 = ""
pic2 = os.path.join(app.config['UPLOAD_FOLDER'], 'img1.webp')
return render_template("after.html",data=pred1,user_image = pic2)
if name == " main ":
app.run(port=1000)