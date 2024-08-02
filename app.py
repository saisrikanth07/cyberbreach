# Importing necessary libraries
from flask import Flask, render_template, request, url_for, flash, redirect,session
import pandas as pd 
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import mysql.connector, re
db=mysql.connector.connect(user="root",password="",port='3306',database='cyber_attack')
cur=db.cursor()

app=Flask(__name__)
app.secret_key="CBJcb786874wrf78chdchsdcv"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/login',methods=['POST','GET'])
def login():
    if request.method=='POST':
        useremail=request.form['useremail']
        session['useremail']=useremail
        userpassword=request.form['userpassword']
        sql="select * from user where Email='%s' and Password='%s'"%(useremail,userpassword)
        cur.execute(sql)
        data=cur.fetchall()
        db.commit()
        if data ==[]:
            msg="user Credentials Are not valid"
            return render_template("login.html",name=msg)
        else:
            return render_template("home.html",myname=data[0][1])
    return render_template('login.html')

@app.route('/registration',methods=["POST","GET"])
def registration():
    if request.method=='POST':
        username=request.form['username']
        useremail = request.form['useremail']
        userpassword = request.form['userpassword']
        conpassword = request.form['conpassword']
        Age = request.form['Age']
        address = request.form['address']
        contact = request.form['contact']
        if userpassword == conpassword:
            sql="select * from user where Email='%s' and Password='%s'"%(useremail,userpassword)
            cur.execute(sql)
            data=cur.fetchall()
            db.commit()
            print(data)
            if data==[]:
                
                sql = "insert into user(Name,Email,Password,Age,Address,Contact)values(%s,%s,%s,%s,%s,%s)"
                val=(username,useremail,userpassword,Age,address,contact)
                cur.execute(sql,val)
                db.commit()
                flash("Registered successfully","success")
                return render_template("login.html")
            else:
                flash("Details are invalid","warning")
                return render_template("registration.html")
        else:
            msg = "Password doesn't match"
            return render_template("registration.html",msg=msg)
    return render_template('registration.html')


@app.route('/home') 
def home():
    return render_template('home.html')


@app.route('/load',methods=["GET","POST"])
def load():
    global df, dataset
    if request.method == "POST":
        data = request.files['data']
        df = pd.read_csv(data)
        dataset = df.head(100)
        msg = 'Data Loaded Successfully'
        return render_template('load.html', msg=msg)
    return render_template('load.html')


@app.route('/view')
def view():
    print(dataset)
    print(dataset.head(2))
    print(dataset.columns)
    return render_template('view.html', columns=dataset.columns.values, rows=dataset.values.tolist())


@app.route('/preprocess', methods=['POST', 'GET'])
def preprocess():
    global x, y, x_train, x_test, y_train, y_test,  hvectorizer,df
    if request.method == "POST":
        size = int(request.form['split'])
        size = size / 100
        dataset["Breach_Type"].replace({'Data Leak': 0, 'Malware': 1, 'SQL Injection': 2, 'Phishing': 3},inplace=True)
        dataset["Attack_Vector"].replace({'Physical': 0, 'Insider': 1, 'Email': 2, 'Network': 3, 'Web': 4},inplace=True)
        dataset["Vulnerability_Type"].replace({'Known': 0, 'Misconfiguration': 1, 'Zero-day': 2},inplace=True)
        dataset["Targeted_System"].replace({'Desktop': 1 ,'IoT': 3, 'Mobile': 0, 'Server': 2},inplace=True)
        dataset["Data_Exfiltrated"].replace({True: 1, False: 0},inplace=True)
        dataset["Attack_Successful"].replace({True: 1, False: 0},inplace=True)
        dataset["User_Notified"].replace({True: 1, False: 0},inplace=True)
        dataset["Legal_Action_Taken"].replace({True: 1, False: 0},inplace=True)
        dataset["Cyber_Hacked"].replace({True: 1, False: 0},inplace=True)
        
       # Assigning the value of x and y 
        x=dataset.drop("Cyber_Hacked",axis=1)
        y=dataset["Cyber_Hacked"]
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=size,random_state=1,stratify=y)

        # describes info about train and test set
        print("Number transactions X_train dataset: ", x_train.shape)
        print("Number transactions y_train dataset: ", y_train.shape)
        print("Number transactions X_test dataset: ", x_test.shape)
        print("Number transactions y_test dataset: ", y_test.shape)

    
        print(x_train,x_test)

        return render_template('preprocess.html', msg='Data Preprocessed and It Splits Successfully')
    return render_template('preprocess.html')

@app.route('/model', methods=['POST', 'GET'])
def model():
    if request.method == "POST":
        global model
        print('ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc')
        s = int(request.form['algo'])
        if s == 0:
            return render_template('model.html', msg='Please Choose an Algorithm to Train')
        elif s == 1:
            print('aaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')
            from sklearn.ensemble import AdaBoostClassifier
            ad = AdaBoostClassifier()
            ad.fit(x_train,y_train)
            y_pred = ad.predict(x_train)
            ac_ad = accuracy_score(y_train, y_pred)
            ac_ad = ac_ad * 100
            print('aaaaaaaaaaaaaaaaaaaaaaaaa')
            msg = 'The accuracy obtained by AdaBoost Classifier is  ' + str(ac_ad) + str('%')
            return render_template('model.html', msg=msg)
        elif s == 2:
            classifier = DecisionTreeClassifier(max_leaf_nodes=39, random_state=0)
            classifier.fit(x_train, y_train)
            y_pred  =  classifier.predict(x_train)            
            
            ac_dt = accuracy_score(y_train, y_pred)
            ac_dt = ac_dt * 100
            msg = 'The accuracy obtained by Decision Tree Classifier is ' + str(ac_dt) + str('%')
            return render_template('model.html', msg=msg)
        elif s == 3:
            from sklearn.ensemble import RandomForestClassifier
            rf=RandomForestClassifier(n_estimators = 50,
                                            min_samples_split = 3,
                                            min_samples_leaf = 2,
                                            max_features = 'log2',
                                            max_depth = 10,
                                            bootstrap = True)
            rf.fit(x_train,y_train)
            rf=rf.fit(x_train,y_train)
            y_pred  =  rf.predict(x_train)            
            
            ac_rf = accuracy_score(y_train, y_pred)
            ac_rf = ac_rf * 100
            msg = 'The accuracy obtained by random Forest Classifier is ' + str(ac_rf) + str('%')
            return render_template('model.html', msg=msg)
        elif s == 4:
            from sklearn.neighbors import KNeighborsClassifier
            knn = KNeighborsClassifier(n_neighbors=12)
            knn.fit(x_train, y_train)
            y_pred  =  knn.predict(x_train)            
            
            ac_knn = accuracy_score(y_train, y_pred)
            ac_knn = ac_knn * 100
            msg = 'The accuracy obtained by K-Nearest Neighbour is ' + str(ac_knn) + str('%')
            return render_template('model.html', msg=msg)
        elif s == 5:
            svc = SVC()
            svc.fit(x_train, y_train)
            y_pred  =  svc.predict(x_train)            
            
            ac_svc = accuracy_score(y_train, y_pred)
            ac_svc = ac_svc * 100
            msg = 'The accuracy obtained by support vector Classifier is ' + str(ac_svc) + str('%')
            return render_template('model.html', msg=msg)
        elif s == 6:
            from sklearn.linear_model import LogisticRegression 

            lr = LogisticRegression()
            lr.fit(x_train, y_train)
            y_pred  =  lr.predict(x_train)            
            
            ac_lr = accuracy_score(y_train, y_pred)
            ac_lr = ac_lr * 100
            msg = 'The accuracy obtained by Logistic Regression is ' + str(ac_lr) + str('%')
            return render_template('model.html', msg=msg)
        
        elif s == 7:
            import xgboost as xgb 
            from sklearn.linear_model import LogisticRegression 
            xgboost = xgb.XGBClassifier()
            xgboost.fit(x_train, y_train)
            y_pred = xgboost.predict(x_test)
            ac_xgb = accuracy_score(y_test, y_pred)
            ac_xgb1 = ac_xgb * 100
            msg = 'The accuracy obtained by XGBoost Classifier is ' + str(ac_xgb1) + '%'
            return render_template('model.html', msg=msg)
    return render_template('model.html')

@app.route('/prediction', methods=["GET", "POST"])
def prediction():
    result = None
    if request.method == "POST":
        # f1=int(request.form['city'])
        f1 = float(request.form['Time_of_Breach'])
        f2 = float(request.form['Breach_Type'])
        f3 = float(request.form['Attack_Vector'])
        f4 = float(request.form['Vulnerability_Type'])
        f5 = float(request.form['Targeted_System'])
        f6 = float(request.form['Data_Exfiltrated'])
        f7 = float(request.form['Attack_Successful'])
        f8 = float(request.form['Attack_Duration_Hours'])
        f9 = float(request.form['Damage_Cost_Dollars'])
        f10 = float(request.form['Incident_Response_Time_Minutes'])
        f11 = float(request.form['User_Notified'])
        f12 = float(request.form['Legal_Action_Taken'])
        
        
        print(f2)
        print(type(f2))

        li = [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12]
        print(li)
        
        # model.fit(X_transformed, y_train)
        
        # print(f2)
        # import pickle
        
        rf = RandomForestClassifier()
        model = rf.fit(x_train, y_train)
        result = model.predict([li])
        print(result)
        print('result is ',result)
        # (Normal  = 0,   Cyber_Hacked  = 1 )
        if result == 0:
            result = 'There is  No-Cyber Hacked'
        else:
            result = 'There is Cyber Hacked'
        

    from sklearn.preprocessing import LabelEncoder

    # Load the dataset
    df = pd.read_csv("cyberhacking_dataset.csv")

    # Drop columns
    columns_to_drop = ['Cyber_Hacked']
    df = df.drop(columns=columns_to_drop)

    # Replace spaces in column names with underscores
    df.columns = [re.sub(r'\s+', '_', col) for col in df.columns]

    # Define object columns to be encoded
    object_columns = df.select_dtypes(include=['object']).columns

    # Store label counts before encoding
    labels = {col: df[col].value_counts().to_dict() for col in object_columns}

    # Initialize LabelEncoder
    le = LabelEncoder()

    # Encode categorical columns and store the encoded value counts
    encodes = {}
    for col in object_columns:
        df[col] = le.fit_transform(df[col])
        value_counts = df[col].value_counts().to_dict()
        encodes[col] = value_counts

    obj_dic = {}

    for key in labels.keys():
        obj_dic[key] = []
        for sub_key, value in labels[key].items():
            for id_key, id_value in encodes[key].items():
                if value == id_value:
                    obj_dic[key].append((sub_key, id_key))
                    break

    # Define Boolean columns to be encoded
    object_columns = df.select_dtypes(include=['bool']).columns

    # Store label counts before encoding
    labels = {col: df[col].value_counts().to_dict() for col in object_columns}

    # Initialize LabelEncoder
    le = LabelEncoder()

    # Encode categorical columns and store the encoded value counts
    encodes = {}
    for col in object_columns:
        df[col] = le.fit_transform(df[col])
        value_counts = df[col].value_counts().to_dict()
        encodes[col] = value_counts

    bool_dic = {}

    for key in labels.keys():
        bool_dic[key] = []
        for sub_key, value in labels[key].items():
            for id_key, id_value in encodes[key].items():
                if value == id_value:
                    bool_dic[key].append((sub_key, id_key))
                    break

    return render_template('prediction.html', obj_data=obj_dic, bool_data=bool_dic, msg=result)










if __name__=='__main__':
    app.run(debug=True)