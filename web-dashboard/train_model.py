import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import sys

# Terminalde utf-8 sorunu yasamamak icin
sys.stdout.reconfigure(encoding='utf-8')

def train_and_evaluate():
    print("Veri seti yukleniyor...")
    df = pd.read_csv('final_merged_dataset.csv')

    # Eksik veya bozuk satirlari temizle
    df = df.dropna()

    # 1. Ozellikleri (Features) ve Hedefi (Target) ayir
    X = df[['ndvi', 'ndmi', 'land_cover']].copy()
    y = df['label']

    # 2. Categorical Variable (Land Cover) icin One-Hot Encoding
    # GEE Dynamic World sınıfları int olarak gelir. Bunları String yapıp kategorik ayırıyoruz.
    X['land_cover'] = X['land_cover'].astype(int).astype(str)
    X = pd.get_dummies(X, columns=['land_cover'], prefix='LC')

    # 3. Veriyi Train / Test Olarak Bol (%80 Train, %20 Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ---------------------------------------------------------
    # MODEL 1: LOJISTIK REGRESYON (Tezdeki Denklem Icin)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print(" MODEL 1: LOJISTIK REGRESYON (YANICILIK DENKLEMI) ")
    print("="*50)
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_train)

    # Test Basarisi
    y_pred_log = log_model.predict(X_test)
    acc_log = accuracy_score(y_test, y_pred_log)
    print(f">> Dogruluk Orani (Accuracy): %{acc_log*100:.2f}")

    # Katsayilari Ekrana Bas
    print("\n[ MATEMATIKSEL KATSAYILAR (Ağırlıklar) ]")
    coefficients = log_model.coef_[0]
    features = X.columns
    sabit = log_model.intercept_[0]

    print(f"Sabit Deger (Intercept): {sabit:.4f}")
    for feature, coef in zip(features, coefficients):
        print(f"{feature}: {coef:.4f}")

    # ---------------------------------------------------------
    # MODEL 2: RANDOM FOREST (Yuksek Basari Icin)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print(" MODEL 2: RANDOM FOREST (KARAR AGACLARI) ")
    print("="*50)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    y_pred_rf = rf_model.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f">> Dogruluk Orani (Accuracy): %{acc_rf*100:.2f}")

    # Feature Importance (Hangi degisken yangini tetikliyor?)
    print("\n[ OZELLIK ONEM DERECELERI (Feature Importance) ]")
    importances = rf_model.feature_importances_
    
    # Buyukten kucuge siralamak icin
    imp_dict = {f: i for f, i in zip(features, importances)}
    sorted_imp = sorted(imp_dict.items(), key=lambda item: item[1], reverse=True)
    
    for feature, imp in sorted_imp:
        print(f"{feature}: %{imp*100:.2f}")

    # ---------------------------------------------------------
    # MODEL KAYDI (Dron Entegrasyonu Icin)
    # ---------------------------------------------------------
    # Gerekli tum sutun isimlerini kaydediyoruz, boylece dron havadayken ayni formati olusturabilir.
    model_data = {
        'model': log_model, # Lojistik Regresyon daha stabil yuzdeler verir
        'features': list(X.columns)
    }
    joblib.dump(model_data, 'fuel_scorer_model.pkl')
    print("\n" + "*"*50)
    print(">>> 'fuel_scorer_model.pkl' BASARIYLA KAYDEDILDI! <<<")
    print("*"*50)

if __name__ == "__main__":
    train_and_evaluate()
