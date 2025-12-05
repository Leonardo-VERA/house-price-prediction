import joblib
import pandas as pd
import numpy as np

model_path = 'models/lightgbm_house_prices_v1.pkl' 

try:
    model = joblib.load(model_path)
    print("✅ Model chargé avec success")
except:
    print("❌ Erreur: je trouve pas le ficher plk.")
    exit()


new_maison = pd.DataFrame({
    'lotsize': [6000],       # Grand terrain
    'bedrooms': [3],         # 3 chambres
    'bathrms': [2],          # 2 salles de bain
    'stories': [2],          # 2 étages
    'garagepl': [1],         # 1 garage
    'driveway_yes': [1],     # Allée pavée (1=Oui, 0=Non)
    'recroom_yes': [0],      # Pas de salle de jeux
    'fullbase_yes': [1],     # Sous-sol
    'gashw_yes': [0],        # Pas de chauffe-eau à gaz
    'airco_yes': [1],        # Oui, climatisation (Important)
    'prefarea_yes': [1]      # Emplacement de choix
})

print("\n--- 🏠 Caractéristiques de la maison à évaluer ---")
print(new_maison)

# predire
prix_estime = model.predict(new_maison)[0]

print(f"\n💰 ESTIMATION DE PRIX DE VENTE: ${prix_estime:,.2f}")
