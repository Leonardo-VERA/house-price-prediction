# Prédiction du Prix des Maisons avec LightGBM

Projet de machine learning pour prédire le prix des maisons à partir de caractéristiques structurelles et d'équipements.

**Dataset:** 546 maisons avec 12 features (lotsize, bedrooms, bathrms, airco, prefarea, etc.)

**Résultat final:** R² = 0.6301 avec LightGBM optimisé

---

## Méthodologie

```
Structure → EDA → Nettoyage → Feature Engineering → Entraînement → Optimisation → Diagnostic
```

---

## 1. Structure du Projet

```
.
├── data/
│   ├── raw/              # Housing.csv original
│   ├── processed/        # housing_clean.csv (one-hot encoded)
│   └── interim/
├── models/               # Modèles entraînés (.pkl)
├── notebooks/            # Analyses et entraînements
│   ├── 0.1-eda-initial-exploration.ipynb
│   ├── 0.2-feature-engineering.ipynb
│   ├── 1.0-train-lightgbm.ipynb
│   ├── 1.1-optimize-lightgbm.ipynb
│   └── DIAGNOSTIC-model-limits.ipynb
├── src/models/           # Scripts de prédiction
├── requirements.txt
└── README.md
```

---

## 2. Installation

```bash
# Créer l'environnement virtuel
python -m venv .venv

# Activer l'environnement
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate     # Windows

# Installer les dépendances
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm jupyter

# Sauvegarder les dépendances
pip freeze > requirements.txt
```

---

## 3. Exploration des Données (EDA)

### Dataset: Housing.csv

| Caractéristique | Détails |
|----------------|---------|
| Taille | 546 maisons |
| Features | 12 variables (7 numériques, 6 binaires) |
| Target | `price` (25,000$ - 190,000$, moyenne: 68,122$) |
| Valeurs manquantes | 0 |
| Duplicatas | 0 |

### Variables les plus corrélées avec le prix

| Variable | Corrélation | Impact |
|----------|-------------|--------|
| `lotsize` | 0.54 | Très fort |
| `bathrms` | 0.52 | Très fort |
| `airco` | 0.45 | Fort |
| `stories` | 0.42 | Fort |
| `garagepl` | 0.38 | Modéré |

**Conclusion:** Les variables catégoriques (`airco`, `driveway`, `prefarea`) ont un impact significatif sur le prix.

---

## 4. Feature Engineering

### Transformation appliquée: One-Hot Encoding

Les 6 variables binaires (yes/no) ont été encodées:
- `driveway_yes`, `recroom_yes`, `fullbase_yes`, `gashw_yes`, `airco_yes`, `prefarea_yes`

**Technique utilisée:**
```python
pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
```

- `drop_first=True`: Évite la multicolinéarité
- Résultat: 11 features finales (après suppression de `rownames`)

---

## 5. Choix du Modèle: Pourquoi LightGBM?

### Alternatives écartées

| Modèle | Pourquoi NON |
|--------|--------------|
| **Régression linéaire** | Trop simple, suppose des relations linéaires |
| **Régression polynomiale** | Instable avec plusieurs variables |
| **SVR** | Lent, difficile à calibrer, rarement meilleur que LightGBM |

### LightGBM: Évolution des arbres de décision

**Arbre généalogique:**

1. **Arbre de décision** → Un seul arbre (simple mais imprécis)
2. **Random Forest** → 100+ arbres indépendants en parallèle (moyenne des résultats)
3. **Gradient Boosting (LightGBM)** → Arbres séquentiels où chaque arbre corrige les erreurs du précédent

**Avantages de LightGBM:**
- Capture les relations non linéaires (ex: airco a plus d'impact dans les quartiers premium)
- Utilise plusieurs variables simultanément
- Technique d'ensemble robuste

---

## 6. Entraînement des Modèles

### Modèle V1: Baseline

**Configuration:**
- Algorithme: LightGBM avec paramètres par défaut
- Transformation: Aucune
- Features: 11 variables

**Résultats:**
| Métrique | Valeur |
|----------|--------|
| R² | 0.6118 |
| MAE | $11,884.73 |
| RMSE | $16,103.94 |

---

### Modèle V2: Optimisé

**Optimisations appliquées:**

1. **Transformation logarithmique:**
   ```python
   y_log = np.log1p(y)  # Lisse la distribution des prix
   ```

2. **Grid Search avec 5-Fold Cross-Validation:**
   - 32 combinaisons testées
   - 160 entraînements (32 × 5 folds)

**Hyperparamètres optimisés:**

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| `learning_rate` | 0.05 | Apprentissage lent = meilleure généralisation |
| `max_depth` | 3 | Arbres simples pour dataset petit |
| `num_leaves` | 10 | Peu de feuilles = évite l'overfitting |
| `n_estimators` | 200 | Plus d'arbres = meilleure correction |
| `min_child_samples` | 20 | Minimum par feuille (conservateur) |

**Résultats:**
| Métrique | Valeur | Amélioration |
|----------|--------|--------------|
| R² | **0.6301** | +1.83% |
| MAE | **$11,707.48** | -$177 |
| RMSE | $15,720.08 | -$384 |

---

## 7. Comparaison des Modèles

| Modèle | MAE | RMSE | R² | Note |
|--------|-----|------|----|------|
| **LightGBM V1 (Baseline)** | $11,884.73 | $16,103.94 | 0.6118 | |
| **LightGBM V2 (Optimisé)** | $11,707.48 | $15,720.08 | **0.6301** | Meilleur |
| **Random Forest** | $12,346.68 | $16,620.37 | 0.5865 | -4.36% vs V2 |

**Conclusion:** Contrairement à l'hypothèse initiale, Random Forest n'a PAS surpassé LightGBM sur ce dataset.

---

## 8. Test sur Maison Témoin

Configuration de la maison de test:
- Lotsize: 6,000 m²
- Bedrooms: 3, Bathrooms: 2, Stories: 2
- Garage: 1 place
- Équipements: Driveway ✓, Sous-sol ✓, Airco ✓, Prefarea ✓

**Prédictions:**

| Modèle | Prédiction | Différence |
|--------|------------|------------|
| V1 (Baseline) | $123,996.34 | - |
| V2 (Optimisé) | $110,268.75 | -$13,727 (-11%) |

Le modèle optimisé est plus conservateur grâce à la transformation logarithmique.

---

## 9. Diagnostic: Le Modèle Peut-il S'Améliorer?

### Tests effectués dans `DIAGNOSTIC-model-limits.ipynb`

#### 9.1 Courbes d'apprentissage

```
R² train:  0.7635
R² CV:     0.6228
Écart:     0.1407 (14%)
```

**Diagnostic:** Équilibre acceptable, pas d'overfitting grave.

---

#### 9.2 Analyse des résidus

```
Moyenne:    $1,674.75  (≈ 0, excellent)
Écart-type: $15,156.37
Min:        -$43,099.63
Max:        $56,100.96
```

**Diagnostic:** Résidus bien centrés, distribution relativement aléatoire.

---

#### 9.3 Random Forest testé

| Modèle | R² | Différence vs V2 |
|--------|----|--------------------|
| LightGBM V2 | 0.6301 | - |
| Random Forest | 0.5865 | **-4.36%** |

**Conclusion:** Random Forest est moins performant que LightGBM sur ce dataset.

---

#### 9.4 Feature Engineering avancé

**5 nouvelles features testées:**
- `bath_per_bedroom` (ratio)
- `total_amenities` (somme des équipements)
- `lotsize_per_bedroom` (densité)
- `premium_combo` (airco × prefarea)
- `size_indicator` (garagepl × stories)

**Résultat:**
```
R² avec nouvelles features: 0.6180
Amélioration vs V2: -1.21%
```

**Diagnostic:** Les nouvelles features ajoutent du bruit au lieu d'améliorer le modèle.

---

#### 9.5 Top 10 des pires prédictions

```
Erreur moyenne: $33,929.15
Erreur en %: 36.1%
```

**Conclusion:** 10 maisons atypiques que le modèle ne peut pas prédire correctement avec les données actuelles.

---

## 10. Conclusion Finale

### Le Modèle V2 est à sa Limite

**Preuves:**
- Random Forest: -4.36% pire
- Feature Engineering: -1.21% pire
- Résidus bien distribués
- Balance train/test acceptable

### Pourquoi R² = 0.63 est le Maximum?

| Limitation | Impact |
|------------|--------|
| **Trop peu de données** | 546 maisons (LightGBM conçu pour des millions) |
| **Features manquantes critiques** | Année de construction, m² construits, état du bien |
| **Pas de localisation précise** | Seulement "prefarea" binaire |

### Pour Améliorer Significativement il Faudrait:

1. **Plus de données:** Minimum 1,000-2,000 maisons (idéal: 5,000+)
2. **Meilleures features:**
   - Année de construction
   - Surface construite (m²) - actuellement seulement `lotsize` (terrain)
   - Coordonnées GPS ou code postal
   - État de la maison (rénové/ancien)
   - Distance au centre-ville

---

## 11. Utilisation

### Prédire avec le Modèle V1 (Baseline)

```bash
python src/models/predict_house_v1.py
```

### Prédire avec le Modèle V2 (Optimisé)

```bash
python src/models/predict_house.py
```

---

## 12. Notebooks Disponibles

| Notebook | Description |
|----------|-------------|
| `0.1-eda-initial-exploration.ipynb` | Analyse exploratoire, visualisations, corrélations |
| `0.2-feature-engineering.ipynb` | One-Hot Encoding, création du dataset final |
| `1.0-train-lightgbm.ipynb` | Entraînement du modèle baseline (V1) |
| `1.1-optimize-lightgbm.ipynb` | Optimisation hyperparamètres + transformation log |
| `DIAGNOSTIC-model-limits.ipynb` | Tests avancés: RF, feature engineering, limites |

---

## 13. Technologies Utilisées

- **Python 3.10**
- **Pandas / NumPy**: Manipulation de données
- **Matplotlib / Seaborn**: Visualisations
- **Scikit-learn**: Métriques, validation croisée, train/test split
- **LightGBM**: Algorithme de Gradient Boosting
- **Jupyter**: Notebooks interactifs

---

## 14. Résultats Visuels

Les graphiques suivants sont disponibles dans les notebooks:

### EDA (0.1-eda-initial-exploration.ipynb)
- Distribution des prix (histogramme + KDE)
- Heatmap de corrélations
- Boxplots: prix vs variables catégoriques

### Entraînement (1.0-train-lightgbm.ipynb)
- Scatter plot: prix réel vs prédit
- Feature importance

### Diagnostic (DIAGNOSTIC-model-limits.ipynb)
- Courbes d'apprentissage (train vs CV)
- 4 graphiques d'analyse des résidus
- Comparaison top 10 pires prédictions

---

## 15. Auteur et Licence

**Projet:** Prédiction du Prix des Maisons
**Context:** 2025-DSP-cours / Module 03 PYTHON
**Dataset:** Housing.csv (546 maisons)

**Note Finale du Projet:** R² = 0.6301 représente le maximum atteignable avec ce dataset limité.
