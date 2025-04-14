# 🌍 Applications concrètes d’un modèle de captioning d’images africaines

Ce modèle, capable de générer automatiquement des descriptions textuelles pour des images de scènes africaines, peut être appliqué dans plusieurs domaines essentiels au quotidien, à la recherche, au commerce et à l’innovation.

---

## 💡 1. Accessibilité et inclusion

**Objectif** : Offrir des descriptions d’images aux personnes malvoyantes.

- 🧑‍🦯 Exemple : un utilisateur prend une photo d’un marché, et l’application lit :  
  *“Des femmes vendent des tomates, piments et ignames sous un hangar en bois.”*

- ✅ **Impact** : améliore l’autonomie des personnes en situation de handicap visuel, en leur fournissant un aperçu verbal de leur environnement.

---

## 🛍️ 2. E-commerce local et mode africaine

**Objectif** : Générer automatiquement des descriptions produits pour les vendeurs africains.

- 👗 Exemple : *“Robe en tissu pagne bogolan, motifs géométriques africains, manches courtes.”*
- 🔍 Application : plateformes de e-commerce, recherche visuelle, filtres de recherche intelligente.

- ✅ **Impact** : 
  - Gain de temps dans la mise en ligne de produits.
  - Valorisation de la mode africaine.
  - Recommandation personnalisée pour les acheteurs.

---

## 🗺️ 3. Agriculture et environnement

**Objectif** : Analyser des paysages agricoles ou ruraux à partir d’images.

---

## 🏙️ 4. Urbanisme et Smart Cities

**Objectif** : Comprendre la dynamique des scènes urbaines en Afrique.

---

## 🍛 5. Tourisme, culture et foodtech

**Objectif** : Identifier et décrire les plats culinaires africains.

- 🍲 Exemple : *“Assiette de watché accompagnée de fritures et deux oeufs.”*

- ✅ **Impact** :
  - Améliore les applications de recommandations touristiques.
  - Permet une base de données culinaire africaine.
  - Potentiel pour des analyses nutritionnelles ou de santé.

---

## 🧠 6. Pré-annotation de jeux de données

**Objectif** : Préparer automatiquement des annotations textuelles pour des datasets.

- 📚 Exemple : Description automatique de milliers d’images rurales ou urbaines pour créer un jeu d'entraînement IA.

- ✅ **Impact** :
  - Réduction du coût humain de l’annotation.
  - Accélération des projets de recherche en IA dans le contexte africain.

---

## 🎓 7. Outils éducatifs et culturels

**Objectif** : Apporter du contexte aux images dans des applications éducatives.

- 📖 Exemple : *“Groupe d’enfants en uniforme jouant sous un arbre dans une cour d’école.”*

- ✅ **Impact** :
  - Apprentissage contextualisé pour les élèves.
  - Création de livres éducatifs intelligents illustrés.

---


# 📦 Datasets collectés et stratégie d’annotation pour le modèle de captioning

## 📁 1. Datasets ciblés

Voici quelques sources que j’ai identifiées pour collecter des images de scènes africaines :

| Nom du dataset | Source | Contenu | Statut |
|----------------|--------|---------|--------|
| **African attire** | [HuggingFace](https://huggingface.co/datasets/inuwamobarak/african-atire) | Styles et habillement traditionnel: 'Adire', 'Idama', 'Idgo', 'Idoma', 'Igala', 'Igbo', 'Tiv', 'Tswana-Shweshwe', 'Xhosa-South Africa', 'Zulu' | Téléchargé ✅ |
| **African Foods Datasets** | [Mendeley Data](https://data.mendeley.com/datasets/rrzhwbg3kw/2) | 6 different foods from Ghana and Cameroon | Téléchargé ✅ |
| **Nigerian Food Dataset** | [Mendeley Data](https://data.mendeley.com/datasets/2vktdxfxv7/2) | 10 different foods from Nigeria | Téléchargé ✅ |
| **AfricanWax Patterns 5KDataset** | [HuggingFace](https://huggingface.co/datasets/paceailab/AfricanWaxPatterns_5KDataset) | Motifs des tissu Wax | En cours ⏳ |
| **AfricanWax Patterns 5KDataset** | [HuggingFace](https://huggingface.co/datasets/paceailab/AfricanWaxPatterns_5KDataset) | Motifs des tissu Wax | En cours ⏳ |

---

## ✍️ 2. Stratégie d’annotation

### 🌟 Objectif :
Créer des **descriptions naturelles**, riches et spécifiques au **contexte africain**, à partir des images, pour entraîner un modèle de **captioning supervisé**.

### 🧰 Outils utilisés pour l’annotation:

- Utilisation d’un modèle existant de captioning pour pré-annoter 
- Formatage JSON ou CSV standard pour l’entraînement

---

## 📌 3. Exemple d'annotation

### 📝 Simulation de Caption

Imaginons ce à quoi pourrait ressembler chaque description dans chaque catégorie.

---

#### 🧵 Mode vestimentaire

**Image :** Femme en tenue traditionnelle  
**Description :**  
> "Une femme portant une robe en tissu wax aux couleurs vives, avec un foulard assorti, debout devant une maison en terre battue."

**Image :** Hommes à un festival culturel  
**Description :**  
> "Deux hommes vêtus de boubous brodés participant à une fête communautaire."

---

#### 🏘️ Paysages ruraux

**Image :** Scène de village  
**Description :**  
> "Un village africain rural avec des cases aux toits de chaume, entouré de savane sèche et d’enfants jouant."

**Image :** Agriculteur dans un champ  
**Description :**  
> "Un homme labourant un champ avec une paire de bœufs dans une communauté agricole rurale."

---

#### 🛒 Marchés

**Image :** Marché en plein air  
**Description :**  
> "Des femmes vendant des tomates, des piments et des légumes-feuilles sous des parasols colorés dans un marché en plein air."

**Image :** Rue animée  
**Description :**  
> "Rue bondée d’un marché africain avec des vendeurs, des brouettes et des clients négociant les prix des produits."

---

#### 🍲 Cuisine africaine

**Image :** Assiette de foutou et soupe  
**Description :**  
> "Un repas traditionnel ghanéen composé de foutou accompagné d’une soupe légère et de morceaux de viande de chèvre."

**Image :** Thiéboudienne  
**Description :**  
> "Un plat sénégalais de riz, poisson et légumes servi dans une grande assiette partagée."

# 📦 Fusion des datasets : Fashion & Cuisine Africaines

## 1. Objectif
Créer un dataset unifié pour entraîner un modèle de *captioning* sur des images africaines dans deux catégories :
- 🧵 Fashion (vêtements traditionnels, motifs wax)
- 🍲 Cuisine (plats nigérians, ghanéens et camerounais)

---

## 2. Datasets sources

### 🧵 Fashion
- **african-attire**
  - Format : `(image, label)`
  - Labels : 10 groupes ethniques (ex. Yoruba, Zulu, Ashanti…)
- **wax-patterns**
  - Format : `(image)`
  - Images de motifs textiles wax (pas de labels)

### 🍲 Cuisine
- **nigerian-dishes**
  - Format : `(image, label)`
  - 10 catégories (ex. jollof rice, egusi soup, suya…)
- **ghanaian-cameroonian-dishes**
  - Format : `(image, label)`
  - 6 catégories (ex. thieboudienne, eru, banku…)

---

## 3. Format cible du dataset fusionné

Chaque élément doit suivre le format suivant :
```json
{
  "image_path": "path/to/image.jpg",
  "category": "fashion" | "food",
  "sub_category": "Yoruba attire" | "Ankara pattern" | "Egusi soup" | "Thieboudienne", 
  "caption": "Une femme portant une robe traditionnelle yoruba en tissu wax coloré."
}
```

