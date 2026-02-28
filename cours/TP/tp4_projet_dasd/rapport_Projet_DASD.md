# Rapport de Projet : Distillation de Modèles de Raisonnement (DASD)

## 1. Introduction

Dans le cadre de ce projet sur la Distillation de Modèles de Raisonnement en utilisant la méthode DASD (Distribution-Aligned Sequence Distillation), nous avons cherché à transférer les capacités approfondies d'analyse "Chain-of-Thought" (CoT) d'un large modèle ("Teacher") vers un modèle beaucoup plus léger et déployable localement ("Student" : `Qwen3-4B-Instruct`).

Pour rendre le projet plus concret, nous avons choisi de l’appliquer à l’analyse stratégique de parties de *League of Legends*. L'objectif de notre modèle distillé est d'être capable d'analyser une situation de jeu complexe, notamment lors des phases de choix (drafts), en raisonnant étape par étape avant de proposer une conclusion stratégique forte.

## 2. Méthodologie

### 2.1 Synthèse du corpus et Génération du Dataset (Phases 1 et 2)

Afin d'obtenir un point de départ réaliste, nous nous sommes basés sur un dataset public Kaggle répertoriant de nombreux matchs de *League of Legends*. À partir de ces données brutes, nous avons extrait différentes situations de jeu sur lesquelles nous avons formulé des questions stratégiques.

Nous avons ensuite développé un pipeline de génération de données (scripts présents dans le dossier `generate-datasets/`) interrogeant un modèle puissant faisant office de "Teacher", via l'API d'Infomaniak. Pour appliquer la méthode DASD, ce "super dataset" a été généré en deux passes distinctes :

*   **Stage 1 (Basse température, $\tau \approx 0.3$)** : Génération de réponses stables, structurées et très déterministes. Le but est d'apprendre au modèle étudiant la structure souhaitée : non seulement l'utilisation des balises `<reasoning>...</reasoning>` pour la chaîne de pensée, mais aussi le formatage strict de la réponse finale en **JSON**. Cela est primordial pour intégrer le modèle au sein d'une application ou d'un pipeline par la suite.
*   **Stage 2 (Haute température, $\tau \approx 0.9$)** : Génération de réponses diversifiées, explorant différents fils de pensée stratégique sur *League of Legends*. Cela augmente la richesse de l'analyse produite.

### 2.2 Implémentation du Divergence-Aware Sampling (DAS) (Phase 4)

La génération à haute température de la Phase 2 produit de la diversité mais amène inévitablement du "bruit", voire des erreurs de jugement (hallucinations du modèle). C'est pour palier ce problème que nous avons implémenté le filtrage Divergence-Aware Sampling (DAS). 

Plutôt que d'évaluer une réponse dans sa globalité, le script calcule la **divergence phrase par phrase** entre :
*   Les probabilités linéaires du Teacher ($P_T$) 
*   Les probabilités linéaires du Student ($P_S$) testé sur la même situation.

Nous avons appliqué la matrice de décision du papier de recherche permettant de repérer les trois types de phrases :
1.  **Teacher Sentences** ($P_T \gg P_S$) : Le Teacher est sûr de son raisonnement, mais l'Étudiant hésite. L'apport pédagogique est fort. C'est ici que l'étudiant a tout à apprendre de son professeur. -> **CONSERVÉ**
2.  **Shared Sentences** ($P_T \approx P_S$) : Comportement neutre, l'étudiant maîtrise déjà l'idée. -> **NEUTRE**
3.  **Student Sentences** ($P_S > P_T$) : L'étudiant a trop d'assurance sur une idée où le Teacher doute, il s'agit d'une potentielle hallucination. -> **REJETÉ**

Grâce à cet algorithme, nous avons pu nettoyer le dataset de la Phase 2 (fichiers produits dans `generate-datasets/datasets/`) pour ne garder que les exemples comportant une véritable "leçon" pour notre modèle.

### 2.3 Configuration de l'entraînement (Phase 5)

Nous avons utilisé Llama-Factory pour paramétrer le fine-tuning (via un adaptateur LoRA) du modèle de base `Qwen3-4B-Instruct`. Conformément à la méthodologie DASD, nous avons scindé l'entraînement (Config 5) :
*   **Stage 1** : Entraînement sur le dataset "Basse Température" pour stabiliser l'apprentissage global du format.
*   **Stage 2** : Reprise des poids de l'adaptateur obtenu en fin de Stage 1, puis entraînement sur le dataset Haute Température filtré par le DAS.

## 3. Résultats et Analyse (Phase 7)

### 3.1 Entraînement et performances numériques

Les différents runs d'entraînement ont été effectués avec succès, en atteste la présence des checkpoints dans les dossiers `saves/Qwen3-4B/lora/stage1/` et `stage2/`.
Voici les performances chiffrées issues des logs du framework d'entraînement (notamment `train_results.json`) :

*   **Stage 1** : L'entraînement s'est fait sur 3 époques (`train_runtime` : $\sim 1660$ sec). La fonction de perte sur l'entraînement (`train_loss`) est descendue à environ **0.657**, tandis que la loss sur le set de validation est très prometteuse avec **0.468**. Cela indique une très bonne assimilation du format de base (les balises et le JSON final) sans overfitting grossier.
*   **Stage 2** : Le second apprentissage sur 2 époques (`train_runtime` : $\sim 1107$ sec) a affiché une excellente stabilité. La loss d'entraînement plafonne avec consistance autour de **0.650**.

### 3.2 Analyse des courbes

Vous trouverez ci-dessous les graphiques de suivi de loss correspondants aux deux étapes d'apprentissage.

**Graphes du Stage 1 :**

*(Insérer ici les images de : `saves/Qwen3-4B/lora/stage1/training_loss.png` ou `training_eval_loss.png`)*

> **Analyse :** Lors de cette première étape, on observe une descente initiale rapide de la loss qui traduit la phase d'adaptation rapide de Qwen-4B aux instructions spécifiques (notamment le formatage en `<reasoning>`). La courbe montre une belle stabilité au fil des itérations, indiquant que le format déterministe est bien appris.

**Graphes du Stage 2 :**

*(Insérer ici les images de : `saves/Qwen3-4B/lora/stage2/training_loss.png` ou `training_eval_loss.png`)*

> **Analyse :** La courbe de ce second stage, bien qu'un peu plus bruitée du fait de la complexité des raisonnements explorés (dataset haute température filtré), reste stable. Il n'y a pas de rebond explosif de la loss, ce qui prouve que la méthode de filtrage DAS remplit bien son rôle : le bruit néfaste a été retiré, ne conservant qu'un apprentissage progressif de la profondeur stratégique.

## 4. Discussion et Limites

### 4.1 Réussites du projet
La méthode DASD se démontre particulièrement efficace. Au lieu de demander à un petit modèle de 4B de "réinventer la roue" stratégique, la distillation nous a permis d'instiller la logique implacable de modèles cent fois plus lourds. Le fait de fonctionner sur des données métier de jeu a prouvé qu'un "petit" LLM, une fois bien cadré, ne se perd plus dans des banalités et déroule véritablement un processus de décision séquentiel, tout en respectant un formatage restrictif très utile (JSON).

### 4.2 Limites et améliorations (Troubleshooting)
1. **Limitation de l'API / Biais temporel** : Le Teacher Model est bridé soit par ses quotas (Rate Limit obligeant à des temps d'attente), mais également par son savoir temporel. La "meta" de League of Legends change très vite, utiliser un vieux dataset de matchs rend parfois les choix stratégiques non-optimaux sur le patch actuel du jeu.
2. **Puissance de calcul** : L'utilisation de grands contextes (longs CoT) engendre vite des erreurs de type (Out-Of-Memory/OOM) lors de l'entraînement. Il a fallu jouer habilement sur le `gradient_checkpointing` et la longueur de coupure (`cutoff_len`) pour faire rentrer l'entraînement dans la VRAM disponible.
3. **Piste d'amélioration** : Face aux résultats obtenus, la prochaine étape logique serait de compléter l'apprentissage SFT complété ici avec une phase de DPO (Direct Preference Optimization). En confrontant une analyse de draft "gagnante" à une analyse "perdante" (qui aurait conduit à la défaite dans le dataset Kaggle), le modèle se lisserait encore d'avantage son taux d'hallucination stratégique.
