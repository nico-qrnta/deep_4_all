# Compte Rendu : Méthodologie de Création d'un Dataset Synthétique (TP4)

Ce document résume la démarche et les outils utilisés pour générer un dataset de qualité destiné à l'entraînement d'un modèle étudiant spécialisé dans la draft de League of Legends.

---

## 1. Ingénierie des Prompts et Diversification

La première étape a consisté à définir une identité et un cadre de réponse pour l'IA (le "Teacher").

*   **System Prompt Expert** : Nous avons configuré l'IA comme un coach professionnel de haut niveau, avec des instructions strictes sur l'ordre officiel de la draft (17 étapes) et les contraintes de rôles (top, jungle, mid, adc, support).
*   **Formatage Structuré** : L'utilisation de schémas JSON stricts a permis d'extraire des données prêtes à l'emploi (picks, bans, raisonnements) sans texte superflu.
*   **Adaptation Iterative** : Les prompts ont été affinés via des tests à petite échelle pour garantir que l'IA propose plusieurs options viables (optimales, moyennes, risquées) plutôt qu'une seule réponse unique.

## 2. Préparation des Données Source (Mocking)

Au lieu de partir de zéro, nous avons utilisé un dataset existant (`matchData.csv`) pour servir de base réaliste.

*   **Transformation de Situations** : Un script de prétraitement sélectionne des lignes du CSV pour "mocker" des situations de draft réelles.
*   **Styles de Draft Aléatoires** : Nous avons implémenté différents styles de génération (full draft, random step, etc.) pour exposer le modèle futur à une grande variété de contextes de jeu.

## 3. Optimisation de la Génération (Async & Température)

Pour accélérer la création du dataset tout en garantissant la diversité des réponses :

*   **Requêtes Asynchrones (`asyncio`)** : Nous avons parallélisé les appels API. Pour chaque situation de draft, le système lance simultanément deux requêtes.
*   **Dualité de Température** : 
    *   **Basse température (0.3)** : Pour obtenir des décisions stables et académiques.
    *   **Haute température (0.9)** : Pour explorer des stratégies plus créatives ou risquées.
*   **Sauvegarde Incrémentale** : Le dataset est sauvegardé toutes les 5 requêtes pour éviter toute perte de données en cas d'interruption.

## 4. Évaluation et Filtrage Qualitatif

Une attention particulière a été portée à la qualité intrinsèque des données générées.

*   **Score de Confiance (Logprobs)** : À chaque réponse, nous extrayons les probabilités logarithmiques (`logprobs`) fournies par l'API. Nous calculons ensuite une note de confiance (0-100) basée sur l'exponentielle de la moyenne des logprobs.


